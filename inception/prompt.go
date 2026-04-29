package inception

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/log"
)

const maxTokensCeiling = 50000

func toMessage(s llm.Message) Message {
	content := []MessageContent{}
	if s.Content != "" {
		content = append(content, MessageContent{
			Type: "text",
			Text: s.Content,
		})
	}

	return Message{
		Role:       s.Role,
		Content:    content,
		ToolCalls:  s.ToolCalls,
		ToolCallId: s.ToolCallId,
	}
}

type Message struct {
	Role       string           `json:"role"`
	Content    []MessageContent `json:"content"`
	ToolCalls  json.RawMessage  `json:"tool_calls,omitempty"`
	ToolCallId string           `json:"tool_call_id,omitempty"`
}

type MessageContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ResponseFormat struct {
	Type string `json:"type"`
}

// Note: Inception chat enforces a temperature floor of 0.5 (range 0.5–1.0).
// llm.Options has no Temperature field today, so we omit the field and let
// the server default (0.75) apply. If a Temperature field is added later,
// clamp to [0.5, 1.0] before sending.
type InferenceRequest struct {
	Model           string         `json:"model"`
	Messages        []Message      `json:"messages"`
	MaxTokens       int            `json:"max_tokens,omitempty"`
	ResponseFormat  ResponseFormat `json:"response_format"`
	Stream          bool           `json:"stream"`
	Tools           []llm.Tool     `json:"tools,omitempty"`
	ToolChoice      string         `json:"tool_choice,omitempty"`
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
}

func createRequest(stream bool, model string, messages []llm.Message, options llm.Options) (io.ReadCloser, error) {
	bodyMessages := make([]Message, len(messages))
	for idx, msg := range messages {
		bodyMessages[idx] = toMessage(msg)
	}

	responseFormat := "text"
	if options.ResponseFormat != "" {
		responseFormat = string(options.ResponseFormat)
	}

	maxTokens := min(options.MaxTokens, maxTokensCeiling)

	reqBody := InferenceRequest{
		Stream:         stream,
		Model:          model,
		Messages:       bodyMessages,
		ResponseFormat: ResponseFormat{responseFormat},
		Tools:          options.Tools,
		MaxTokens:      maxTokens,
	}

	if len(options.Tools) > 0 {
		reqBody.ToolChoice = "auto"
	} else {
		reqBody.ReasoningEffort = reasoningEffort(options.Thinking)
	}

	resp, err := newRequest("/v1/chat/completions", reqBody, options.Timeout, options.Ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to send completions request: %s", err.Error())
	}

	if resp.StatusCode != http.StatusOK {
		fullResponse, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("API request failed with status %d", resp.StatusCode)
		}

		return nil, errors.New(string(fullResponse))
	}

	return resp.Body, nil
}

type Provider struct{}

var _ llm.Provider = &Provider{}

func (*Provider) SupportsStructuredOutput() bool {
	return true
}

func (*Provider) SupportsStreaming() bool {
	return true
}

func (*Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Prompt(model string, messages []llm.Message, options llm.Options) (llm.Response, error) {
	resp, err := createRequest(false, model, messages, options)
	if err != nil {
		return llm.Response{}, err
	}
	defer resp.Close()

	respContent := struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}{}
	err = json.NewDecoder(resp).Decode(&respContent)
	if err != nil {
		return llm.Response{}, err
	}
	if len(respContent.Choices) == 0 {
		return llm.Response{}, errors.New("no responses")
	}

	currentUsage := llm.TokenUsage{
		InputTokens:       respContent.Usage.PromptTokens,
		OutputTokens:      respContent.Usage.CompletionTokens,
		CachedInputTokens: respContent.Usage.PromptTokensDetails.CachedTokens,
	}

	rawLastMessage := respContent.Choices[len(respContent.Choices)-1].Message
	var lastMessage struct {
		Content   *string `json:"content"`
		ToolCalls []struct {
			Id       string `json:"id"`
			Type     string `json:"type"`
			Function *struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	}
	err = json.Unmarshal(rawLastMessage, &lastMessage)
	if err != nil {
		return llm.Response{}, fmt.Errorf("failed to unmarshal response: %s", err.Error())
	}

	if len(lastMessage.ToolCalls) > 0 {
		tools := lastMessage.ToolCalls
		var jsonTools []byte
		jsonTools, err = json.Marshal(tools)
		if err != nil {
			return llm.Response{}, fmt.Errorf("failed to marshal tools: %s", err.Error())
		}
		messages = append(messages, llm.Message{
			Role:      "assistant",
			ToolCalls: jsonTools,
		})

		for _, toolCall := range tools {
			if toolCall.Type != "function" {
				return llm.Response{}, errors.New("unsupported tool type " + toolCall.Type)
			}
			if toolCall.Function == nil {
				return llm.Response{}, errors.New("missing function")
			}

			foundTool := false
			var response any
			var resolveErr error
			log.Info("llm tool call " + toolCall.Function.Name)
			for _, tool := range options.Tools {
				if toolCall.Function.Name != tool.Function.Name {
					continue
				}
				foundTool = true

				var arguments json.RawMessage
				if uerr := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); uerr != nil {
					arguments = json.RawMessage("null")
				}

				response, resolveErr = tool.Resolver(arguments)
				break
			}
			if !foundTool {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: not found",
					ToolCallId: toolCall.Id,
				})
				continue
			}

			if resolveErr != nil {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: " + resolveErr.Error(),
					ToolCallId: toolCall.Id,
				})
				continue
			}

			responseJson, err := json.Marshal(response)
			if err != nil {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: " + err.Error(),
					ToolCallId: toolCall.Id,
				})
				continue
			}

			messages = append(messages, llm.Message{
				Role:       "tool",
				Content:    string(responseJson),
				ToolCallId: toolCall.Id,
			})
		}

		innerResp, err := p.Prompt(model, messages, options)
		if err != nil {
			return llm.Response{}, err
		}
		innerResp.Usage.InputTokens += currentUsage.InputTokens
		innerResp.Usage.OutputTokens += currentUsage.OutputTokens
		innerResp.Usage.CachedInputTokens += currentUsage.CachedInputTokens
		return innerResp, nil
	}

	if lastMessage.Content == nil {
		return llm.Response{}, errors.New("missing content")
	}

	messages = append(messages, llm.Message{
		Role:    "assistant",
		Content: *lastMessage.Content,
	})

	return llm.Response{
		Value:        *lastMessage.Content,
		Conversation: messages,
		Usage:        currentUsage,
	}, nil
}

func (*Provider) Stream(model string, messages []llm.Message, options llm.Options) (chan string, error) {
	resp, err := createRequest(true, model, messages, options)
	if err != nil {
		return nil, err
	}

	linesChannel := make(chan string)
	go func() {
		defer func() {
			resp.Close()
			close(linesChannel)
		}()

		reader := bufio.NewReader(resp)
		for {
			line, _, err := reader.ReadLine()
			if err != nil {
				break
			}

			lineStr := string(line)
			lineStr, ok := strings.CutPrefix(lineStr, "data:")
			if !ok {
				continue
			}

			lineStr = strings.TrimSpace(lineStr)
			contentJson := struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}{}
			err = json.Unmarshal([]byte(lineStr), &contentJson)
			if err != nil {
				continue
			}
			if len(contentJson.Choices) == 0 {
				continue
			}

			delta := contentJson.Choices[0].Delta.Content
			if delta == "" {
				continue
			}

			linesChannel <- delta
		}
	}()

	return linesChannel, nil
}
