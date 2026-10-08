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
	Type       string          `json:"type"`
	JsonSchema *llm.JsonSchema `json:"json_schema,omitempty"`
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

	responseFormat := ResponseFormat{Type: "text"}
	if options.ResponseFormat != "" {
		responseFormat.Type = string(options.ResponseFormat)
	}
	if options.ResponseFormat == llm.ResponseFormatJsonSchema {
		responseFormat.JsonSchema = &options.JsonSchema
	}

	maxTokens := min(options.MaxTokens, maxTokensCeiling)

	reqBody := InferenceRequest{
		Stream:          stream,
		Model:           model,
		Messages:        bodyMessages,
		ResponseFormat:  responseFormat,
		Tools:           options.Tools,
		MaxTokens:       maxTokens,
		ReasoningEffort: reasoningEffort(options.Thinking),
	}

	if len(options.Tools) > 0 {
		reqBody.ToolChoice = string(options.ToolChoice)
	}

	resp, err := newRequest("/v1/chat/completions", reqBody, options.Timeout, options.Ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to send completions request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, llm.NewErr(resp)
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

func (*Provider) Call(model string, messages []llm.Message, options llm.Options) (llm.Turn, error) {
	resp, err := createRequest(false, model, messages, options)
	if err != nil {
		return llm.Turn{}, err
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
		return llm.Turn{}, err
	}
	if len(respContent.Choices) == 0 {
		return llm.Turn{}, errors.New("no responses")
	}

	usage := llm.TokenUsage{
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
		return llm.Turn{}, fmt.Errorf("failed to unmarshal response: %s", err.Error())
	}

	if len(lastMessage.ToolCalls) > 0 {
		jsonTools, err := json.Marshal(lastMessage.ToolCalls)
		if err != nil {
			return llm.Turn{}, fmt.Errorf("failed to marshal tools: %s", err.Error())
		}

		turn := llm.Turn{
			Message: llm.Message{Role: "assistant", ToolCalls: jsonTools},
			Usage:   usage,
		}
		for _, toolCall := range lastMessage.ToolCalls {
			if toolCall.Type != "function" {
				return llm.Turn{}, errors.New("unsupported tool type " + toolCall.Type)
			}
			if toolCall.Function == nil {
				return llm.Turn{}, errors.New("missing function")
			}

			var arguments json.RawMessage
			if uerr := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); uerr != nil {
				arguments = json.RawMessage("null")
			}
			turn.ToolCalls = append(turn.ToolCalls, llm.ToolCall{
				Id:        toolCall.Id,
				Name:      toolCall.Function.Name,
				Arguments: arguments,
			})
		}
		return turn, nil
	}

	if lastMessage.Content == nil {
		return llm.Turn{}, errors.New("missing content")
	}

	return llm.Turn{Message: llm.Assistant(*lastMessage.Content), Usage: usage}, nil
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
