package openai

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"bitbucket.org/teamscript/go-llm"
	"bitbucket.org/teamscript/go-llm/log"
)

func toMessage(s llm.Message) Message {
	content := []MessageContent{}
	if s.Content != "" {
		content = append(content, MessageContent{
			Type: "text",
			Text: s.Content,
		})
	}

	role := s.Role
	if s.Role == "system" {
		role = "developer"
	}

	return Message{
		Role:       role,
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

type InferenceRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	ResponseFormat      ResponseFormat `json:"response_format"`
	Stream              bool           `json:"stream"`
	Store               bool           `json:"store"`
	Tools               []llm.Tool     `json:"tools"`
	ToolChoice          string         `json:"tool_choice,omitempty"`
	ReasoningEffort     string         `json:"reasoning_effort"`
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

	reasoningEffort := "minimal"
	switch options.Thinking {
	case llm.LowThinking:
		reasoningEffort = "low"
	case llm.MediumThinking:
		reasoningEffort = "medium"
	case llm.HighThinking:
		reasoningEffort = "high"
	}

	reqBody := InferenceRequest{
		Stream:          stream,
		Model:           model,
		Messages:        bodyMessages,
		ResponseFormat:  ResponseFormat{responseFormat},
		Store:           false,
		Tools:           options.Tools,
		ReasoningEffort: reasoningEffort,
	}

	if len(options.Tools) > 0 {
		reqBody.ToolChoice = "auto"
	}

	reqBody.MaxCompletionTokens = options.MaxTokens

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

func (p *Provider) Prompt(model string, messages []llm.Message, options llm.Options) (string, error) {
	resp, err := createRequest(false, model, messages, options)
	if err != nil {
		return "", err
	}
	defer resp.Close()

	respContent := struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}{}
	err = json.NewDecoder(resp).Decode(&respContent)
	if err != nil {
		return "", err
	}
	if len(respContent.Choices) == 0 {
		return "", errors.New("no responses")
	}

	rawLastMessage := respContent.Choices[len(respContent.Choices)-1].Message
	var lastMessage struct {
		Content   *string `json:"content"`
		ToolCalls []struct {
			Id       string `json:"id"`
			Type     string `json:"type"`
			Function *struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	}
	err = json.Unmarshal(rawLastMessage, &lastMessage)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %s", err.Error())
	}

	if len(lastMessage.ToolCalls) > 0 {
		tools := lastMessage.ToolCalls
		var jsonTools []byte
		jsonTools, err = json.Marshal(tools)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tools: %s", err.Error())
		}
		messages = append(messages, llm.Message{
			Role:      "assistant",
			ToolCalls: jsonTools,
		})

		for _, tool := range tools {
			if tool.Type != "function" {
				return "", errors.New("unsupported tool type " + tool.Type)
			}
			if tool.Function == nil {
				return "", errors.New("missing function")
			}

			foundTool := false
			var response any
			var err error
			log.Info("llm tool call " + tool.Function.Name)
			for _, tool := range options.Tools {
				if tool.Function.Name == tool.Function.Name {
					foundTool = true
					response, err = tool.Resolver(tool.Function.Parameters)
					break
				}
			}
			if !foundTool {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: not found",
					ToolCallId: tool.Id,
				})
				continue
			}

			if err != nil {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: " + err.Error(),
					ToolCallId: tool.Id,
				})
				continue
			}

			responseJson, err := json.Marshal(response)
			if err != nil {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: " + err.Error(),
					ToolCallId: tool.Id,
				})
				continue
			}

			messages = append(messages, llm.Message{
				Role:       "tool",
				Content:    string(responseJson),
				ToolCallId: tool.Id,
			})
		}

		return p.Prompt(model, messages, options)
	}

	if lastMessage.Content == nil {
		return "", errors.New("missing content")
	}
	return *lastMessage.Content, nil
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
