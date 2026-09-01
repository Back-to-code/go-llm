package openai

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

// InputContent is a single content part of an input message.
type InputContent struct {
	Type string `json:"type"` // "input_text" for user/developer, "output_text" for assistant
	Text string `json:"text"`
}

// InputMessage is a plain conversation message in the request input.
type InputMessage struct {
	Role    string         `json:"role"` // "user", "assistant", "developer"
	Content []InputContent `json:"content"`
}

// FunctionCallOutput is the result of a tool call, fed back to the model.
type FunctionCallOutput struct {
	Type   string `json:"type"` // always "function_call_output"
	CallId string `json:"call_id"`
	Output string `json:"output"`
}

// Tool is the flat function tool definition used by the Responses API.
// The chat/completions route nests these under a "function" key, the
// Responses API does not.
type Tool struct {
	Type        string          `json:"type"` // always "function"
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict"`
}

type TextFormat struct {
	Type string `json:"type"`
}

type TextConfig struct {
	Format TextFormat `json:"format"`
}

type ReasoningConfig struct {
	Effort string `json:"effort"`
}

type InferenceRequest struct {
	Model           string           `json:"model"`
	Input           []any            `json:"input"`
	MaxOutputTokens int              `json:"max_output_tokens,omitempty"`
	Text            TextConfig       `json:"text"`
	Stream          bool             `json:"stream"`
	Store           bool             `json:"store"`
	Tools           []Tool           `json:"tools,omitempty"`
	ToolChoice      string           `json:"tool_choice,omitempty"`
	Reasoning       *ReasoningConfig `json:"reasoning,omitempty"`
	Include         []string         `json:"include,omitempty"`
}

// toInput converts the provider agnostic messages into Responses API input
// items. Assistant messages that carry ToolCalls hold the raw output items of
// a previous response (reasoning items included) and are spliced back in
// verbatim, which is what keeps the model's reasoning alive across tool calls.
func toInput(messages []llm.Message) ([]any, error) {
	input := make([]any, 0, len(messages))

	for _, msg := range messages {
		if msg.Role == "tool" {
			input = append(input, FunctionCallOutput{
				Type:   "function_call_output",
				CallId: msg.ToolCallId,
				Output: msg.Content,
			})
			continue
		}

		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			var items []json.RawMessage
			if err := json.Unmarshal(msg.ToolCalls, &items); err != nil {
				return nil, fmt.Errorf("failed to decode stored output items: %s", err.Error())
			}
			for _, item := range items {
				input = append(input, item)
			}
			continue
		}

		if msg.Content == "" {
			continue
		}

		role := msg.Role
		contentType := "input_text"
		switch role {
		case "system":
			role = "developer"
		case "assistant":
			contentType = "output_text"
		}

		input = append(input, InputMessage{
			Role:    role,
			Content: []InputContent{{Type: contentType, Text: msg.Content}},
		})
	}

	return input, nil
}

func toTools(tools []llm.Tool) []Tool {
	if len(tools) == 0 {
		return nil
	}

	converted := make([]Tool, len(tools))
	for idx, tool := range tools {
		converted[idx] = Tool{
			Type:        "function",
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
			Strict:      tool.Strict,
		}
	}

	return converted
}

// mentionsJson reports whether any message says "json", which the API demands
// of the input before it accepts a json_object response format.
func mentionsJson(messages []llm.Message) bool {
	for _, msg := range messages {
		if strings.Contains(strings.ToLower(msg.Content), "json") {
			return true
		}
	}

	return false
}

func createRequest(stream bool, model string, messages []llm.Message, options llm.Options) (io.ReadCloser, error) {
	input, err := toInput(messages)
	if err != nil {
		return nil, err
	}

	responseFormat := "text"
	if options.ResponseFormat != "" {
		responseFormat = string(options.ResponseFormat)
	}

	if responseFormat == "json_object" && !mentionsJson(messages) {
		return nil, errors.New(`response format json_object requires a message mentioning "json"`)
	}

	reqBody := InferenceRequest{
		Stream:          stream,
		Model:           model,
		Input:           input,
		Text:            TextConfig{Format: TextFormat{Type: responseFormat}},
		Store:           false,
		Tools:           toTools(options.Tools),
		MaxOutputTokens: options.MaxTokens,
	}

	if len(options.Tools) > 0 {
		reqBody.ToolChoice = "auto"
	}

	if effort := reasoningEffort(model, options.Thinking); effort != "" {
		reqBody.Reasoning = &ReasoningConfig{Effort: effort}
	}

	// With store false, reasoning items only come back replayable if we ask for
	// them encrypted, and only a tool round ever replays them. Gated on the
	// model, not on the effort above: the fixed effort models reason without
	// taking an effort parameter.
	if len(options.Tools) > 0 && supportsReasoning(model) {
		reqBody.Include = []string{"reasoning.encrypted_content"}
	}

	resp, err := newRequest("/v1/responses", reqBody, options.Timeout, options.Ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to send responses request: %s", err.Error())
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, llm.NewErr(resp)
	}

	return resp.Body, nil
}

// OutputItem is a single item of the Responses API output array. Only the
// fields the provider acts on are decoded; replaying an item uses the raw JSON.
type OutputItem struct {
	Type    string `json:"type"` // "message", "reasoning", "function_call", ...
	Content []struct {
		Type    string `json:"type"` // "output_text" or "refusal"
		Text    string `json:"text"`
		Refusal string `json:"refusal"`
	} `json:"content"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	CallId    string `json:"call_id"`
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
		Status           string            `json:"status"`
		Output           []json.RawMessage `json:"output"`
		IncompleteDetail *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Usage struct {
			InputTokens        int `json:"input_tokens"`
			OutputTokens       int `json:"output_tokens"`
			InputTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
	}{}
	err = json.NewDecoder(resp).Decode(&respContent)
	if err != nil {
		return llm.Response{}, err
	}

	if respContent.Error != nil && respContent.Error.Message != "" {
		return llm.Response{}, errors.New(respContent.Error.Message)
	}
	if len(respContent.Output) == 0 {
		return llm.Response{}, errors.New("no responses")
	}

	currentUsage := llm.TokenUsage{
		InputTokens:       respContent.Usage.InputTokens,
		OutputTokens:      respContent.Usage.OutputTokens,
		CachedInputTokens: respContent.Usage.InputTokensDetails.CachedTokens,
	}

	var text strings.Builder
	var refusal string
	var toolCalls []OutputItem
	for _, rawItem := range respContent.Output {
		var item OutputItem
		if err := json.Unmarshal(rawItem, &item); err != nil {
			return llm.Response{}, fmt.Errorf("failed to unmarshal response: %s", err.Error())
		}

		switch item.Type {
		case "message":
			for _, part := range item.Content {
				switch part.Type {
				case "output_text":
					text.WriteString(part.Text)
				case "refusal":
					refusal = part.Refusal
				}
			}
		case "function_call":
			toolCalls = append(toolCalls, item)
		}
	}

	// A truncated response can hold a half written function call, so the tool
	// call loop below is skipped for one.
	truncated := respContent.Status == "incomplete"
	if truncated {
		reason := "unknown reason"
		if respContent.IncompleteDetail != nil && respContent.IncompleteDetail.Reason != "" {
			reason = respContent.IncompleteDetail.Reason
		}
		if text.Len() == 0 {
			return llm.Response{}, errors.New("incomplete response: " + reason)
		}
		log.Info("llm response incomplete (" + reason + "), returning partial content")
	}

	if len(toolCalls) > 0 && !truncated {
		// Keep every output item (reasoning included) on the assistant message
		// so the next request can replay them, otherwise the model loses its
		// train of thought between tool calls.
		outputItems, err := json.Marshal(respContent.Output)
		if err != nil {
			return llm.Response{}, fmt.Errorf("failed to marshal tools: %s", err.Error())
		}
		messages = append(messages, llm.Message{
			Role:      "assistant",
			ToolCalls: outputItems,
		})

		for _, toolCall := range toolCalls {
			foundTool := false
			var response any
			var resolveErr error
			log.Info("llm tool call " + toolCall.Name)
			for _, tool := range options.Tools {
				if toolCall.Name != tool.Function.Name {
					continue
				}
				foundTool = true

				var arguments json.RawMessage
				if uerr := json.Unmarshal([]byte(toolCall.Arguments), &arguments); uerr != nil {
					arguments = json.RawMessage("null")
				}

				response, resolveErr = tool.Resolver(arguments)
				break
			}
			if !foundTool {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: not found",
					ToolCallId: toolCall.CallId,
				})
				continue
			}

			if resolveErr != nil {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: " + resolveErr.Error(),
					ToolCallId: toolCall.CallId,
				})
				continue
			}

			responseJson, err := json.Marshal(response)
			if err != nil {
				messages = append(messages, llm.Message{
					Role:       "tool",
					Content:    "error: " + err.Error(),
					ToolCallId: toolCall.CallId,
				})
				continue
			}

			messages = append(messages, llm.Message{
				Role:       "tool",
				Content:    string(responseJson),
				ToolCallId: toolCall.CallId,
			})
		}

		// Recurse to continue the conversation after tool calls.
		// Accumulate token usage from this round with the inner rounds.
		innerResp, err := p.Prompt(model, messages, options)
		if err != nil {
			return llm.Response{}, err
		}
		innerResp.Usage.InputTokens += currentUsage.InputTokens
		innerResp.Usage.OutputTokens += currentUsage.OutputTokens
		innerResp.Usage.CachedInputTokens += currentUsage.CachedInputTokens
		return innerResp, nil
	}

	if text.Len() == 0 {
		if refusal != "" {
			return llm.Response{}, errors.New("refused: " + refusal)
		}

		return llm.Response{}, errors.New("missing content")
	}

	// Append the final assistant message to the conversation.
	messages = append(messages, llm.Message{
		Role:    "assistant",
		Content: text.String(),
	})

	return llm.Response{
		Value:        text.String(),
		Conversation: messages,
		Usage:        currentUsage,
	}, nil
}

// streamDelta returns the text delta of one SSE data payload, or "" for every
// other event. Failures are logged rather than returned: the channel Stream
// hands back cannot carry them.
func streamDelta(payload string) string {
	event := struct {
		Type     string `json:"type"`
		Delta    string `json:"delta"`
		Message  string `json:"message"`
		Response *struct {
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
			IncompleteDetails *struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
		} `json:"response"`
	}{}
	if err := json.Unmarshal([]byte(payload), &event); err != nil {
		return ""
	}

	switch event.Type {
	case "response.output_text.delta":
		return event.Delta
	case "error":
		log.Info("llm stream error: " + event.Message)
	case "response.failed":
		if event.Response != nil && event.Response.Error != nil {
			log.Info("llm stream failed: " + event.Response.Error.Message)
		}
	case "response.incomplete":
		reason := "unknown reason"
		if event.Response != nil && event.Response.IncompleteDetails != nil && event.Response.IncompleteDetails.Reason != "" {
			reason = event.Response.IncompleteDetails.Reason
		}
		log.Info("llm stream incomplete: " + reason)
	}

	return ""
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

		// ReadString over ReadLine: the completed event carries the whole
		// response object, which overflows a fixed line buffer.
		reader := bufio.NewReader(resp)
		for {
			line, readErr := reader.ReadString('\n')

			if payload, ok := strings.CutPrefix(strings.TrimSpace(line), "data:"); ok {
				if delta := streamDelta(strings.TrimSpace(payload)); delta != "" {
					linesChannel <- delta
				}
			}

			if readErr != nil {
				if readErr != io.EOF {
					log.Info("llm stream read error: " + readErr.Error())
				}
				break
			}
		}
	}()

	return linesChannel, nil
}
