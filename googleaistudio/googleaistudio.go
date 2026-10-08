package googleaistudio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/Back-to-code/go-llm"
	apikey "github.com/Back-to-code/go-llm/apikeys"
)

// BaseURL is the Google AI Studio API base URL. Exported for test overrides.
var BaseURL = "https://generativelanguage.googleapis.com"

type Provider struct{}

type ResponseFormat struct {
	Type string `json:"type"`
}

type Response struct {
	Candidates []struct {
		Content struct {
			Parts []Part `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
	} `json:"usageMetadata"`
}

type Content struct {
	Role  string `json:"role"` // "model", "user"
	Parts []Part `json:"parts"`
}

type Part struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
}

type SystemInstruction struct {
	Parts []Part `json:"parts,omitempty"`
}

type GenerationConfig struct {
	MaxOutputTokens    int             `json:"maxOutputTokens,omitempty"`
	ResponseMimeType   string          `json:"response_mime_type,omitempty"`
	ResponseJsonSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *ThinkingConfig `json:"thinkingConfig,omitempty"`
}

func (*Provider) SupportsStructuredOutput() bool {
	return true
}

func (*Provider) SupportsStreaming() bool {
	return false
}

func (*Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Call(model string, messages []llm.Message, opts llm.Options) (llm.Turn, error) {
	chatResponse, err := p.doRequest(model, messages, opts)
	if err != nil {
		return llm.Turn{}, err
	}

	candidates := chatResponse.Candidates
	if len(candidates) == 0 {
		return llm.Turn{}, errors.New("chat did not return any results")
	}

	usage := llm.TokenUsage{
		InputTokens:       chatResponse.UsageMetadata.PromptTokenCount,
		OutputTokens:      chatResponse.UsageMetadata.CandidatesTokenCount,
		CachedInputTokens: chatResponse.UsageMetadata.CachedContentTokenCount,
	}

	candidate := candidates[len(candidates)-1]
	parts := candidate.Content.Parts
	if len(parts) == 0 {
		// UNEXPECTED_TOOL_CALL means the model tried a tool while the mode was NONE.
		return llm.Turn{}, fmt.Errorf("chat did not return any result parts, finish reason %s", candidate.FinishReason)
	}

	var functionParts []Part
	for _, part := range parts {
		if part.FunctionCall != nil {
			functionParts = append(functionParts, part)
		}
	}

	if len(functionParts) > 0 {
		// Store the function calls on the assistant message so they can be
		// reconstructed into functionCall parts on the next round-trip.
		toolCallsJson, err := json.Marshal(functionParts)
		if err != nil {
			return llm.Turn{}, fmt.Errorf("marshaling function calls: %w", err)
		}

		turn := llm.Turn{
			Message: llm.Message{Role: "assistant", ToolCalls: toolCallsJson},
			Usage:   usage,
		}
		for position, part := range functionParts {
			turn.ToolCalls = append(turn.ToolCalls, llm.ToolCall{
				Id:        callId(part.FunctionCall, position),
				Name:      part.FunctionCall.Name,
				Arguments: part.FunctionCall.Args,
			})
		}
		return turn, nil
	}

	var text string
	for _, part := range parts {
		if part.Text != "" {
			text += part.Text
		}
	}
	if text == "" {
		return llm.Turn{}, errors.New("chat did not return any text content")
	}

	return llm.Turn{Message: llm.Assistant(text), Usage: usage}, nil
}

func (*Provider) doRequest(model string, messages []llm.Message, opts llm.Options) (*Response, error) {
	apiKey, err := apikey.GoogleAiStudio()
	if err != nil {
		return nil, llm.Permanent(err)
	}

	contents, systemParts, err := convertMessages(messages)
	if err != nil {
		return nil, llm.Permanent(err)
	}

	var systemInstruction *SystemInstruction
	if len(systemParts) > 0 {
		systemInstruction = &SystemInstruction{
			Parts: systemParts,
		}
	}

	generationConfig := GenerationConfig{
		MaxOutputTokens: opts.MaxTokens,
		ThinkingConfig:  getThinkingConfig(model, opts.Thinking),
	}
	// Gemini rejects a response mime type next to forced function calling.
	// Nothing is lost: a forced call answers with function calls, never text.
	forcedToolCall := len(opts.Tools) > 0 && opts.ToolChoice == llm.ToolChoiceRequired
	if !forcedToolCall {
		switch opts.ResponseFormat {
		case llm.ResponseFormatJsonObject:
			generationConfig.ResponseMimeType = "application/json"
		case llm.ResponseFormatJsonSchema:
			generationConfig.ResponseMimeType = "application/json"
			generationConfig.ResponseJsonSchema = opts.JsonSchema.Schema
		}
	}

	requestPayload := struct {
		SystemInstruction *SystemInstruction `json:"system_instruction,omitempty"`
		Contents          []Content          `json:"contents"`
		GenerationConfig  GenerationConfig   `json:"generationConfig"`
		Tools             []GeminiTool       `json:"tools,omitempty"`
		ToolConfig        *ToolConfig        `json:"toolConfig,omitempty"`
	}{
		SystemInstruction: systemInstruction,
		Contents:          contents,
		GenerationConfig:  generationConfig,
		Tools:             convertTools(opts.Tools),
	}
	if len(opts.Tools) > 0 {
		requestPayload.ToolConfig = toolConfig(opts.ToolChoice)
	}

	requestPayloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, llm.Permanent(fmt.Errorf("marshaling payload: %s", err.Error()))
	}
	requestBody := bytes.NewReader(requestPayloadBytes)

	var req *http.Request
	url := BaseURL + "/v1beta/models/" + model + ":generateContent?key=" + apiKey
	method := "POST"
	if opts.Ctx == nil {
		req, err = http.NewRequest(method, url, requestBody)
	} else {
		req, err = http.NewRequestWithContext(opts.Ctx, method, url, requestBody)
	}
	if err != nil {
		return nil, llm.Permanent(fmt.Errorf("creating request: %s", err.Error()))
	}

	req.Header.Set("Content-Type", "application/json")

	client := http.Client{Timeout: opts.Timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, llm.NewErr(resp)
	}

	chatResponse := Response{}
	err = json.NewDecoder(resp.Body).Decode(&chatResponse)
	if err != nil {
		return nil, fmt.Errorf("failed to decode body: %s", err.Error())
	}

	return &chatResponse, nil
}

func (*Provider) Stream(model string, messages []llm.Message, opts llm.Options) (chan string, error) {
	// FIXME
	return nil, errors.New("google ai studio provider does currently not support streaming")
}
