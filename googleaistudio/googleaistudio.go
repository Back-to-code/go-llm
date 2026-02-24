package googleaistudio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/Back-to-code/go-llm"
	apikey "github.com/Back-to-code/go-llm/apikeys"
)

type Provider struct{}

type ResponseFormat struct {
	Type string `json:"type"`
}

type Response struct {
	Candidates []struct {
		Content struct {
			Parts []Part `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
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
	MaxOutputTokens  int             `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string          `json:"response_mime_type,omitempty"`
	ThinkingConfig   *ThinkingConfig `json:"thinkingConfig,omitempty"`
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

func (p *Provider) Prompt(model string, messages []llm.Message, opts llm.Options) (string, error) {
	chatResponse, err := p.doRequest(model, messages, opts)
	if err != nil {
		return "", err
	}

	candidates := chatResponse.Candidates
	if len(candidates) == 0 {
		return "", errors.New("chat did not return any results")
	}

	parts := candidates[len(candidates)-1].Content.Parts
	if len(parts) == 0 {
		return "", errors.New("chat did not return any result parts")
	}

	// Check if the model is requesting function calls
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
			return "", fmt.Errorf("marshaling function calls: %w", err)
		}
		messages = append(messages, llm.Message{
			Role:      "assistant",
			ToolCalls: toolCallsJson,
		})

		// Resolve each function call
		responses, err := resolveToolCalls(functionParts, opts.Tools)
		if err != nil {
			return "", fmt.Errorf("resolving tool calls: %w", err)
		}

		// Append each result as a "tool" message. We use ToolCallId to carry
		// the function name (Gemini has no call IDs — it matches by name).
		for _, resp := range responses {
			responseJson, err := json.Marshal(resp.Response)
			if err != nil {
				responseJson = []byte(`{"error":"failed to marshal response"}`)
			}
			messages = append(messages, llm.Message{
				Role:       "tool",
				Content:    string(responseJson),
				ToolCallId: resp.Name,
			})
		}

		// Continue the conversation
		return p.Prompt(model, messages, opts)
	}

	// No function calls — return the text response
	var text string
	for _, part := range parts {
		if part.Text != "" {
			text += part.Text
		}
	}
	if text == "" {
		return "", errors.New("chat did not return any text content")
	}
	return text, nil
}

// doRequest builds and sends a single generateContent request, returning the
// parsed response. This is separated from Prompt so the tool-call loop can
// call it repeatedly without duplicating HTTP logic.
func (*Provider) doRequest(model string, messages []llm.Message, opts llm.Options) (*Response, error) {
	apiKey, err := apikey.GoogleAiStudio()
	if err != nil {
		return nil, err
	}

	contents, systemParts, err := convertMessages(messages)
	if err != nil {
		return nil, err
	}

	var systemInstruction *SystemInstruction
	if len(systemParts) > 0 {
		systemInstruction = &SystemInstruction{
			Parts: systemParts,
		}
	}

	var responseMimeType string
	if opts.ResponseFormat == llm.ResponseFormatJsonObject {
		responseMimeType = "application/json"
	}

	// Build tools and tool_config if tools are provided
	geminiTools := convertTools(opts.Tools)

	requestPayload := struct {
		SystemInstruction *SystemInstruction `json:"system_instruction,omitempty"`
		Contents          []Content          `json:"contents"`
		GenerationConfig  GenerationConfig   `json:"generationConfig"`
		Tools             []GeminiTool       `json:"tools,omitempty"`
	}{
		SystemInstruction: systemInstruction,
		Contents:          contents,
		GenerationConfig: GenerationConfig{
			MaxOutputTokens:  opts.MaxTokens,
			ResponseMimeType: responseMimeType,
			ThinkingConfig:   getThinkingConfig(model, opts.Thinking),
		},
		Tools: geminiTools,
	}

	requestPayloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, fmt.Errorf("marshaling payload: %s", err.Error())
	}
	requestBody := bytes.NewReader(requestPayloadBytes)

	var req *http.Request
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + apiKey
	method := "POST"
	if opts.Ctx == nil {
		req, err = http.NewRequest(method, url, requestBody)
	} else {
		req, err = http.NewRequestWithContext(opts.Ctx, method, url, requestBody)
	}
	if err != nil {
		return nil, fmt.Errorf("creating request: %s", err.Error())
	}

	req.Header.Set("Content-Type", "application/json")

	client := http.Client{Timeout: opts.Timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sending request: %s", err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("reading response body: %s", err.Error())
		}
		return nil, errors.New(string(respBody))
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
