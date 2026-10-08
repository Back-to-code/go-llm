package togetherai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/Back-to-code/go-llm"
	apikey "github.com/Back-to-code/go-llm/apikeys"
)

// BaseURL is the Together AI API base URL. Exported for test overrides.
var BaseURL = "https://api.together.xyz"

type Provider struct{}

type ResponseFormat struct {
	Type       string          `json:"type"`
	JsonSchema *llm.JsonSchema `json:"json_schema,omitempty"`
}

func (*Provider) SupportsStructuredOutput() bool {
	return true
}

func (*Provider) SupportsStreaming() bool {
	return false
}

func (*Provider) SupportsTools() bool {
	return false
}

func (*Provider) Call(model string, messages []llm.Message, opts llm.Options) (llm.Turn, error) {
	requestPayload := struct {
		Messages       []llm.Message   `json:"messages"`
		Model          string          `json:"model"`
		MaxTokens      int             `json:"max_tokens,omitempty"`
		Stream         bool            `json:"stream"`
		ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	}{
		Messages:  messages,
		Model:     model,
		MaxTokens: opts.MaxTokens,
		Stream:    false,
	}
	if opts.ResponseFormat != "" {
		requestPayload.ResponseFormat = &ResponseFormat{Type: string(opts.ResponseFormat)}
	}
	if opts.ResponseFormat == llm.ResponseFormatJsonSchema {
		requestPayload.ResponseFormat.JsonSchema = &opts.JsonSchema
	}

	requestPayloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return llm.Turn{}, llm.Permanent(fmt.Errorf("marshaling payload: %s", err.Error()))
	}
	requestBody := bytes.NewReader(requestPayloadBytes)

	var req *http.Request
	url := BaseURL + "/v1/chat/completions"
	method := "POST"
	if opts.Ctx == nil {
		req, err = http.NewRequest(method, url, requestBody)
	} else {
		req, err = http.NewRequestWithContext(opts.Ctx, method, url, requestBody)
	}
	if err != nil {
		return llm.Turn{}, llm.Permanent(fmt.Errorf("creating request: %s", err.Error()))
	}

	apiKey, err := apikey.TogetherAi()
	if err != nil {
		return llm.Turn{}, llm.Permanent(err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := http.Client{
		Timeout: opts.Timeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		return llm.Turn{}, fmt.Errorf("sending request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return llm.Turn{}, llm.NewErr(resp)
	}

	var responsePayload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	err = json.NewDecoder(resp.Body).Decode(&responsePayload)
	if err != nil {
		return llm.Turn{}, fmt.Errorf("decoding response: %s", err.Error())
	}

	if len(responsePayload.Choices) == 0 {
		return llm.Turn{}, errors.New("no responses")
	}

	return llm.Turn{
		Message: llm.Assistant(responsePayload.Choices[0].Message.Content),
		Usage: llm.TokenUsage{
			InputTokens:  responsePayload.Usage.PromptTokens,
			OutputTokens: responsePayload.Usage.CompletionTokens,
		},
	}, nil
}

func (*Provider) Stream(model string, messages []llm.Message, opts llm.Options) (chan string, error) {
	// FIXME
	return nil, errors.New("togetherai provider does currently not support streaming")
}
