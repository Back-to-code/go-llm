package togetherai

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

func (*Provider) SupportsStructuredOutput() bool {
	return true
}

func (*Provider) SupportsStreaming() bool {
	return false
}

func (*Provider) SupportsTools() bool {
	return false
}

func (*Provider) Prompt(model string, messages []llm.Message, opts llm.Options) (llm.Response, error) {
	requestPayload := struct {
		Messages       []llm.Message   `json:"messages"`
		Model          string          `json:"model"`
		MaxTokens      int             `json:"max_tokens,omitempty"`
		Stream         bool            `json:"stream"`
		ResponseFormat *ResponseFormat `json:"responseFormat,omitempty"`
	}{
		Messages:  messages,
		Model:     model,
		MaxTokens: opts.MaxTokens,
		Stream:    false,
	}
	if opts.ResponseFormat != "" {
		requestPayload.ResponseFormat = &ResponseFormat{string(opts.ResponseFormat)}
	}

	requestPayloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return llm.Response{}, fmt.Errorf("marshaling payload: %s", err.Error())
	}
	requestBody := bytes.NewReader(requestPayloadBytes)

	var req *http.Request
	url := "https://api.together.xyz/v1/chat/completions"
	method := "POST"
	if opts.Ctx == nil {
		req, err = http.NewRequest(method, url, requestBody)
	} else {
		req, err = http.NewRequestWithContext(opts.Ctx, method, url, requestBody)
	}
	if err != nil {
		return llm.Response{}, fmt.Errorf("creating request: %s", err.Error())
	}

	apiKey, err := apikey.TogetherAi()
	if err != nil {
		return llm.Response{}, err
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := http.Client{
		Timeout: opts.Timeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		return llm.Response{}, fmt.Errorf("sending request: %s", err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return llm.Response{}, err
		}
		return llm.Response{}, errors.New(string(respBody))
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
		return llm.Response{}, fmt.Errorf("decoding response: %s", err.Error())
	}

	content := responsePayload.Choices[0].Message.Content

	// Append the final assistant message to the conversation.
	messages = append(messages, llm.Message{
		Role:    "assistant",
		Content: content,
	})

	return llm.Response{
		Value:        content,
		Conversation: messages,
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
