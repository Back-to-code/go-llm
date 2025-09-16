package togetherai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"bitbucket.org/teamscript/go-llm"
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

func (*Provider) Prompt(model string, messages []llm.Message, opts llm.Options) (string, error) {
	requestPayload := struct {
		Messages       []llm.Message   `json:"messages"`
		Model          string          `json:"model"`
		MaxTokens      int             `json:"max_tokens"`
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
		return "", fmt.Errorf("marshaling payload: %s", err.Error())
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
		return "", fmt.Errorf("creating request: %s", err.Error())
	}

	apiKey := os.Getenv("TOGETHER_AI_TOKEN")
	if apiKey == "" {
		return "", errors.New("TOGETHER_AI_TOKEN environment variable not set")
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := http.Client{
		Timeout: opts.Timeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("sending request: %s", err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}
		return "", errors.New(string(respBody))
	}

	var responsePayload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	err = json.NewDecoder(resp.Body).Decode(&responsePayload)
	if err != nil {
		return "", fmt.Errorf("decoding response: %s", err.Error())
	}

	return responsePayload.Choices[0].Message.Content, nil
}

func (*Provider) Stream(model string, messages []llm.Message, opts llm.Options) (chan string, error) {
	// FIXME
	return nil, errors.New("togetherai provider does currently not support streaming")
}
