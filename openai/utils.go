package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	apikey "bitbucket.org/teamscript/go-llm/apikeys"
)

func newRequest(path string, body any, timeout time.Duration, ctx context.Context) (*http.Response, error) {
	// Convert request body to JSON
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON: %v", err)
	}

	// Create the HTTP request
	var req *http.Request
	if ctx == nil {
		req, err = http.NewRequest("POST", "https://api.openai.com"+path, bytes.NewBuffer(jsonData))
	} else {
		req, err = http.NewRequestWithContext(ctx, "POST", "https://api.openai.com"+path, bytes.NewBuffer(jsonData))
	}
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	apiKey, err := apikey.OpenAi()
	if err != nil {
		return nil, err
	}

	// Add headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	if timeout == 0 {
		timeout = time.Second * 30
	}

	// Make the request
	return (&http.Client{
		Timeout: timeout,
	}).Do(req)
}
