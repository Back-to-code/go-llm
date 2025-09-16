package googleaistudio

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

type Response struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

type Content struct {
	Role  string `json:"role"` // "model", "user"
	Parts []Part `json:"parts"`
}

type Part struct {
	Text string `json:"text"`
}

type SystemInstruction struct {
	Parts []Part `json:"parts,omitempty"`
}

type GenerationConfig struct {
	MaxOutputTokens  int    `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string `json:"response_mime_type,omitempty"`
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
	apiKey := os.Getenv("GOOGLE_AI_STUDIO_KEY")
	if apiKey == "" {
		return "", errors.New("GOOGLE_AI_STUDIO_KEY environment variable not set")
	}

	content := []Content{}
	systemParts := []Part{}
	for _, message := range messages {
		role := ""
		switch message.Role {
		case "user":
			role = "user"
		case "assistant":
			role = "model"
		case "system":
			systemParts = append(systemParts, Part{message.Content})
			continue
		default:
			return "", errors.New(message.Role + " role currently not supported for gemini")
		}

		content = append(content, Content{
			Role:  role,
			Parts: []Part{{message.Content}},
		})
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

	requestPayload := struct {
		SystemInstruction *SystemInstruction `json:"system_instruction,omitempty"`
		Contents          []Content          `json:"contents"`
		GenerationConfig  GenerationConfig   `json:"generationConfig"`
	}{
		SystemInstruction: systemInstruction,
		Contents:          content,
		GenerationConfig: GenerationConfig{
			MaxOutputTokens:  opts.MaxTokens,
			ResponseMimeType: responseMimeType,
		},
	}

	requestPayloadBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return "", fmt.Errorf("marshaling payload: %s", err.Error())
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
		return "", fmt.Errorf("creating request: %s", err.Error())
	}

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
			return "", fmt.Errorf("reading response body: %s", err.Error())
		}
		return "", errors.New(string(respBody))
	}

	chatResponse := Response{}
	err = json.NewDecoder(resp.Body).Decode(&chatResponse)
	if err != nil {
		return "", fmt.Errorf("failed to decode body: %s", err.Error())
	}

	candidates := chatResponse.Candidates
	if len(candidates) == 0 {
		return "", errors.New("chat did not return any results")
	}

	parts := candidates[len(candidates)-1].Content.Parts
	if len(parts) == 0 {
		return "", errors.New("chat did not return any result parts")
	}

	return parts[len(parts)-1].Text, nil
}

func (*Provider) Stream(model string, messages []llm.Message, opts llm.Options) (chan string, error) {
	// FIXME
	return nil, errors.New("google ai studio provider does currently not support streaming")
}
