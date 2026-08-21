package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("API request failed with status %d", e.StatusCode)
	}
	return fmt.Sprintf("API request failed with status %d: %s", e.StatusCode, e.Body)
}

func (e *APIError) quotaExhausted() bool {
	code, errType := e.errorFields()
	return code == "insufficient_quota" || errType == "insufficient_quota"
}

// errorFields reads the {"error": {"type": ..., "code": ...}} shape shared by
// the OpenAI and Google AI Studio error bodies.
func (e *APIError) errorFields() (code string, errType string) {
	var body struct {
		Error struct {
			Code json.RawMessage `json:"code"`
			Type string          `json:"type"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(e.Body), &body) != nil {
		return "", ""
	}

	// code is a string on OpenAI and a number on Google AI Studio; only the
	// string form carries meaning here.
	_ = json.Unmarshal(body.Error.Code, &code)

	return code, body.Error.Type
}

func IsPermanent(err error) bool {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return false
	}

	switch apiErr.StatusCode {
	case http.StatusRequestTimeout, http.StatusConflict:
		return false
	case http.StatusTooManyRequests:
		// A rate limit clears on its own, an exhausted quota does not.
		return apiErr.quotaExhausted()
	}

	if apiErr.StatusCode >= 500 {
		return false
	}

	return apiErr.StatusCode >= 400
}
