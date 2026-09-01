package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Err is a provider failure that keeps the HTTP status alongside the response
// body, so a caller can tell a permanent rejection from a transient one.
type Err struct {
	StatusCode int // 0 when the failure was not an HTTP status failure
	Body       string
}

// NewErr reads resp's body into an *Err. A failed read comes back as the read
// error instead, which IsPermanent treats as transient. It does not close the
// body: the providers differ on when they close theirs.
func NewErr(resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading status %d response body: %w", resp.StatusCode, err)
	}

	return &Err{StatusCode: resp.StatusCode, Body: string(body)}
}

func (e *Err) Error() string {
	switch {
	case e.StatusCode != 0 && e.Body != "":
		return fmt.Sprintf("status %d: %s", e.StatusCode, e.Body)
	case e.StatusCode != 0:
		return fmt.Sprintf("status %d", e.StatusCode)
	case e.Body != "":
		return e.Body
	default:
		return "empty error"
	}
}

// IsPermanent reports whether err is a rejection that resending cannot fix.
// It is false for nil, for errors that do not wrap an *Err, and for an *Err
// without an HTTP status, so network and read failures stay retryable.
func IsPermanent(err error) bool {
	var e *Err
	if !errors.As(err, &e) {
		return false
	}

	if e.StatusCode < 400 || e.StatusCode >= 500 {
		return false
	}

	switch e.StatusCode {
	case http.StatusRequestTimeout, http.StatusConflict:
		return false
	case http.StatusTooManyRequests:
		// A rate limit clears on its own, a spent quota does not.
		body := struct {
			Error struct {
				Type string `json:"type"`
				// A string on OpenAI, a number on Google AI Studio, so decoding
				// it as a string would fail the whole body.
				Code json.RawMessage `json:"code"`
			} `json:"error"`
		}{}
		if json.Unmarshal([]byte(e.Body), &body) != nil {
			return false
		}
		code := strings.Trim(string(body.Error.Code), `"`)
		return body.Error.Type == "insufficient_quota" || code == "insufficient_quota"
	}

	return true
}
