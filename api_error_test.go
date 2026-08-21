package llm_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/openai"
)

// The body gpt-5.6-sol returns for function tools combined with a reasoning
// effort.
const toolsWithReasoningBody = `{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.6-sol in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.","type":"invalid_request_error","param":"reasoning_effort","code":null}}`

const insufficientQuotaBody = `{"error":{"message":"You have no credits remaining.","type":"insufficient_quota","code":"insufficient_quota"}}`

const rateLimitBody = `{"error":{"message":"Rate limit reached for gpt-5.6-sol.","type":"rate_limit_error","code":"rate_limit_exceeded"}}`

func TestAPIErrorPermanent(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"bad request", http.StatusBadRequest, toolsWithReasoningBody, true},
		{"unauthorized", http.StatusUnauthorized, "", true},
		{"forbidden", http.StatusForbidden, "", true},
		{"not found", http.StatusNotFound, "", true},
		{"unprocessable", http.StatusUnprocessableEntity, "", true},
		{"request timeout", http.StatusRequestTimeout, "", false},
		{"conflict", http.StatusConflict, "", false},
		{"rate limited", http.StatusTooManyRequests, rateLimitBody, false},
		{"rate limited, empty body", http.StatusTooManyRequests, "", false},
		{"quota exhausted", http.StatusTooManyRequests, insufficientQuotaBody, true},
		{"internal error", http.StatusInternalServerError, "", false},
		{"bad gateway", http.StatusBadGateway, "", false},
		{"service unavailable", http.StatusServiceUnavailable, "", false},
		{"gateway timeout", http.StatusGatewayTimeout, "", false},
		// Google AI Studio puts a number in error.code.
		{"numeric code", http.StatusBadRequest, `{"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}}`, true},
		{"unparsable body", http.StatusTooManyRequests, "not json at all", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := &llm.APIError{StatusCode: tc.status, Body: tc.body}
			if got := llm.IsPermanent(err); got != tc.want {
				t.Errorf("IsPermanent() = %v, want %v (status %d)", got, tc.want, tc.status)
			}
		})
	}
}

func TestIsPermanentNonAPIError(t *testing.T) {
	if llm.IsPermanent(http.ErrHandlerTimeout) {
		t.Error("IsPermanent() = true for a non-API error, want false")
	}
	if llm.IsPermanent(nil) {
		t.Error("IsPermanent(nil) = true, want false")
	}
}

func TestAPIErrorMessageKeepsStatusAndBody(t *testing.T) {
	err := &llm.APIError{StatusCode: http.StatusBadRequest, Body: toolsWithReasoningBody}
	msg := err.Error()
	if !strings.Contains(msg, "400") {
		t.Errorf("error message lost the status code: %q", msg)
	}
	if !strings.Contains(msg, "reasoning_effort") {
		t.Errorf("error message lost the response body: %q", msg)
	}

	withoutBody := (&llm.APIError{StatusCode: http.StatusBadGateway}).Error()
	if strings.HasSuffix(withoutBody, ": ") {
		t.Errorf("empty body should not leave a dangling separator: %q", withoutBody)
	}
}

type response struct {
	status int
	body   string
}

const okBody = `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`

// serveStatuses walks the given responses in order, repeating the last one once
// exhausted, and reports how many requests arrived.
func serveStatuses(t *testing.T, responses ...response) (*llm.Model, *int) {
	t.Helper()

	t.Setenv("OPENAI_TOKEN", "test-token")

	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := requests
		requests++
		if idx >= len(responses) {
			idx = len(responses) - 1
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(responses[idx].status)
		_, _ = w.Write([]byte(responses[idx].body))
	}))
	t.Cleanup(server.Close)

	prev := openai.BaseURL
	openai.BaseURL = server.URL
	t.Cleanup(func() { openai.BaseURL = prev })

	return &llm.Model{Name: "gpt-5.6-sol", Provider: &openai.Provider{}}, &requests
}

// Before this fix every permanent 400 cost five identical API calls.
func TestPromptDoesNotRetryPermanentError(t *testing.T) {
	model, requests := serveStatuses(t, response{http.StatusBadRequest, toolsWithReasoningBody})

	_, err := model.PromptSingle("hi", llm.Options{})
	if err == nil {
		t.Fatal("PromptSingle succeeded, want an error")
	}
	if *requests != 1 {
		t.Errorf("sent %d requests, want 1", *requests)
	}

	if !llm.IsPermanent(err) {
		t.Errorf("error is not reported as permanent: %v", err)
	}
	if !strings.Contains(err.Error(), "reasoning_effort") {
		t.Errorf("error lost the API explanation: %v", err)
	}
}

func TestPromptDoesNotRetryExhaustedQuota(t *testing.T) {
	model, requests := serveStatuses(t, response{http.StatusTooManyRequests, insufficientQuotaBody})

	if _, err := model.PromptSingle("hi", llm.Options{}); err == nil {
		t.Fatal("PromptSingle succeeded, want an error")
	}
	if *requests != 1 {
		t.Errorf("sent %d requests, want 1", *requests)
	}
}

func TestPromptRetriesTransientError(t *testing.T) {
	model, requests := serveStatuses(t,
		response{http.StatusInternalServerError, `{"error":{"message":"boom"}}`},
		response{http.StatusOK, okBody},
	)

	resp, err := model.PromptSingle("hi", llm.Options{})
	if err != nil {
		t.Fatalf("PromptSingle returned error: %v", err)
	}
	if resp.Value != "done" {
		t.Errorf("got %q, want %q", resp.Value, "done")
	}
	if *requests != 2 {
		t.Errorf("sent %d requests, want 2", *requests)
	}
}

func TestPromptRetriesRateLimit(t *testing.T) {
	model, requests := serveStatuses(t,
		response{http.StatusTooManyRequests, rateLimitBody},
		response{http.StatusOK, okBody},
	)

	if _, err := model.PromptSingle("hi", llm.Options{}); err != nil {
		t.Fatalf("PromptSingle returned error: %v", err)
	}
	if *requests != 2 {
		t.Errorf("sent %d requests, want 2", *requests)
	}
}

func TestPromptNoRetryOptionStillSendsOnce(t *testing.T) {
	model, requests := serveStatuses(t, response{http.StatusInternalServerError, "boom"})

	if _, err := model.PromptSingle("hi", llm.Options{NoRetry: true}); err == nil {
		t.Fatal("PromptSingle succeeded, want an error")
	}
	if *requests != 1 {
		t.Errorf("sent %d requests, want 1", *requests)
	}
}
