package llm_test

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

const (
	insufficientQuotaBody = `{"error":{"message":"You have no credits remaining.","type":"insufficient_quota","code":"insufficient_quota"}}`
	rateLimitBody         = `{"error":{"message":"Rate limit reached for gpt-5","type":"requests","code":"rate_limit_exceeded"}}`
	numericCodeBody       = `{"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT"}}`
)

func TestIsPermanent(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "error that is not an Err", err: errors.New("dial tcp: connection refused"), want: false},
		{name: "Err without a status", err: &llm.Err{Body: "unexpected EOF"}, want: false},
		{name: "status below 400", err: &llm.Err{StatusCode: 302, Body: "moved"}, want: false},
		{name: "bad request", err: &llm.Err{StatusCode: 400, Body: `{"error":{"message":"unsupported parameter"}}`}, want: true},
		{name: "unauthorized", err: &llm.Err{StatusCode: 401, Body: "invalid api key"}, want: true},
		{name: "forbidden", err: &llm.Err{StatusCode: 403, Body: "forbidden"}, want: true},
		{name: "not found", err: &llm.Err{StatusCode: 404, Body: "model not found"}, want: true},
		{name: "unprocessable entity", err: &llm.Err{StatusCode: 422, Body: "unprocessable"}, want: true},
		{name: "request timeout", err: &llm.Err{StatusCode: 408, Body: "timeout"}, want: false},
		{name: "conflict", err: &llm.Err{StatusCode: 409, Body: "conflict"}, want: false},
		{name: "too many requests with an empty body", err: &llm.Err{StatusCode: 429}, want: false},
		{name: "too many requests with a rate limit body", err: &llm.Err{StatusCode: 429, Body: rateLimitBody}, want: false},
		{name: "too many requests with an unparsable body", err: &llm.Err{StatusCode: 429, Body: "<html>Too Many Requests</html>"}, want: false},
		{name: "too many requests with a numeric error code", err: &llm.Err{StatusCode: 429, Body: numericCodeBody}, want: false},
		{name: "too many requests with an exhausted quota", err: &llm.Err{StatusCode: 429, Body: insufficientQuotaBody}, want: true},
		{name: "too many requests with an exhausted quota type only", err: &llm.Err{StatusCode: 429, Body: `{"error":{"type":"insufficient_quota"}}`}, want: true},
		{name: "too many requests with an exhausted quota code only", err: &llm.Err{StatusCode: 429, Body: `{"error":{"code":"insufficient_quota"}}`}, want: true},
		{name: "bad request with a numeric error code", err: &llm.Err{StatusCode: 400, Body: numericCodeBody}, want: true},
		{name: "internal server error", err: &llm.Err{StatusCode: 500, Body: "boom"}, want: false},
		{name: "bad gateway", err: &llm.Err{StatusCode: 502, Body: "bad gateway"}, want: false},
		{name: "service unavailable", err: &llm.Err{StatusCode: 503, Body: "overloaded"}, want: false},
		{name: "wrapped Err", err: fmt.Errorf("prompting model: %w", &llm.Err{StatusCode: 404, Body: "model not found"}), want: true},
		{name: "wrapped Err that is transient", err: fmt.Errorf("prompting model: %w", &llm.Err{StatusCode: 500, Body: "boom"}), want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := llm.IsPermanent(tc.err); got != tc.want {
				t.Fatalf("IsPermanent(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestErrErrorKeepsEveryPopulatedField(t *testing.T) {
	cases := []struct {
		name     string
		err      *llm.Err
		contains []string
	}{
		{name: "status and body", err: &llm.Err{StatusCode: 404, Body: "model not found"}, contains: []string{"404", "model not found"}},
		{name: "status only", err: &llm.Err{StatusCode: 429}, contains: []string{"429"}},
		{name: "body only", err: &llm.Err{Body: "model not found"}, contains: []string{"model not found"}},
		{name: "nothing populated", err: &llm.Err{}, contains: nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.err.Error()
			if got == "" {
				t.Fatalf("Error() = %q, want a non-empty message", got)
			}
			for _, want := range tc.contains {
				if !strings.Contains(got, want) {
					t.Fatalf("Error() = %q, want it to contain %q", got, want)
				}
			}
			assertNoDanglingSeparator(t, got)
		})
	}
}

func TestNewErrCarriesStatusAndBody(t *testing.T) {
	resp := &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader("unsupported parameter"))}

	err := llm.NewErr(resp)

	var e *llm.Err
	if !errors.As(err, &e) {
		t.Fatalf("NewErr() = %v, want an *llm.Err", err)
	}
	if e.StatusCode != 400 {
		t.Fatalf("StatusCode = %d, want 400", e.StatusCode)
	}
	if e.Body != "unsupported parameter" {
		t.Fatalf("Body = %q, want %q", e.Body, "unsupported parameter")
	}
	if !llm.IsPermanent(err) {
		t.Fatalf("IsPermanent(%v) = false, want true", err)
	}
}

func TestNewErrReturnsTheReadError(t *testing.T) {
	readErr := errors.New("unable to read from closed buffer")
	resp := &http.Response{StatusCode: 200, Body: failingBody{readErr}}

	err := llm.NewErr(resp)

	if !errors.Is(err, readErr) {
		t.Fatalf("NewErr() = %v, want it to wrap %v", err, readErr)
	}
	if !strings.Contains(err.Error(), "200") {
		t.Fatalf("NewErr() = %v, want it to name the status", err)
	}
	if llm.IsPermanent(err) {
		t.Fatalf("IsPermanent(%v) = true, want false", err)
	}
}

type failingBody struct{ err error }

func (b failingBody) Read([]byte) (int, error) { return 0, b.err }
func (failingBody) Close() error               { return nil }

func assertNoDanglingSeparator(t *testing.T, msg string) {
	t.Helper()
	for _, suffix := range []string{" ", ":", "("} {
		if strings.HasSuffix(msg, suffix) {
			t.Fatalf("Error() = %q, want no trailing %q", msg, suffix)
		}
	}
}
