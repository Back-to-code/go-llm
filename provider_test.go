package llm_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

const neverSucceeds int32 = math.MaxInt32

func TestPromptRetriesOnlyTransientErrors(t *testing.T) {
	cases := []struct {
		name            string
		failures        int32 // failed calls before the stub starts succeeding
		failWith        error
		options         llm.Options
		wantCalls       int32
		wantValue       string
		wantErrContains string
	}{
		{
			name:            "bad request is not retried",
			failures:        neverSucceeds,
			failWith:        &llm.Err{StatusCode: 400, Body: `{"error":{"message":"unsupported parameter: reasoning"}}`},
			wantCalls:       1,
			wantErrContains: "unsupported parameter: reasoning",
		},
		{
			name:            "exhausted quota is not retried",
			failures:        neverSucceeds,
			failWith:        &llm.Err{StatusCode: 429, Body: insufficientQuotaBody},
			wantCalls:       1,
			wantErrContains: "You have no credits remaining.",
		},
		{
			name:      "server error is retried",
			failures:  1,
			failWith:  &llm.Err{StatusCode: 500, Body: "boom"},
			wantCalls: 2,
			wantValue: "recovered",
		},
		{
			name:      "rate limit is retried",
			failures:  1,
			failWith:  &llm.Err{StatusCode: 429, Body: rateLimitBody},
			wantCalls: 2,
			wantValue: "recovered",
		},
		{
			name:            "error without a status uses every retry",
			failures:        neverSucceeds,
			failWith:        errors.New("dial tcp: connection refused"),
			wantCalls:       5,
			wantErrContains: "connection refused",
		},
		{
			name:            "NoRetry stops after the first transient error",
			failures:        neverSucceeds,
			failWith:        &llm.Err{StatusCode: 500, Body: "boom"},
			options:         llm.Options{NoRetry: true},
			wantCalls:       1,
			wantErrContains: "boom",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider := &stubProvider{}
			provider.promptFn = func(_ string, messages []llm.Message, _ llm.Options) (llm.Response, error) {
				if provider.promptCalls.Load() <= tc.failures {
					return llm.Response{}, tc.failWith
				}
				return llm.Response{Value: "recovered", Conversation: messages}, nil
			}
			model := &llm.Model{Name: "stub", Provider: provider}

			resp, err := model.Prompt([]llm.Message{llm.User("hi")}, tc.options)

			if got := provider.promptCalls.Load(); got != tc.wantCalls {
				t.Fatalf("provider calls = %d, want %d", got, tc.wantCalls)
			}
			if tc.wantValue != "" {
				if err != nil {
					t.Fatalf("Prompt() error = %v, want nil", err)
				}
				if resp.Value != tc.wantValue {
					t.Fatalf("Prompt() value = %q, want %q", resp.Value, tc.wantValue)
				}
				return
			}
			if err == nil {
				t.Fatalf("Prompt() error = nil, want one containing %q", tc.wantErrContains)
			}
			if !strings.Contains(err.Error(), tc.wantErrContains) {
				t.Fatalf("Prompt() error = %v, want it to contain %q", err, tc.wantErrContains)
			}
		})
	}
}
