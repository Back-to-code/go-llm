package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	llm "github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/cache"
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
			name:            "request that was never sent is not retried",
			failures:        neverSucceeds,
			failWith:        llm.Permanent(errors.New("OPENAI_TOKEN environment variable not set")),
			wantCalls:       1,
			wantErrContains: "OPENAI_TOKEN",
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
			provider.callFn = func(string, []llm.Message, llm.Options) (llm.Turn, error) {
				if provider.calls.Load() <= tc.failures {
					return llm.Turn{}, tc.failWith
				}
				return llm.Turn{Message: llm.Assistant("recovered")}, nil
			}
			model := &llm.Model{Name: "stub", Provider: provider}

			resp, err := model.Prompt([]llm.Message{llm.User("hi")}, tc.options)

			if got := provider.calls.Load(); got != tc.wantCalls {
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

func TestCacheKeyCoversOptionsAndToolFields(t *testing.T) {
	store := map[string]string{}
	prevGetter, prevSetter := cache.Getter, cache.Setter
	cache.Getter = func(key string) (string, error) { return store[key], nil }
	cache.Setter = func(key, value string, _ time.Duration) error {
		store[key] = value
		return nil
	}
	t.Cleanup(func() { cache.Getter, cache.Setter = prevGetter, prevSetter })

	schema := func(property string) llm.Options {
		return llm.Options{
			Cache:          time.Hour,
			ResponseFormat: llm.ResponseFormatJsonSchema,
			JsonSchema:     llm.JsonSchema{Schema: json.RawMessage(`{"type":"object","properties":{"` + property + `":{}}}`)},
		}
	}
	toolCall := func(id string) []llm.Message {
		return []llm.Message{
			llm.User("hi"),
			{Role: "assistant", ToolCalls: json.RawMessage(`[{"id":"` + id + `"}]`)},
			{Role: "tool", Content: "ok", ToolCallId: id},
		}
	}

	cases := []struct {
		name                   string
		firstMsgs, secondMsgs  []llm.Message
		firstOpts, secondOpts  llm.Options
		wantSecondFromTheCache bool
	}{
		{
			name:      "same input",
			firstMsgs: []llm.Message{llm.User("hi")}, secondMsgs: []llm.Message{llm.User("hi")},
			firstOpts: schema("color"), secondOpts: schema("color"),
			wantSecondFromTheCache: true,
		},
		{
			name:      "different schema",
			firstMsgs: []llm.Message{llm.User("hi")}, secondMsgs: []llm.Message{llm.User("hi")},
			firstOpts: schema("color"), secondOpts: schema("size"),
		},
		{
			name:      "different thinking",
			firstMsgs: []llm.Message{llm.User("hi")}, secondMsgs: []llm.Message{llm.User("hi")},
			firstOpts: llm.Options{Cache: time.Hour}, secondOpts: llm.Options{Cache: time.Hour, Thinking: llm.HighThinking},
		},
		{
			name:      "different tool calls",
			firstMsgs: toolCall("call_1"), secondMsgs: toolCall("call_2"),
			firstOpts: llm.Options{Cache: time.Hour}, secondOpts: llm.Options{Cache: time.Hour},
		},
		{
			name:      "same tool calls",
			firstMsgs: toolCall("call_1"), secondMsgs: toolCall("call_1"),
			firstOpts: llm.Options{Cache: time.Hour}, secondOpts: llm.Options{Cache: time.Hour},
			wantSecondFromTheCache: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clear(store)
			provider := &stubProvider{}
			provider.callFn = func(string, []llm.Message, llm.Options) (llm.Turn, error) {
				return llm.Turn{Message: llm.Assistant(fmt.Sprintf("answer %d", provider.calls.Load()))}, nil
			}
			model := &llm.Model{Name: "stub", Provider: provider}

			if _, err := model.Prompt(tc.firstMsgs, tc.firstOpts); err != nil {
				t.Fatalf("first Prompt() error = %v", err)
			}
			resp, err := model.Prompt(tc.secondMsgs, tc.secondOpts)
			if err != nil {
				t.Fatalf("second Prompt() error = %v", err)
			}

			fromTheCache := resp.Value == "answer 1"
			if fromTheCache != tc.wantSecondFromTheCache {
				t.Errorf("second Value = %q, want it from the cache: %v", resp.Value, tc.wantSecondFromTheCache)
			}
		})
	}
}

func TestCancelledCallIsNotReportedAsRetrying(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	provider := &stubProvider{callFn: func(string, []llm.Message, llm.Options) (llm.Turn, error) {
		cancel()
		return llm.Turn{}, fmt.Errorf("sending request: %w", context.Canceled)
	}}
	model := &llm.Model{Name: "stub", Provider: provider}

	var results []llm.CallResult
	_, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		Ctx:       ctx,
		OnCallEnd: func(_ llm.CallInfo, result llm.CallResult) { results = append(results, result) },
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Prompt() error = %v, want context.Canceled", err)
	}
	if provider.calls.Load() != 1 {
		t.Errorf("provider calls = %d, want 1", provider.calls.Load())
	}
	if len(results) != 1 || results[0].Retrying {
		t.Errorf("call results = %+v, want one that is not retrying", results)
	}
}

func TestStreamRejectsToolChoiceRequired(t *testing.T) {
	provider := &stubProvider{callFn: okCallProvider("unused")}
	model := &llm.Model{Name: "stub", Provider: provider}

	_, err := model.Stream([]llm.Message{llm.User("hi")}, llm.Options{
		Tools:      []llm.Tool{{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return nil, nil }}},
		ToolChoice: llm.ToolChoiceRequired,
	})
	if err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("Stream() error = %v, want the tool choice required error", err)
	}
}
