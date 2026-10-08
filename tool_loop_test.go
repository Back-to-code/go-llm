package llm_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

func toolTurn(calls ...llm.ToolCall) llm.Turn {
	return llm.Turn{
		Message:   llm.Message{Role: "assistant", ToolCalls: json.RawMessage(`[]`)},
		ToolCalls: calls,
	}
}

func lookupCall(id string) llm.ToolCall {
	return llm.ToolCall{Id: id, Name: "lookup", Arguments: json.RawMessage(`{}`)}
}

func lookupTool(runs *int, result any) llm.Tool {
	return llm.Tool{
		Function: llm.FunctionDef{Name: "lookup"},
		Resolver: func(json.RawMessage) (any, error) {
			*runs++
			return result, nil
		},
	}
}

// greedyProvider calls a tool on every request it is allowed to, and records
// the tool choice and messages of each request.
func greedyProvider(callsPerTurn int) (*stubProvider, *[]llm.ToolChoice, *[][]llm.Message) {
	var choices []llm.ToolChoice
	var requests [][]llm.Message
	provider := &stubProvider{}
	provider.callFn = func(_ string, messages []llm.Message, options llm.Options) (llm.Turn, error) {
		choices = append(choices, options.ToolChoice)
		requests = append(requests, messages)
		if options.ToolChoice == llm.ToolChoiceNone {
			return llm.Turn{Message: llm.Assistant("final")}, nil
		}

		calls := make([]llm.ToolCall, callsPerTurn)
		for idx := range calls {
			calls[idx] = lookupCall(fmt.Sprintf("call_%d_%d", len(choices), idx))
		}
		return toolTurn(calls...), nil
	}
	return provider, &choices, &requests
}

func TestMaxToolCallsEndsTheLoop(t *testing.T) {
	provider, choices, requests := greedyProvider(1)
	model := &llm.Model{Name: "stub", Provider: provider}

	var runs int
	resp, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		NoRetry:      true,
		Tools:        []llm.Tool{lookupTool(&runs, "ok")},
		ToolChoice:   llm.ToolChoiceRequired,
		MaxToolCalls: 2,
	})
	if err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if resp.Value != "final" {
		t.Errorf("Value = %q, want final", resp.Value)
	}
	if runs != 2 {
		t.Errorf("resolver ran %d times, want 2", runs)
	}

	wantChoices := []llm.ToolChoice{llm.ToolChoiceRequired, llm.ToolChoiceAuto, llm.ToolChoiceNone}
	if fmt.Sprint(*choices) != fmt.Sprint(wantChoices) {
		t.Errorf("tool choices = %v, want %v", *choices, wantChoices)
	}

	lastRequest := (*requests)[len(*requests)-1]
	lastToolMessage := lastRequest[len(lastRequest)-1]
	if lastToolMessage.Role != "tool" || lastToolMessage.Content != "ok\n\nThe tool call budget is used up, no more tools will run. Answer with what you have." {
		t.Errorf("last message of the final request = %+v, want the tool output followed by the budget notice", lastToolMessage)
	}
	for _, msg := range resp.Conversation {
		if msg.Role == "user" && msg.Content != "hi" {
			t.Errorf("conversation holds user message %q, want only the caller's", msg.Content)
		}
	}
}

func TestToolCallsPastTheBudgetAreAnsweredWithAnError(t *testing.T) {
	provider, _, _ := greedyProvider(3)
	model := &llm.Model{Name: "stub", Provider: provider}

	var runs int
	var results []llm.ToolResult
	resp, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		NoRetry:      true,
		Tools:        []llm.Tool{lookupTool(&runs, "ok")},
		MaxToolCalls: 2,
		OnToolCall:   func(result llm.ToolResult) { results = append(results, result) },
	})
	if err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if runs != 2 {
		t.Errorf("resolver ran %d times, want 2", runs)
	}

	var toolMessages []llm.Message
	for _, msg := range resp.Conversation {
		if msg.Role == "tool" {
			toolMessages = append(toolMessages, msg)
		}
	}
	if len(toolMessages) != 3 {
		t.Fatalf("got %d tool messages, want one per call", len(toolMessages))
	}
	if got := toolMessages[2]; !strings.HasPrefix(got.Content, "error: tool call budget used up\n\n") || got.ToolCallId != "call_1_2" {
		t.Errorf("third tool message = %+v, want the budget error and notice for call_1_2", got)
	}

	if len(results) != 3 || !errors.Is(results[2].Err, llm.ErrToolBudgetUsedUp) {
		t.Errorf("tool results = %+v, want the third to report ErrToolBudgetUsedUp", results)
	}
}

func TestToolCallsWhileToolChoiceIsNoneFail(t *testing.T) {
	provider := &stubProvider{callFn: func(string, []llm.Message, llm.Options) (llm.Turn, error) {
		return toolTurn(lookupCall("call_1")), nil
	}}
	model := &llm.Model{Name: "stub", Provider: provider}

	var runs int
	_, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		NoRetry:    true,
		Tools:      []llm.Tool{lookupTool(&runs, "ok")},
		ToolChoice: llm.ToolChoiceNone,
	})
	if err == nil || !strings.Contains(err.Error(), "tool choice was none") {
		t.Fatalf("Prompt() error = %v, want the tool choice none error", err)
	}
	if runs != 0 {
		t.Errorf("resolver ran %d times, want 0", runs)
	}
}

// stubbornProvider calls a tool on every request, tool choice none included,
// until it has ignored none ignoreNone times.
func stubbornProvider(ignoreNone int) *stubProvider {
	ignored := 0
	provider := &stubProvider{}
	provider.callFn = func(_ string, _ []llm.Message, options llm.Options) (llm.Turn, error) {
		if options.ToolChoice == llm.ToolChoiceNone {
			if ignored == ignoreNone {
				return llm.Turn{Message: llm.Assistant("final"), Usage: llm.TokenUsage{InputTokens: 1}}, nil
			}
			ignored++
		}
		turn := toolTurn(lookupCall(fmt.Sprintf("call_%d", provider.calls.Load())))
		turn.Usage = llm.TokenUsage{InputTokens: 1}
		return turn, nil
	}
	return provider
}

func TestToolCallsPastTheBudgetAreRefusedAndRetried(t *testing.T) {
	provider := stubbornProvider(1)
	model := &llm.Model{Name: "stub", Provider: provider}

	var runs int
	resp, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		NoRetry:      true,
		Tools:        []llm.Tool{lookupTool(&runs, "ok")},
		MaxToolCalls: 1,
	})
	if err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if resp.Value != "final" {
		t.Errorf("Value = %q, want final", resp.Value)
	}
	if runs != 1 {
		t.Errorf("resolver ran %d times, want 1", runs)
	}
	if provider.calls.Load() != 3 {
		t.Errorf("provider calls = %d, want the call, the ignored none and the answer", provider.calls.Load())
	}
}

func TestRepeatedToolCallsPastTheBudgetEndThePromptWithTheConversation(t *testing.T) {
	provider := stubbornProvider(math.MaxInt)
	model := &llm.Model{Name: "stub", Provider: provider}

	var runs int
	resp, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		NoRetry:      true,
		Tools:        []llm.Tool{lookupTool(&runs, "ok")},
		MaxToolCalls: 1,
	})
	if !errors.Is(err, llm.ErrToolBudgetUsedUp) {
		t.Fatalf("Prompt() error = %v, want ErrToolBudgetUsedUp", err)
	}
	if runs != 1 {
		t.Errorf("resolver ran %d times, want 1", runs)
	}
	if provider.calls.Load() != 3 {
		t.Errorf("provider calls = %d, want the call and two ignored nones", provider.calls.Load())
	}
	if resp.Usage.InputTokens != 3 {
		t.Errorf("usage = %+v, want the input tokens of all 3 calls", resp.Usage)
	}

	// user, then 3 times an assistant tool call answered by a tool message
	if len(resp.Conversation) != 7 {
		t.Fatalf("got %d messages, want 7", len(resp.Conversation))
	}
	if last := resp.Conversation[6]; last.Role != "tool" || last.ToolCallId != "call_3" {
		t.Errorf("last message = %+v, want the answer to call_3", last)
	}
}

func TestMaxToolCallsDefaultsToDefaultMaxToolCalls(t *testing.T) {
	provider, _, _ := greedyProvider(1)
	model := &llm.Model{Name: "stub", Provider: provider}

	var runs int
	if _, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		NoRetry: true,
		Tools:   []llm.Tool{lookupTool(&runs, "ok")},
	}); err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if runs != llm.DefaultMaxToolCalls {
		t.Errorf("resolver ran %d times, want %d", runs, llm.DefaultMaxToolCalls)
	}
}

func TestFallbackModelsShareTheToolBudget(t *testing.T) {
	failing := &stubProvider{}
	failing.callFn = func(string, []llm.Message, llm.Options) (llm.Turn, error) {
		if failing.calls.Load() <= 2 {
			return toolTurn(lookupCall("call_1")), nil
		}
		return llm.Turn{}, &llm.Err{StatusCode: 400, Body: "bad"}
	}
	fallback, choices, _ := greedyProvider(1)

	var runs int
	resp, err := llm.NewFallbackModel(
		&llm.Model{Name: "failing", Provider: failing},
		&llm.Model{Name: "fallback", Provider: fallback},
	).Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		Tools:        []llm.Tool{lookupTool(&runs, "ok")},
		MaxToolCalls: 2,
	})
	if err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if resp.Value != "final" {
		t.Errorf("Value = %q, want final", resp.Value)
	}
	if runs != 2 {
		t.Errorf("resolver ran %d times, want 2", runs)
	}
	if want := []llm.ToolChoice{llm.ToolChoiceNone}; fmt.Sprint(*choices) != fmt.Sprint(want) {
		t.Errorf("fallback tool choices = %v, want %v", *choices, want)
	}
}

func TestToolOutputReachesTheModel(t *testing.T) {
	cases := []struct {
		name     string
		tool     llm.Tool
		wantTool string
	}{
		{
			name:     "string is sent as is",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return "line 1\n\"quoted\"", nil }},
			wantTool: "line 1\n\"quoted\"",
		},
		{
			name:     "raw message is sent as is",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return json.RawMessage(`{"a": 1}`), nil }},
			wantTool: `{"a": 1}`,
		},
		{
			name:     "empty string is sent as an empty JSON string",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return "", nil }},
			wantTool: `""`,
		},
		{
			name:     "empty raw message is sent as an empty JSON string",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return json.RawMessage{}, nil }},
			wantTool: `""`,
		},
		{
			name:     "other values are JSON encoded",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return map[string]int{"a": 1}, nil }},
			wantTool: `{"a":1}`,
		},
		{
			name:     "resolver error",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return nil, errors.New("db down") }},
			wantTool: "error: db down",
		},
		{
			name:     "unknown tool",
			tool:     llm.Tool{Function: llm.FunctionDef{Name: "other"}, Resolver: func(json.RawMessage) (any, error) { return "unused", nil }},
			wantTool: "error: tool not found",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider, _, _ := greedyProvider(1)
			model := &llm.Model{Name: "stub", Provider: provider}

			resp, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
				NoRetry:      true,
				Tools:        []llm.Tool{tc.tool},
				MaxToolCalls: 2,
			})
			if err != nil {
				t.Fatalf("Prompt() error = %v", err)
			}

			toolMessage := resp.Conversation[2]
			if toolMessage.Role != "tool" || toolMessage.Content != tc.wantTool {
				t.Errorf("tool message = %+v, want content %q", toolMessage, tc.wantTool)
			}
		})
	}
}

func TestRetryResendsOnlyTheFailedRequest(t *testing.T) {
	provider := &stubProvider{}
	provider.callFn = func(string, []llm.Message, llm.Options) (llm.Turn, error) {
		switch provider.calls.Load() {
		case 1:
			return toolTurn(lookupCall("call_1")), nil
		case 2:
			return llm.Turn{}, &llm.Err{StatusCode: 500, Body: "boom"}
		}
		return llm.Turn{Message: llm.Assistant("final")}, nil
	}
	model := &llm.Model{Name: "stub", Provider: provider}

	type event struct {
		kind           string
		index, attempt int
		retrying       bool
		failed         bool
	}
	var events []event
	var runs int
	resp, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		Tools: []llm.Tool{lookupTool(&runs, "ok")},
		OnCallStart: func(info llm.CallInfo) {
			events = append(events, event{kind: "start", index: info.Index, attempt: info.Attempt})
		},
		OnCallEnd: func(info llm.CallInfo, result llm.CallResult) {
			events = append(events, event{kind: "end", index: info.Index, attempt: info.Attempt, retrying: result.Retrying, failed: result.Err != nil})
		},
		OnToolCall: func(llm.ToolResult) {
			events = append(events, event{kind: "tool"})
		},
	})
	if err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if resp.Value != "final" {
		t.Errorf("Value = %q, want final", resp.Value)
	}
	if runs != 1 {
		t.Errorf("resolver ran %d times, want 1", runs)
	}

	want := []event{
		{kind: "start", index: 0, attempt: 1},
		{kind: "end", index: 0, attempt: 1},
		{kind: "tool"},
		{kind: "start", index: 1, attempt: 1},
		{kind: "end", index: 1, attempt: 1, retrying: true, failed: true},
		{kind: "start", index: 1, attempt: 2},
		{kind: "end", index: 1, attempt: 2},
	}
	if fmt.Sprint(events) != fmt.Sprint(want) {
		t.Errorf("events =\n  %v\nwant\n  %v", events, want)
	}
}

func TestPromptRejectsInvalidOptions(t *testing.T) {
	schema := llm.JsonSchema{Schema: json.RawMessage(`{"type":"object"}`)}
	tools := []llm.Tool{{Function: llm.FunctionDef{Name: "lookup"}, Resolver: func(json.RawMessage) (any, error) { return nil, nil }}}

	cases := []struct {
		name            string
		options         llm.Options
		wantErrContains string
	}{
		{
			name:            "json_schema without a schema",
			options:         llm.Options{ResponseFormat: llm.ResponseFormatJsonSchema},
			wantErrContains: "requires a JsonSchema",
		},
		{
			name:            "schema without json_schema",
			options:         llm.Options{ResponseFormat: llm.ResponseFormatJsonObject, JsonSchema: schema},
			wantErrContains: "requires response format json_schema",
		},
		{
			name:            "required without tools",
			options:         llm.Options{ToolChoice: llm.ToolChoiceRequired},
			wantErrContains: "at least one tool",
		},
		{
			name:            "unknown tool choice",
			options:         llm.Options{Tools: tools, ToolChoice: "sometimes"},
			wantErrContains: "unknown tool choice",
		},
		{
			name:            "negative MaxToolCalls",
			options:         llm.Options{Tools: tools, MaxToolCalls: -1},
			wantErrContains: "cannot be negative",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider := &stubProvider{callFn: okCallProvider("unused")}
			model := &llm.Model{Name: "stub", Provider: provider}

			_, err := model.Prompt([]llm.Message{llm.User("hi")}, tc.options)
			if err == nil || !strings.Contains(err.Error(), tc.wantErrContains) {
				t.Fatalf("Prompt() error = %v, want one containing %q", err, tc.wantErrContains)
			}
			if provider.calls.Load() != 0 {
				t.Errorf("provider called %d times, want 0", provider.calls.Load())
			}
		})
	}
}

func TestJsonSchemaNameDefaultsToResponse(t *testing.T) {
	var got llm.JsonSchema
	provider := &stubProvider{callFn: func(_ string, _ []llm.Message, options llm.Options) (llm.Turn, error) {
		got = options.JsonSchema
		return llm.Turn{Message: llm.Assistant("{}")}, nil
	}}
	model := &llm.Model{Name: "stub", Provider: provider}

	_, err := model.Prompt([]llm.Message{llm.User("hi")}, llm.Options{
		ResponseFormat: llm.ResponseFormatJsonSchema,
		JsonSchema:     llm.JsonSchema{Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("Prompt() error = %v", err)
	}
	if got.Name != "response" {
		t.Errorf("schema name = %q, want response", got.Name)
	}
}
