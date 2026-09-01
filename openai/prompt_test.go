package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	llm "github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/log"
)

// Test the tool resolver loop threads the correct arguments to the correct
// resolver, and propagates resolver errors back to the model. Historical bug:
// the inner `err :=` in the resolver loop shadowed the outer, and resolver
// errors were silently swallowed. Also, an earlier iteration of the loop
// compared `tool.Function.Name` against itself (`tool.Function.Name ==
// tool.Function.Name`), matching the first registered tool regardless of the
// call. Both are regression targets here.
func TestPromptToolResolverThreadsArgsAndErrors(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	var callCount int
	var gotToolName string
	var gotArgs string
	var secondRequest map[string]any

	// First server response: model asks for the second tool with specific args.
	// Second response: the final assistant content.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if r.URL.Path != "/v1/responses" {
			t.Errorf("request went to %s, want /v1/responses", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)

		if callCount == 1 {
			resp := `{"status":"completed","output":[` +
				`{"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"secret"},` +
				`{"id":"fc_1","type":"function_call","call_id":"call_1","name":"second_tool","arguments":"{\"x\":42}"}` +
				`]}`
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(resp))
			return
		}

		json.Unmarshal(body, &secondRequest)

		resp := `{"status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(resp))
	}))
	defer server.Close()

	prev := BaseURL
	BaseURL = server.URL
	defer func() { BaseURL = prev }()

	tools := []llm.Tool{
		{
			Function: llm.FunctionDef{
				Name:       "first_tool",
				Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`),
			},
			Resolver: func(args json.RawMessage) (any, error) {
				t.Fatalf("first_tool should not be called; got args %s", string(args))
				return nil, nil
			},
		},
		{
			Function: llm.FunctionDef{
				Name:       "second_tool",
				Parameters: json.RawMessage(`{"type":"object","properties":{"x":{"type":"number"}}}`),
			},
			Resolver: func(args json.RawMessage) (any, error) {
				gotToolName = "second_tool"
				gotArgs = string(args)
				return map[string]string{"status": "ok"}, nil
			},
		},
	}

	p := &Provider{}
	out, err := p.Prompt("gpt-test", []llm.Message{llm.User("hi")}, llm.Options{Tools: tools})
	if err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}
	if out.Value != "done" {
		t.Fatalf("Prompt returned %q, want %q", out.Value, "done")
	}
	if gotToolName != "second_tool" {
		t.Fatalf("wrong tool resolved: got %q", gotToolName)
	}
	if gotArgs != `{"x":42}` {
		t.Fatalf("wrong args threaded to resolver: got %q", gotArgs)
	}

	// The follow up request must replay the reasoning item and the function
	// call, and add the function_call_output for the resolved tool.
	input, _ := secondRequest["input"].([]any)
	var types []string
	for _, item := range input {
		entry, _ := item.(map[string]any)
		itemType, _ := entry["type"].(string)
		if itemType == "" {
			itemType = "message:" + entry["role"].(string)
		}
		types = append(types, itemType)
	}
	want := []string{"message:user", "reasoning", "function_call", "function_call_output"}
	if len(types) != len(want) {
		t.Fatalf("follow up input items = %v, want %v", types, want)
	}
	for idx := range want {
		if types[idx] != want[idx] {
			t.Fatalf("follow up input items = %v, want %v", types, want)
		}
	}
	lastItem := input[len(input)-1].(map[string]any)
	if lastItem["call_id"] != "call_1" {
		t.Fatalf("function_call_output call_id = %v, want call_1", lastItem["call_id"])
	}
	if lastItem["output"] != `{"status":"ok"}` {
		t.Fatalf("function_call_output output = %v", lastItem["output"])
	}
}

// Reasoning effort and function tools have to be sent together. On
// /v1/chat/completions OpenAI rejects that combination since gpt-5.4, which is
// why this provider uses /v1/responses.
func TestReasoningEffortIsSentAlongsideTools(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &request)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`))
	}))
	defer server.Close()

	prev := BaseURL
	BaseURL = server.URL
	defer func() { BaseURL = prev }()

	p := &Provider{}
	_, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{
		Tools:    stubTools(),
		Thinking: llm.MediumThinking,
	})
	if err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}

	reasoning, ok := request["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "medium" {
		t.Fatalf("reasoning = %v, want effort medium", request["reasoning"])
	}

	// Tools are flat on the Responses API, no nested "function" object.
	requestTools, _ := request["tools"].([]any)
	if len(requestTools) != 1 {
		t.Fatalf("tools = %v, want 1 tool", request["tools"])
	}
	tool := requestTools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" {
		t.Fatalf("tool = %v", tool)
	}
	if _, nested := tool["function"]; nested {
		t.Fatalf("tool still uses the chat/completions nesting: %v", tool)
	}

	// store is false, so reasoning has to come back encrypted to be replayable.
	include, _ := request["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %v, want [reasoning.encrypted_content]", request["include"])
	}
}

func stubTools() []llm.Tool {
	return []llm.Tool{{
		Function: llm.FunctionDef{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
		},
		Resolver: func(json.RawMessage) (any, error) { return nil, nil },
	}}
}

// serveResponse spins up a stub /v1/responses endpoint returning body, and
// captures the decoded request of the last call it served.
func serveResponse(t *testing.T, body string) (*httptest.Server, *map[string]any) {
	t.Helper()

	request := map[string]any{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqBody, _ := io.ReadAll(r.Body)
		json.Unmarshal(reqBody, &request)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(body))
	}))

	prev := BaseURL
	BaseURL = server.URL
	t.Cleanup(func() {
		BaseURL = prev
		server.Close()
	})

	return server, &request
}

// Pro models are sent no effort, yet they reason, so the encrypted reasoning
// has to be requested off the model rather than off the effort: without it the
// replayed reasoning items of a tool call round are rejected with "Item with id
// 'rs_...' not found. Items are not persisted when `store` is set to false."
func TestEncryptedReasoningIsRequestedForProModels(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	_, request := serveResponse(t, `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`)

	p := &Provider{}
	if _, err := p.Prompt("gpt-5-pro", []llm.Message{llm.User("hi")}, llm.Options{
		Thinking: llm.HighThinking,
		Tools:    stubTools(),
	}); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}

	if _, ok := (*request)["reasoning"]; ok {
		t.Errorf("reasoning = %v, want it absent for a fixed effort model", (*request)["reasoning"])
	}
	include, _ := (*request)["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %v, want [reasoning.encrypted_content]", (*request)["include"])
	}
}

func TestEncryptedReasoningIsNotRequestedForNonReasoningModels(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	_, request := serveResponse(t, `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`)

	p := &Provider{}
	if _, err := p.Prompt("gpt-4o", []llm.Message{llm.User("hi")}, llm.Options{
		Thinking: llm.HighThinking,
		Tools:    stubTools(),
	}); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}

	if _, ok := (*request)["include"]; ok {
		t.Errorf("include = %v, want it absent for a non reasoning model", (*request)["include"])
	}
}

// Without tools the output items are never replayed, only the text is read off
// them, so paying for a high effort encrypted reasoning blob buys nothing.
func TestEncryptedReasoningIsNotRequestedWithoutTools(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	_, request := serveResponse(t, `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`)

	p := &Provider{}
	if _, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{Thinking: llm.HighThinking}); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}

	if _, ok := (*request)["include"]; ok {
		t.Errorf("include = %v, want it absent without tools", (*request)["include"])
	}
}

// The zero value of Options.Thinking has to leave the effort out of the request
// entirely rather than pick a level on the caller's behalf.
func TestAutoThinkingSendsNoReasoningConfig(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	_, request := serveResponse(t, `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`)

	p := &Provider{}
	if _, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{}); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}

	if _, ok := (*request)["reasoning"]; ok {
		t.Errorf("reasoning = %v, want it absent for AutoThinking", (*request)["reasoning"])
	}
}

// A truncated response returns its partial text instead of an error: an error
// sends the retry loop through five more full length requests.
func TestIncompleteResponseReturnsPartialText(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	serveResponse(t, `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"half a sen"}]}]}`)

	p := &Provider{}
	resp, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{})
	if err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}
	if resp.Value != "half a sen" {
		t.Fatalf("Value = %q, want the partial text", resp.Value)
	}
}

// A truncated response holding only a half written function call must not be
// fed back into the tool loop.
func TestIncompleteResponseWithoutTextSkipsToolCalls(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	var callCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"cit"}]}`))
	}))
	defer server.Close()

	prev := BaseURL
	BaseURL = server.URL
	defer func() { BaseURL = prev }()

	tools := []llm.Tool{{
		Function: llm.FunctionDef{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object"}`)},
		Resolver: func(json.RawMessage) (any, error) {
			t.Error("resolver called for a truncated function call")
			return nil, nil
		},
	}}

	p := &Provider{}
	_, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{Tools: tools})
	if err == nil || !strings.Contains(err.Error(), "max_output_tokens") {
		t.Fatalf("err = %v, want it to name max_output_tokens", err)
	}
	if callCount != 1 {
		t.Errorf("served %d requests, want 1", callCount)
	}
}

// The API rejects a json_object format when the input never says "json", so the
// provider has to catch it before the request goes out.
func TestJsonObjectFormatRequiresTheWordJson(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	var callCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"{}"}]}]}`))
	}))
	defer server.Close()

	prev := BaseURL
	BaseURL = server.URL
	defer func() { BaseURL = prev }()

	p := &Provider{}
	_, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("give me the color blue as an object")}, llm.Options{
		ResponseFormat: llm.ResponseFormatJsonObject,
	})
	if err == nil || !strings.Contains(err.Error(), "json") {
		t.Fatalf("err = %v, want it to name the json requirement", err)
	}
	if callCount != 0 {
		t.Errorf("served %d requests, want the request withheld", callCount)
	}

	if _, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("reply as JSON")}, llm.Options{
		ResponseFormat: llm.ResponseFormatJsonObject,
	}); err != nil {
		t.Fatalf("Prompt returned error once the input mentions JSON: %v", err)
	}
}

// A refusal is the message content instead of the text, so reporting it as
// "missing content" loses the only explanation there is.
func TestRefusalIsReported(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	serveResponse(t, `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}]}`)

	p := &Provider{}
	_, err := p.Prompt("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{})
	if err == nil || !strings.Contains(err.Error(), "I can't help with that.") {
		t.Fatalf("err = %v, want the refusal text", err)
	}
}

// Two regressions at once: an SSE event far larger than a fixed line buffer
// must not break the deltas around it, and a truncated stream must say so.
func TestStreamSurvivesLargeEventsAndReportsIncomplete(t *testing.T) {
	os.Setenv("OPENAI_TOKEN", "test-token")
	defer os.Unsetenv("OPENAI_TOKEN")

	padding := strings.Repeat("x", 200000)
	events := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"he\"}\n\n" +
		"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"text\":\"" + padding + "\"}}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"llo\"}\n\n" +
		"data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(events))
	}))
	defer server.Close()

	prev := BaseURL
	BaseURL = server.URL
	defer func() { BaseURL = prev }()

	var logged []string
	prevLogger := log.Logger
	log.Logger = func(msg string) { logged = append(logged, msg) }
	defer func() { log.Logger = prevLogger }()

	p := &Provider{}
	lines, err := p.Stream("gpt-5.4-mini", []llm.Message{llm.User("hi")}, llm.Options{})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	var streamed strings.Builder
	for line := range lines {
		streamed.WriteString(line)
	}
	if streamed.String() != "hello" {
		t.Errorf("streamed %q, want \"hello\"", streamed.String())
	}

	if !strings.Contains(strings.Join(logged, "\n"), "llm stream incomplete: max_output_tokens") {
		t.Errorf("logged %v, want the incomplete reason", logged)
	}
}
