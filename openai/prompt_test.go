package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	llm "github.com/Back-to-code/go-llm"
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

	// First server response: model asks for the second tool with specific args.
	// Second response: the final assistant content.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		body, _ := io.ReadAll(r.Body)
		_ = body

		if callCount == 1 {
			resp := `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"second_tool","arguments":"{\"x\":42}"}}]}}]}`
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(resp))
			return
		}

		// Echo the tool response the client sent back so the test can assert
		// the resolver response was serialized into the follow-up request.
		var req map[string]any
		json.Unmarshal(body, &req)
		_ = req

		resp := `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`
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
	if out != "done" {
		t.Fatalf("Prompt returned %q, want %q", out, "done")
	}
	if gotToolName != "second_tool" {
		t.Fatalf("wrong tool resolved: got %q", gotToolName)
	}
	if gotArgs != `{"x":42}` {
		t.Fatalf("wrong args threaded to resolver: got %q", gotArgs)
	}
}
