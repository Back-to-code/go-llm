package inception

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

// serveResponse spins up a stub /v1/chat/completions endpoint returning body,
// and captures the decoded request of the last call it served.
func serveResponse(t *testing.T, body string) *map[string]any {
	t.Helper()
	t.Setenv("INCEPTION_API_KEY", "test-token")

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

	return &request
}

const textResponse = `{"choices":[{"message":{"role":"assistant","content":"{}"}}]}`

func stubTools() []llm.Tool {
	return []llm.Tool{{
		Type:     "function",
		Function: llm.FunctionDef{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)},
		Resolver: func(json.RawMessage) (any, error) { return nil, nil },
	}}
}

func TestJsonSchemaFormatIsSent(t *testing.T) {
	request := serveResponse(t, textResponse)

	p := &Provider{}
	if _, err := p.Call("mercury-2", []llm.Message{llm.User("give me a color")}, llm.Options{
		ResponseFormat: llm.ResponseFormatJsonSchema,
		JsonSchema: llm.JsonSchema{
			Name:   "color",
			Schema: json.RawMessage(`{"type":"object"}`),
			Strict: true,
		},
	}); err != nil {
		t.Fatalf("Call returned error: %v", err)
	}

	format, _ := (*request)["response_format"].(map[string]any)
	schema, _ := format["json_schema"].(map[string]any)
	if format["type"] != "json_schema" || schema["name"] != "color" || schema["strict"] != true || schema["schema"] == nil {
		t.Fatalf("response_format = %v", format)
	}
}

func TestToolChoiceIsSent(t *testing.T) {
	for _, choice := range []llm.ToolChoice{llm.ToolChoiceAuto, llm.ToolChoiceRequired, llm.ToolChoiceNone} {
		t.Run(string(choice), func(t *testing.T) {
			request := serveResponse(t, textResponse)

			p := &Provider{}
			if _, err := p.Call("mercury-2", []llm.Message{llm.User("hi")}, llm.Options{
				Tools:      stubTools(),
				ToolChoice: choice,
			}); err != nil {
				t.Fatalf("Call returned error: %v", err)
			}

			if (*request)["tool_choice"] != string(choice) {
				t.Errorf("tool_choice = %v, want %s", (*request)["tool_choice"], choice)
			}
		})
	}
}

func TestToolCallsAreReturnedUnresolved(t *testing.T) {
	serveResponse(t, `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Amsterdam\"}"}}]}}]}`)

	p := &Provider{}
	turn, err := p.Call("mercury-2", []llm.Message{llm.User("hi")}, llm.Options{Tools: stubTools()})
	if err != nil {
		t.Fatalf("Call returned error: %v", err)
	}

	if len(turn.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v, want 1", turn.ToolCalls)
	}
	call := turn.ToolCalls[0]
	if call.Id != "call_1" || call.Name != "get_weather" || string(call.Arguments) != `{"city":"Amsterdam"}` {
		t.Errorf("tool call = %+v", call)
	}
	if turn.Message.Role != "assistant" || len(turn.Message.ToolCalls) == 0 {
		t.Errorf("message = %+v, want an assistant message carrying the raw tool calls", turn.Message)
	}
}

func TestReasoningEffortIsSentAlongsideTools(t *testing.T) {
	request := serveResponse(t, textResponse)

	p := &Provider{}
	if _, err := p.Call("mercury-2.5", []llm.Message{llm.User("hi")}, llm.Options{
		Tools:    stubTools(),
		Thinking: llm.HighThinking,
	}); err != nil {
		t.Fatalf("Call returned error: %v", err)
	}

	if (*request)["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", (*request)["reasoning_effort"])
	}
}
