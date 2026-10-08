package googleaistudio

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

// serveResponse spins up a stub generateContent endpoint returning body, and
// captures the decoded request of the last call it served.
func serveResponse(t *testing.T, body string) *map[string]any {
	t.Helper()
	t.Setenv("GOOGLE_AI_STUDIO_KEY", "test-key")

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

const textResponse = `{"candidates":[{"content":{"role":"model","parts":[{"text":"{}"}]}}]}`

func stubTools() []llm.Tool {
	return []llm.Tool{{
		Function: llm.FunctionDef{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)},
		Resolver: func(json.RawMessage) (any, error) { return nil, nil },
	}}
}

func TestJsonSchemaIsSent(t *testing.T) {
	request := serveResponse(t, textResponse)

	p := &Provider{}
	if _, err := p.Call("gemini-3.5-flash-lite", []llm.Message{llm.User("give me a color")}, llm.Options{
		ResponseFormat: llm.ResponseFormatJsonSchema,
		JsonSchema:     llm.JsonSchema{Name: "color", Schema: json.RawMessage(`{"type":"object","properties":{"color":{"type":"string"}}}`)},
	}); err != nil {
		t.Fatalf("Call returned error: %v", err)
	}

	config, _ := (*request)["generationConfig"].(map[string]any)
	if config["response_mime_type"] != "application/json" {
		t.Errorf("response_mime_type = %v, want application/json", config["response_mime_type"])
	}
	if _, ok := config["responseJsonSchema"].(map[string]any)["properties"]; !ok {
		t.Errorf("responseJsonSchema = %v, want the schema", config["responseJsonSchema"])
	}
}

func TestToolChoiceIsSentAsFunctionCallingMode(t *testing.T) {
	cases := []struct {
		choice   llm.ToolChoice
		wantMode any
	}{
		{choice: llm.ToolChoiceAuto, wantMode: nil},
		{choice: llm.ToolChoiceRequired, wantMode: "ANY"},
		{choice: llm.ToolChoiceNone, wantMode: "NONE"},
	}

	for _, tc := range cases {
		t.Run(string(tc.choice), func(t *testing.T) {
			request := serveResponse(t, textResponse)

			p := &Provider{}
			if _, err := p.Call("gemini-3.5-flash-lite", []llm.Message{llm.User("hi")}, llm.Options{
				Tools:      stubTools(),
				ToolChoice: tc.choice,
			}); err != nil {
				t.Fatalf("Call returned error: %v", err)
			}

			var mode any
			if toolConfig, ok := (*request)["toolConfig"].(map[string]any); ok {
				mode = toolConfig["functionCallingConfig"].(map[string]any)["mode"]
			}
			if mode != tc.wantMode {
				t.Errorf("functionCallingConfig.mode = %v, want %v", mode, tc.wantMode)
			}
		})
	}
}

func TestToolRoundTrip(t *testing.T) {
	serveResponse(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Amsterdam"}},"thoughtSignature":"sig"}]}}]}`)

	p := &Provider{}
	turn, err := p.Call("gemini-3.5-flash-lite", []llm.Message{llm.User("hi")}, llm.Options{Tools: stubTools()})
	if err != nil {
		t.Fatalf("Call returned error: %v", err)
	}
	if len(turn.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v, want 1", turn.ToolCalls)
	}
	call := turn.ToolCalls[0]
	if call.Id != "get_weather#0" || call.Name != "get_weather" || string(call.Arguments) != `{"city":"Amsterdam"}` {
		t.Errorf("tool call = %+v", call)
	}

	request := serveResponse(t, textResponse)
	conversation := []llm.Message{
		llm.User("hi"),
		turn.Message,
		{Role: "tool", Content: "sunny\n18 degrees", ToolCallId: call.Id},
	}
	if _, err := p.Call("gemini-3.5-flash-lite", conversation, llm.Options{Tools: stubTools()}); err != nil {
		t.Fatalf("Call returned error: %v", err)
	}

	contents, _ := (*request)["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %v, want user, model and function response", contents)
	}
	modelPart := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if modelPart["thoughtSignature"] != "sig" {
		t.Errorf("model part = %v, want the thought signature replayed", modelPart)
	}
	responsePart := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if responsePart["name"] != "get_weather" || responsePart["response"].(map[string]any)["output"] != "sunny\n18 degrees" {
		t.Errorf("function response = %v, want the plain text output", responsePart)
	}
}

func TestParallelCallsToOneFunctionGetTheirOwnIds(t *testing.T) {
	serveResponse(t, `{"candidates":[{"content":{"role":"model","parts":[
		{"functionCall":{"name":"get_weather","args":{"city":"Amsterdam"}}},
		{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}},
		{"functionCall":{"id":"fc_berlin","name":"get_weather","args":{"city":"Berlin"}}}
	]}}]}`)

	p := &Provider{}
	turn, err := p.Call("gemini-3.5-flash-lite", []llm.Message{llm.User("hi")}, llm.Options{Tools: stubTools()})
	if err != nil {
		t.Fatalf("Call returned error: %v", err)
	}

	var ids []string
	conversation := []llm.Message{llm.User("hi"), turn.Message}
	for _, call := range turn.ToolCalls {
		ids = append(ids, call.Id)
		conversation = append(conversation, llm.Message{Role: "tool", Content: "sunny", ToolCallId: call.Id})
	}
	if want := []string{"get_weather#0", "get_weather#1", "fc_berlin"}; fmt.Sprint(ids) != fmt.Sprint(want) {
		t.Fatalf("tool call ids = %v, want %v", ids, want)
	}

	contents, _, err := convertMessages(conversation)
	if err != nil {
		t.Fatalf("convertMessages returned error: %v", err)
	}
	var responses []string
	for _, part := range contents[2].Parts {
		responses = append(responses, part.FunctionResponse.Name+"/"+part.FunctionResponse.Id)
	}
	if want := []string{"get_weather/", "get_weather/", "get_weather/fc_berlin"}; fmt.Sprint(responses) != fmt.Sprint(want) {
		t.Errorf("function responses = %v, want %v", responses, want)
	}
}

func TestUserMessageAfterFunctionResponsesJoinsTheirTurn(t *testing.T) {
	contents, _, err := convertMessages([]llm.Message{
		llm.User("hi"),
		{Role: "assistant", ToolCalls: json.RawMessage(`[{"functionCall":{"name":"a","args":{}}},{"functionCall":{"name":"b","args":{}}}]`)},
		{Role: "tool", Content: "1", ToolCallId: "a"},
		{Role: "tool", Content: "2", ToolCallId: "b"},
		llm.User("answer now"),
	})
	if err != nil {
		t.Fatalf("convertMessages returned error: %v", err)
	}

	if len(contents) != 3 {
		t.Fatalf("got %d contents, want user, model and one user turn holding both responses and the text", len(contents))
	}
	parts := contents[2].Parts
	if len(parts) != 3 || parts[0].FunctionResponse == nil || parts[1].FunctionResponse == nil || parts[2].Text != "answer now" {
		t.Errorf("last turn parts = %+v", parts)
	}
}
