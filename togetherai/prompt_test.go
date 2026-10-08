package togetherai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

func TestResponseFormatIsSent(t *testing.T) {
	cases := []struct {
		name    string
		options llm.Options
		want    string
	}{
		{
			name:    "json_object",
			options: llm.Options{ResponseFormat: llm.ResponseFormatJsonObject},
			want:    `{"type":"json_object"}`,
		},
		{
			name: "json_schema",
			options: llm.Options{
				ResponseFormat: llm.ResponseFormatJsonSchema,
				JsonSchema:     llm.JsonSchema{Name: "color", Schema: json.RawMessage(`{"type":"object"}`)},
			},
			want: `{"type":"json_schema","json_schema":{"name":"color","schema":{"type":"object"}}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("TOGETHER_AI_TOKEN", "test-token")

			var request map[string]json.RawMessage
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				json.Unmarshal(body, &request)
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{"choices":[{"message":{"content":"{}"}}]}`))
			}))
			defer server.Close()

			prev := BaseURL
			BaseURL = server.URL
			defer func() { BaseURL = prev }()

			p := &Provider{}
			if _, err := p.Call("model", []llm.Message{llm.User("hi")}, tc.options); err != nil {
				t.Fatalf("Call returned error: %v", err)
			}

			if got := string(request["response_format"]); got != tc.want {
				t.Errorf("response_format = %s, want %s", got, tc.want)
			}
		})
	}
}
