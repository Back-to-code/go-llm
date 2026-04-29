package llm_test

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/googleaistudio"
	"github.com/Back-to-code/go-llm/inception"
	"github.com/Back-to-code/go-llm/openai"
	"github.com/Back-to-code/go-llm/togetherai"
	"github.com/joho/godotenv"
)

func init() {
	// Load .env file if present; ignore errors (env vars may already be set).
	_ = godotenv.Load()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func skipIfEnvMissing(t *testing.T, key string) {
	t.Helper()
	if strings.TrimSpace(os.Getenv(key)) == "" {
		t.Skipf("skipping: %s environment variable not set", key)
	}
}

// weatherTool returns a simple tool that the model can call.
// It accepts a JSON object with a "city" field and returns a static forecast.
func weatherTool() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.FunctionDef{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"city": {
						"type": "string",
						"description": "The city name"
					}
				},
				"required": ["city"]
			}`),
		},
		Resolver: func(args json.RawMessage) (any, error) {
			return map[string]any{
				"city":        "Amsterdam",
				"temperature": "18°C",
				"condition":   "Partly cloudy",
			}, nil
		},
	}
}

// assertResponse validates all the common invariants of a Response.
func assertResponse(t *testing.T, resp llm.Response, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if resp.Value == "" {
		t.Fatal("expected non-empty Value")
	}
	if resp.String() != resp.Value {
		t.Fatalf("String() = %q, want %q", resp.String(), resp.Value)
	}
	if len(resp.Conversation) == 0 {
		t.Fatal("expected non-empty Conversation")
	}

	// The last message in the conversation should be the assistant response.
	last := resp.Conversation[len(resp.Conversation)-1]
	if last.Role != "assistant" {
		t.Fatalf("expected last conversation message role to be 'assistant', got %q", last.Role)
	}
	if last.Content != resp.Value {
		t.Fatalf("expected last conversation message content to equal Value:\n  content = %q\n  value   = %q", last.Content, resp.Value)
	}
}

func assertUsageNonZero(t *testing.T, usage llm.TokenUsage) {
	t.Helper()
	if usage.InputTokens <= 0 {
		t.Errorf("expected InputTokens > 0, got %d", usage.InputTokens)
	}
	if usage.OutputTokens <= 0 {
		t.Errorf("expected OutputTokens > 0, got %d", usage.OutputTokens)
	}
}

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------

func TestOpenAI(t *testing.T) {
	skipIfEnvMissing(t, "OPENAI_TOKEN")

	provider := &openai.Provider{}
	model := &llm.Model{Name: "gpt-5.4-nano", Provider: provider}

	t.Run("PromptSingle", func(t *testing.T) {
		resp, err := model.PromptSingle("Reply with only the word 'hello'.", llm.Options{NoRetry: true})
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if !strings.Contains(strings.ToLower(resp.Value), "hello") {
			t.Errorf("expected response to contain 'hello', got: %q", resp.Value)
		}

		// Conversation should have at least user + assistant.
		if len(resp.Conversation) < 2 {
			t.Fatalf("expected at least 2 messages in conversation, got %d", len(resp.Conversation))
		}
		if resp.Conversation[0].Role != "user" {
			t.Errorf("expected first message role 'user', got %q", resp.Conversation[0].Role)
		}
	})

	t.Run("Prompt", func(t *testing.T) {
		messages := []llm.Message{
			llm.System("You are a helpful assistant. Always reply in one short sentence."),
			llm.User("What is 2+2?"),
		}
		resp, err := model.Prompt(messages, llm.Options{NoRetry: true})
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		// Conversation should have system + user + assistant = at least 3.
		if len(resp.Conversation) < 3 {
			t.Fatalf("expected at least 3 messages in conversation, got %d", len(resp.Conversation))
		}
	})

	t.Run("PromptWithTools", func(t *testing.T) {
		resp, err := model.Prompt(
			[]llm.Message{
				llm.System("You have access to a weather tool. Use it to answer the question. After getting the result, reply with a short sentence."),
				llm.User("What is the weather in Amsterdam?"),
			},
			llm.Options{
				NoRetry: true,
				Tools:   []llm.Tool{weatherTool()},
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		// With tool calls the conversation should contain intermediate messages:
		// system, user, assistant (tool_calls), tool (response), assistant (final).
		// That's at least 5 messages.
		if len(resp.Conversation) < 5 {
			t.Fatalf("expected at least 5 messages in conversation with tool calls, got %d", len(resp.Conversation))
		}

		// Verify tool call messages exist in the conversation.
		hasToolCall := false
		hasToolResponse := false
		for _, msg := range resp.Conversation {
			if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
				hasToolCall = true
			}
			if msg.Role == "tool" {
				hasToolResponse = true
			}
		}
		if !hasToolCall {
			t.Error("expected at least one assistant message with ToolCalls in conversation")
		}
		if !hasToolResponse {
			t.Error("expected at least one tool response message in conversation")
		}

		// Usage should be accumulated across the tool-call round-trips,
		// so input tokens should be higher than a simple prompt because
		// the conversation grew between rounds.
		if resp.Usage.InputTokens < 10 {
			t.Errorf("expected accumulated InputTokens to be significant, got %d", resp.Usage.InputTokens)
		}
	})

	t.Run("PromptJSON", func(t *testing.T) {
		resp, err := model.PromptSingle(
			`Return a JSON object with a single key "color" and value "blue". No other text.`,
			llm.Options{
				NoRetry:        true,
				ResponseFormat: llm.ResponseFormatJsonObject,
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		var parsed map[string]string
		if err := json.Unmarshal([]byte(resp.Value), &parsed); err != nil {
			t.Fatalf("expected valid JSON response, got parse error: %v\nraw: %q", err, resp.Value)
		}
		if parsed["color"] != "blue" {
			t.Errorf("expected color=blue, got %q", parsed["color"])
		}
	})

	t.Run("YesNo", func(t *testing.T) {
		result, err := llm.YesNo(model.PromptSingle("Is the sky blue? Reply with only yes or no.", llm.Options{NoRetry: true}))
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if !result {
			t.Error("expected YesNo to return true for 'is the sky blue'")
		}
	})
}

// ---------------------------------------------------------------------------
// Google AI Studio
// ---------------------------------------------------------------------------

func TestGoogleAIStudio(t *testing.T) {
	skipIfEnvMissing(t, "GOOGLE_AI_STUDIO_KEY")

	provider := &googleaistudio.Provider{}
	model := &llm.Model{Name: "gemini-3.1-flash-lite-preview", Provider: provider}

	t.Run("PromptSingle", func(t *testing.T) {
		resp, err := model.PromptSingle("Reply with only the word 'hello'.", llm.Options{NoRetry: true})
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if !strings.Contains(strings.ToLower(resp.Value), "hello") {
			t.Errorf("expected response to contain 'hello', got: %q", resp.Value)
		}

		if len(resp.Conversation) < 2 {
			t.Fatalf("expected at least 2 messages in conversation, got %d", len(resp.Conversation))
		}
		if resp.Conversation[0].Role != "user" {
			t.Errorf("expected first message role 'user', got %q", resp.Conversation[0].Role)
		}
	})

	t.Run("Prompt", func(t *testing.T) {
		messages := []llm.Message{
			llm.System("You are a helpful assistant. Always reply in one short sentence."),
			llm.User("What is 2+2?"),
		}
		resp, err := model.Prompt(messages, llm.Options{NoRetry: true})
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		// System message is extracted as system_instruction in Google AI Studio,
		// so the conversation passed to the provider only has user + assistant.
		// But from Model.Prompt's perspective the original messages are preserved.
		if len(resp.Conversation) < 3 {
			t.Fatalf("expected at least 3 messages in conversation, got %d", len(resp.Conversation))
		}
	})

	t.Run("PromptWithTools", func(t *testing.T) {
		resp, err := model.Prompt(
			[]llm.Message{
				llm.System("You have access to a weather tool. Use it to answer the question. After getting the result, reply with a short sentence."),
				llm.User("What is the weather in Amsterdam?"),
			},
			llm.Options{
				NoRetry: true,
				Tools:   []llm.Tool{weatherTool()},
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if len(resp.Conversation) < 5 {
			t.Fatalf("expected at least 5 messages in conversation with tool calls, got %d", len(resp.Conversation))
		}

		hasToolCall := false
		hasToolResponse := false
		for _, msg := range resp.Conversation {
			if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
				hasToolCall = true
			}
			if msg.Role == "tool" {
				hasToolResponse = true
			}
		}
		if !hasToolCall {
			t.Error("expected at least one assistant message with ToolCalls in conversation")
		}
		if !hasToolResponse {
			t.Error("expected at least one tool response message in conversation")
		}
	})

	t.Run("PromptJSON", func(t *testing.T) {
		resp, err := model.PromptSingle(
			`Return a JSON object with a single key "color" and value "blue". No other text.`,
			llm.Options{
				NoRetry:        true,
				ResponseFormat: llm.ResponseFormatJsonObject,
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		var parsed map[string]string
		if err := json.Unmarshal([]byte(resp.Value), &parsed); err != nil {
			t.Fatalf("expected valid JSON response, got parse error: %v\nraw: %q", err, resp.Value)
		}
		if parsed["color"] != "blue" {
			t.Errorf("expected color=blue, got %q", parsed["color"])
		}
	})

	t.Run("YesNo", func(t *testing.T) {
		result, err := llm.YesNo(model.PromptSingle("Is the sky blue? Reply with only yes or no.", llm.Options{NoRetry: true}))
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if !result {
			t.Error("expected YesNo to return true for 'is the sky blue'")
		}
	})
}

// ---------------------------------------------------------------------------
// Together AI
// ---------------------------------------------------------------------------

func TestTogetherAI(t *testing.T) {
	skipIfEnvMissing(t, "TOGETHER_AI_TOKEN")

	provider := &togetherai.Provider{}
	model := &llm.Model{Name: "Qwen/Qwen3.5-9B", Provider: provider}

	// Together AI can be slow/flaky, use a longer timeout and allow retries.
	opts := llm.Options{Timeout: 60 * time.Second}

	t.Run("PromptSingle", func(t *testing.T) {
		resp, err := model.PromptSingle("Reply with only the word 'hello'.", opts)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if !strings.Contains(strings.ToLower(resp.Value), "hello") {
			t.Errorf("expected response to contain 'hello', got: %q", resp.Value)
		}

		if len(resp.Conversation) < 2 {
			t.Fatalf("expected at least 2 messages in conversation, got %d", len(resp.Conversation))
		}
		if resp.Conversation[0].Role != "user" {
			t.Errorf("expected first message role 'user', got %q", resp.Conversation[0].Role)
		}
	})

	t.Run("Prompt", func(t *testing.T) {
		messages := []llm.Message{
			llm.System("You are a helpful assistant. Always reply in one short sentence."),
			llm.User("What is 2+2?"),
		}
		resp, err := model.Prompt(messages, opts)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if len(resp.Conversation) < 3 {
			t.Fatalf("expected at least 3 messages in conversation, got %d", len(resp.Conversation))
		}
	})

	t.Run("PromptJSON", func(t *testing.T) {
		resp, err := model.PromptSingle(
			`Return a JSON object with a single key "color" and value "blue". No other text.`,
			llm.Options{
				Timeout:        60 * time.Second,
				ResponseFormat: llm.ResponseFormatJsonObject,
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		var parsed map[string]string
		if err := json.Unmarshal([]byte(resp.Value), &parsed); err != nil {
			t.Fatalf("expected valid JSON response, got parse error: %v\nraw: %q", err, resp.Value)
		}
		if parsed["color"] != "blue" {
			t.Errorf("expected color=blue, got %q", parsed["color"])
		}
	})

	t.Run("YesNo", func(t *testing.T) {
		result, err := llm.YesNo(model.PromptSingle("Is the sky blue? Reply with only yes or no.", opts))
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if !result {
			t.Error("expected YesNo to return true for 'is the sky blue'")
		}
	})

	// Together AI does not support tools, verify it errors correctly.
	t.Run("ToolsUnsupported", func(t *testing.T) {
		_, err := model.Prompt(
			[]llm.Message{llm.User("What is the weather?")},
			llm.Options{
				NoRetry: true,
				Tools:   []llm.Tool{weatherTool()},
			},
		)
		if err == nil {
			t.Fatal("expected error when using tools with Together AI provider")
		}
		if !strings.Contains(err.Error(), "does not support tools") {
			t.Errorf("expected 'does not support tools' error, got: %v", err)
		}
	})

	// Together AI does not support streaming, verify it errors correctly.
	t.Run("StreamUnsupported", func(t *testing.T) {
		_, err := model.Stream(
			[]llm.Message{llm.User("Hello")},
			llm.Options{NoRetry: true},
		)
		if err == nil {
			t.Fatal("expected error when using streaming with Together AI provider")
		}
		if !strings.Contains(err.Error(), "does not support streaming") {
			t.Errorf("expected 'does not support streaming' error, got: %v", err)
		}
	})
}

// ---------------------------------------------------------------------------
// Inception
// ---------------------------------------------------------------------------

func TestInception(t *testing.T) {
	skipIfEnvMissing(t, "INCEPTION_API_KEY")

	provider := &inception.Provider{}
	model := &llm.Model{Name: "mercury-2", Provider: provider}

	t.Run("PromptSingle", func(t *testing.T) {
		resp, err := model.PromptSingle("Reply with only the word 'hello'.", llm.Options{NoRetry: true})
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if !strings.Contains(strings.ToLower(resp.Value), "hello") {
			t.Errorf("expected response to contain 'hello', got: %q", resp.Value)
		}

		if len(resp.Conversation) < 2 {
			t.Fatalf("expected at least 2 messages in conversation, got %d", len(resp.Conversation))
		}
		if resp.Conversation[0].Role != "user" {
			t.Errorf("expected first message role 'user', got %q", resp.Conversation[0].Role)
		}
	})

	t.Run("Prompt", func(t *testing.T) {
		messages := []llm.Message{
			llm.System("You are a helpful assistant. Always reply in one short sentence."),
			llm.User("What is 2+2?"),
		}
		resp, err := model.Prompt(messages, llm.Options{NoRetry: true})
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if len(resp.Conversation) < 3 {
			t.Fatalf("expected at least 3 messages in conversation, got %d", len(resp.Conversation))
		}
	})

	t.Run("PromptWithTools", func(t *testing.T) {
		resp, err := model.Prompt(
			[]llm.Message{
				llm.System("You have access to a weather tool. Use it to answer the question. After getting the result, reply with a short sentence."),
				llm.User("What is the weather in Amsterdam?"),
			},
			llm.Options{
				NoRetry: true,
				Tools:   []llm.Tool{weatherTool()},
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		if len(resp.Conversation) < 5 {
			t.Fatalf("expected at least 5 messages in conversation with tool calls, got %d", len(resp.Conversation))
		}

		hasToolCall := false
		hasToolResponse := false
		for _, msg := range resp.Conversation {
			if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
				hasToolCall = true
			}
			if msg.Role == "tool" {
				hasToolResponse = true
			}
		}
		if !hasToolCall {
			t.Error("expected at least one assistant message with ToolCalls in conversation")
		}
		if !hasToolResponse {
			t.Error("expected at least one tool response message in conversation")
		}
	})

	t.Run("PromptJSON", func(t *testing.T) {
		resp, err := model.PromptSingle(
			`Return a JSON object with a single key "color" and value "blue". No other text.`,
			llm.Options{
				NoRetry:        true,
				ResponseFormat: llm.ResponseFormatJsonObject,
			},
		)
		assertResponse(t, resp, err)
		assertUsageNonZero(t, resp.Usage)

		var parsed map[string]string
		if err := json.Unmarshal([]byte(resp.Value), &parsed); err != nil {
			t.Fatalf("expected valid JSON response, got parse error: %v\nraw: %q", err, resp.Value)
		}
		if parsed["color"] != "blue" {
			t.Errorf("expected color=blue, got %q", parsed["color"])
		}
	})

	t.Run("YesNo", func(t *testing.T) {
		result, err := llm.YesNo(model.PromptSingle("Is the sky blue? Reply with only yes or no.", llm.Options{NoRetry: true}))
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if !result {
			t.Error("expected YesNo to return true for 'is the sky blue'")
		}
	})
}
