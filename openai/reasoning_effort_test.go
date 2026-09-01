package openai

import (
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

var allThinking = []llm.Thinking{
	llm.AutoThinking,
	llm.NoThinking,
	llm.MinimalThinking,
	llm.LowThinking,
	llm.MediumThinking,
	llm.HighThinking,
}

// Verified against the live API: gpt-5.4-pro and gpt-5.5-pro answer
// "'none' is not supported … Supported values are: 'medium', 'high'", and
// chat-latest answers "Supported values are: 'medium'". Sending no effort at
// all is accepted by every one of them.
func TestNarrowEffortModelsGetNoEffort(t *testing.T) {
	models := []string{
		"gpt-5-pro",
		"gpt-5.4-pro",
		"gpt-5.5-pro",
		"o1-pro",
		"o3-pro",
		"chat-latest",
		"gpt-5-chat-latest",
	}

	for _, model := range models {
		for _, thinking := range allThinking {
			if effort := reasoningEffort(model, thinking); effort != "" {
				t.Errorf("reasoningEffort(%q, %v) = %q, want no effort", model, thinking, effort)
			}
		}
	}
}

func TestNonReasoningModelsGetNoEffort(t *testing.T) {
	for _, model := range []string{"gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"} {
		for _, thinking := range allThinking {
			if effort := reasoningEffort(model, thinking); effort != "" {
				t.Errorf("reasoningEffort(%q, %v) = %q, want no effort", model, thinking, effort)
			}
			if supportsReasoning(model) {
				t.Errorf("supportsReasoning(%q) = true", model)
			}
		}
	}
}

// The pro models reason, so their output items still have to be replayed with
// encrypted content even though they take no effort.
func TestProModelsStillSupportReasoning(t *testing.T) {
	for _, model := range []string{"gpt-5-pro", "gpt-5.5-pro", "o3-pro"} {
		if !supportsReasoning(model) {
			t.Errorf("supportsReasoning(%q) = false", model)
		}
	}
}

func TestAutoThinkingGetsNoEffort(t *testing.T) {
	models := []string{"gpt-5", "gpt-5.6-sol", "gpt-5.4-mini", "gpt-5-nano", "o4-mini", "gpt-5.3-codex"}

	for _, model := range models {
		if effort := reasoningEffort(model, llm.AutoThinking); effort != "" {
			t.Errorf("reasoningEffort(%q, AutoThinking) = %q, want no effort", model, effort)
		}
	}
}

func TestEffortMapping(t *testing.T) {
	cases := []struct {
		model    string
		thinking llm.Thinking
		want     string
	}{
		{"gpt-5.6-sol", llm.NoThinking, "none"},
		{"gpt-5.6-sol", llm.MinimalThinking, "none"},
		{"gpt-5.6-sol", llm.LowThinking, "low"},
		{"gpt-5.6-sol", llm.MediumThinking, "medium"},
		{"gpt-5.6-sol", llm.HighThinking, "high"},
		{"gpt-5.4-mini", llm.NoThinking, "none"},
		{"gpt-5-nano", llm.NoThinking, "minimal"},
		{"gpt-5-nano", llm.MinimalThinking, "minimal"},
		{"o4-mini", llm.NoThinking, "low"},
		{"o4-mini", llm.HighThinking, "high"},
		{"gpt-5.3-codex", llm.MinimalThinking, "low"},
	}

	for _, c := range cases {
		if effort := reasoningEffort(c.model, c.thinking); effort != c.want {
			t.Errorf("reasoningEffort(%q, %v) = %q, want %q", c.model, c.thinking, effort, c.want)
		}
	}
}
