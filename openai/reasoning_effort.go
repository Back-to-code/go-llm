package openai

import (
	"strings"

	"github.com/Back-to-code/go-llm"
)

var nonReasoningModels = []string{
	"gpt-1",
	"gpt-2",
	"gpt-3",
	"gpt-4",
}

// narrowsEffortRange reports whether the model accepts less than the effort
// range of its family: every pro model rejects "none", chat-latest takes only
// "medium". Both are left to the server default rather than mapped.
func narrowsEffortRange(model string) bool {
	return strings.Contains(model, "-pro") || strings.Contains(model, "chat-latest")
}

func hasModelPrefix(model string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}

	return false
}

// supportsReasoning reports whether the model emits reasoning items, which the
// pro models do even though reasoningEffort returns "" for them.
func supportsReasoning(model string) bool {
	return !hasModelPrefix(strings.ToLower(model), nonReasoningModels)
}

func reasoningEffort(model string, thinking llm.Thinking) string {
	model = strings.ToLower(model)

	if thinking == llm.AutoThinking {
		return ""
	}

	if hasModelPrefix(model, nonReasoningModels) || narrowsEffortRange(model) {
		return ""
	}

	if strings.HasPrefix(model, "o") {
		switch thinking {
		case llm.NoThinking, llm.LowThinking, llm.MinimalThinking:
			return "low"
		case llm.MediumThinking:
			return "medium"
		case llm.HighThinking:
			return "high"
		}
	}

	if strings.Contains(model, "codex") {
		switch thinking {
		case llm.NoThinking, llm.MinimalThinking, llm.LowThinking:
			return "low"
		case llm.MediumThinking:
			return "medium"
		case llm.HighThinking:
			return "high"
		}
	}

	if model == "gpt-5" || strings.HasPrefix(model, "gpt-5-mini") || strings.HasPrefix(model, "gpt-5-nano") {
		switch thinking {
		case llm.MinimalThinking, llm.NoThinking:
			return "minimal"
		case llm.LowThinking:
			return "low"
		case llm.MediumThinking:
			return "medium"
		case llm.HighThinking:
			return "high"
		}
	}

	switch thinking {
	case llm.MinimalThinking, llm.NoThinking:
		return "none"
	case llm.LowThinking:
		return "low"
	case llm.MediumThinking:
		return "medium"
	case llm.HighThinking:
		return "high"
	}

	return ""
}
