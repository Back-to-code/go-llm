package openai

import (
	"strings"

	"bitbucket.org/teamscript/go-llm"
)

func reasoningEffort(model string, thinking llm.Thinking) string {
	model = strings.ToLower(model)

	noThinkingOrDeafult := []string{
		// Models that do not support reasoning
		"gpt-1",
		"gpt-2",
		"gpt-3",
		"gpt-4",

		// Models that only have support for one reasoning effort and will use that as default
		"o1-pro",
		"o3-pro",
		"gpt-5-pro",
	}
	for _, modelNamePrefix := range noThinkingOrDeafult {
		if strings.HasPrefix(model, modelNamePrefix) {
			return ""
		}
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
