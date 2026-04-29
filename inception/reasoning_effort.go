package inception

import "github.com/Back-to-code/go-llm"

func reasoningEffort(thinking llm.Thinking) string {
	switch thinking {
	case llm.NoThinking, llm.MinimalThinking:
		return "instant"
	case llm.LowThinking:
		return "low"
	case llm.MediumThinking:
		return "medium"
	case llm.HighThinking:
		return "high"
	}
	return ""
}
