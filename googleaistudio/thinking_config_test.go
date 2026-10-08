package googleaistudio

import (
	"fmt"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

func TestGetThinkingConfig(t *testing.T) {
	cases := []struct {
		model    string
		thinking llm.Thinking
		want     string // "level:X", "budget:N" or "none"
	}{
		{model: "gemini-3.5-flash-lite", thinking: llm.AutoThinking, want: "none"},

		{model: "gemini-3.1-pro-preview", thinking: llm.NoThinking, want: "level:LOW"},
		{model: "gemini-3.1-pro-preview", thinking: llm.MinimalThinking, want: "level:LOW"},
		{model: "gemini-3.1-pro-preview", thinking: llm.MediumThinking, want: "level:MEDIUM"},
		{model: "gemini-3.1-pro-preview-customtools", thinking: llm.HighThinking, want: "level:HIGH"},
		{model: "gemini-3-pro-preview", thinking: llm.MediumThinking, want: "level:HIGH"},
		{model: "gemini-3-pro-preview", thinking: llm.NoThinking, want: "level:LOW"},
		{model: "gemini-3.8-flash", thinking: llm.NoThinking, want: "level:LOW"},
		{model: "gemini-3.7-flash", thinking: llm.MinimalThinking, want: "level:LOW"},
		{model: "gemini-3.5-flash", thinking: llm.NoThinking, want: "level:MINIMAL"},
		{model: "gemini-3-flash-preview", thinking: llm.MinimalThinking, want: "level:MINIMAL"},
		{model: "gemini-3.5-flash-lite", thinking: llm.NoThinking, want: "level:MINIMAL"},
		{model: "gemini-3.1-flash-lite", thinking: llm.LowThinking, want: "level:LOW"},
		{model: "models/gemini-3.8-flash", thinking: llm.HighThinking, want: "level:HIGH"},

		{model: "gemini-2.5-flash-lite", thinking: llm.NoThinking, want: "budget:0"},
		{model: "gemini-2.5-flash-lite", thinking: llm.MinimalThinking, want: "budget:512"},
		{model: "gemini-2.5-flash", thinking: llm.HighThinking, want: "budget:24576"},
		{model: "gemini-2.5-pro", thinking: llm.NoThinking, want: "budget:128"},
		{model: "gemini-2.5-pro", thinking: llm.HighThinking, want: "budget:32768"},

		{model: "gemini-2.0-flash", thinking: llm.HighThinking, want: "none"},
		{model: "gemini-3.1-flash-lite-image", thinking: llm.LowThinking, want: "none"},
		{model: "gemini-3.8-flash-tts", thinking: llm.LowThinking, want: "none"},
		{model: "gemini-flash-latest", thinking: llm.LowThinking, want: "none"},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s/%d", tc.model, tc.thinking), func(t *testing.T) {
			config := getThinkingConfig(tc.model, tc.thinking)

			got := "none"
			switch {
			case config == nil:
			case config.ThinkingLevel != "" && config.ThinkingBudget != nil:
				got = "both"
			case config.ThinkingBudget != nil:
				got = fmt.Sprintf("budget:%d", *config.ThinkingBudget)
			default:
				got = "level:" + config.ThinkingLevel
			}
			if got != tc.want {
				t.Errorf("getThinkingConfig(%q, %d) = %s, want %s", tc.model, tc.thinking, got, tc.want)
			}
		})
	}
}
