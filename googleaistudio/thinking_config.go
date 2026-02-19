package googleaistudio

import (
	"strings"

	"github.com/Back-to-code/go-llm"
)

type ThinkingConfig struct {
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
	ThinkingBudget int    `json:"thinkingBudget,omitempty"`
}

func thinkingLvl(lvl string) *ThinkingConfig {
	return &ThinkingConfig{
		ThinkingLevel: lvl,
	}
}

func thinkingBudget(budget int) *ThinkingConfig {
	return &ThinkingConfig{
		ThinkingBudget: budget,
	}
}

var thinkingMappings = []struct {
	ModelPrefix string
	lvls        map[llm.Thinking]*ThinkingConfig
}{
	{
		ModelPrefix: "gemini-3-pro",
		lvls: map[llm.Thinking]*ThinkingConfig{
			llm.NoThinking:      thinkingLvl("LOW"),
			llm.MinimalThinking: thinkingLvl("LOW"),
			llm.LowThinking:     thinkingLvl("LOW"),
			llm.MediumThinking:  thinkingLvl("HIGH"),
			llm.HighThinking:    thinkingLvl("HIGH"),
		},
	},
	{
		ModelPrefix: "gemini-3-flash",
		lvls: map[llm.Thinking]*ThinkingConfig{
			llm.NoThinking:      thinkingLvl("MINIMAL"),
			llm.MinimalThinking: thinkingLvl("MINIMAL"),
			llm.LowThinking:     thinkingLvl("LOW"),
			llm.MediumThinking:  thinkingLvl("MEDIUM"),
			llm.HighThinking:    thinkingLvl("HIGH"),
		},
	},
	{
		ModelPrefix: "gemini-2.5",
		lvls: map[llm.Thinking]*ThinkingConfig{
			llm.NoThinking:      thinkingBudget(-1),
			llm.MinimalThinking: thinkingBudget(512),
			llm.LowThinking:     thinkingBudget(1_024),
			llm.MediumThinking:  thinkingBudget(8_192),
			llm.HighThinking:    thinkingBudget(32_576),
		},
	},
}

func getThinkingConfig(model string, thinking llm.Thinking) *ThinkingConfig {
	for _, mapping := range thinkingMappings {
		if strings.HasPrefix(model, mapping.ModelPrefix) {
			return mapping.lvls[thinking]
		}
	}

	return nil
}
