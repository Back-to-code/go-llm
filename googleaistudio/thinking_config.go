package googleaistudio

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/Back-to-code/go-llm"
)

type ThinkingConfig struct {
	ThinkingLevel string `json:"thinkingLevel,omitempty"`
	// A pointer because 0 is the budget that turns thinking off.
	ThinkingBudget *int `json:"thinkingBudget,omitempty"`
}

var modelPattern = regexp.MustCompile(`^(?:models/)?gemini-(\d+)(?:\.(\d+))?-(pro|flash-lite|flash)\b`)

// Image, speech and live variants share a family name but accept other levels
// than its text models, if any.
var nonTextVariants = []string{"-image", "-tts", "-audio", "-live"}

type geminiModel struct {
	version int // 305 for gemini-3.5
	variant string
}

func parseModel(model string) (geminiModel, bool) {
	for _, variant := range nonTextVariants {
		if strings.Contains(model, variant) {
			return geminiModel{}, false
		}
	}

	match := modelPattern.FindStringSubmatch(model)
	if match == nil {
		return geminiModel{}, false
	}

	major, _ := strconv.Atoi(match[1])
	minor, _ := strconv.Atoi(match[2])
	return geminiModel{version: major*100 + minor, variant: match[3]}, true
}

// getThinkingConfig returns nil, leaving the model on its default, for
// AutoThinking and for models it does not know to think.
func getThinkingConfig(model string, thinking llm.Thinking) *ThinkingConfig {
	if thinking == llm.AutoThinking {
		return nil
	}

	parsed, ok := parseModel(model)
	switch {
	case !ok, parsed.version < 205:
		return nil
	case parsed.version < 300:
		// Gemini 2.5 errors on thinkingLevel.
		return &ThinkingConfig{ThinkingBudget: thinkingBudget(parsed.variant, thinking)}
	}

	return &ThinkingConfig{ThinkingLevel: thinkingLevel(parsed, thinking)}
}

func thinkingBudget(variant string, thinking llm.Thinking) *int {
	budget := 0
	switch thinking {
	case llm.MinimalThinking:
		budget = 512
	case llm.LowThinking:
		budget = 1_024
	case llm.MediumThinking:
		budget = 8_192
	case llm.HighThinking:
		budget = 24_576
	}

	if variant == "pro" {
		// 2.5 Pro cannot turn thinking off and takes 128 to 32768.
		switch thinking {
		case llm.NoThinking, llm.MinimalThinking:
			budget = 128
		case llm.HighThinking:
			budget = 32_768
		}
	}

	return &budget
}

// thinkingLevel maps onto the nearest level the model accepts. No Gemini 3
// model can turn thinking off, so NoThinking gets the lowest level.
func thinkingLevel(model geminiModel, thinking llm.Thinking) string {
	switch thinking {
	case llm.LowThinking:
		return "LOW"
	case llm.MediumThinking:
		if model.variant == "pro" && model.version < 301 {
			// 3.0 Pro only accepts LOW and HIGH.
			return "HIGH"
		}
		return "MEDIUM"
	case llm.HighThinking:
		return "HIGH"
	}

	// Pro and Flash from 3.7 on reject MINIMAL with an error.
	if model.variant == "pro" || (model.variant == "flash" && model.version >= 307) {
		return "LOW"
	}
	return "MINIMAL"
}
