package apikey

import (
	"errors"
	"os"
	"strings"
)

const (
	googleAiStudio = "GOOGLE_AI_STUDIO_KEY"
	togetherAi     = "TOGETHER_AI_TOKEN"
	openAi         = "OPENAI_TOKEN"
	inception      = "INCEPTION_API_KEY"
)

var apiKeys = []string{
	googleAiStudio,
	togetherAi,
	openAi,
	inception,
}

func getKeyFn(key string) func() (string, error) {
	return func() (string, error) {
		value := strings.TrimSpace(os.Getenv(key))
		if value == "" {
			return "", errors.New(key + " environment variable not set")
		}

		return value, nil
	}
}

var GoogleAiStudio = getKeyFn(googleAiStudio)
var TogetherAi = getKeyFn(togetherAi)
var OpenAi = getKeyFn(openAi)
var Inception = getKeyFn(inception)

type RequiredApiKeys struct {
	GoogleAiStudio bool
	TogetherAi     bool
	OpenAi         bool
	Inception      bool
}

func AllApiKeysSet(requirements RequiredApiKeys) bool {
	if requirements.GoogleAiStudio && googleAiStudio == "" {
		return false
	}
	if requirements.TogetherAi && togetherAi == "" {
		return false
	}
	if requirements.OpenAi && openAi == "" {
		return false
	}
	if requirements.Inception && inception == "" {
		return false
	}

	return true
}
