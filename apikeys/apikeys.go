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
)

var apiKeys = []string{
	googleAiStudio,
	togetherAi,
	openAi,
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

func AllApiKeysSet() bool {
	for _, key := range apiKeys {
		if strings.TrimSpace(os.Getenv(key)) == "" {
			return false
		}
	}

	return true
}
