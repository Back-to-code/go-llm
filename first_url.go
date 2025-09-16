package llm

import "strings"

func FirstUrl(response string) (string, bool) {
	for _, token := range Tokens(response) {
		if strings.HasPrefix(token, "https://") || strings.HasPrefix(token, "http://") {
			token = strings.TrimRight(token, " .,!?")
			return token, true
		}
	}

	return "", false
}
