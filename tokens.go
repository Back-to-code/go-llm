package llm

import "strings"

func Tokens(response string) []string {
	response = strings.NewReplacer(
		"\n", " ",
		"\t", " ",
		"\r", " ",
	).Replace(response)
	words := strings.Split(response, " ")

	// Remove empty tokens
	for i := len(words) - 1; i >= 0; i-- {
		if words[i] == "" {
			words = append(words[:i], words[i+1:]...)
		}
	}

	return words
}
