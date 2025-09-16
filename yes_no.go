package llm

import "strings"

func YesNo(resp string, err error) (bool, error) {
	if err != nil {
		return false, err
	}

	yesWords := []string{
		"yes",
		"true",
		"ja",
	}

	resp = strings.ToLower(strings.Split(resp, "\n")[0])

	for _, yesWord := range yesWords {
		if strings.Contains(resp, yesWord) {
			return true, nil
		}
	}

	return false, nil
}
