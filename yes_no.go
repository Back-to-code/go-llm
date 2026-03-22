package llm

import "strings"

func YesNo(resp Response, err error) (bool, error) {
	if err != nil {
		return false, err
	}

	yesWords := []string{
		"yes",
		"true",
		"ja",
	}

	value := strings.ToLower(strings.Split(resp.Value, "\n")[0])

	for _, yesWord := range yesWords {
		if strings.Contains(value, yesWord) {
			return true, nil
		}
	}

	return false, nil
}
