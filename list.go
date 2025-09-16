package llm

import (
	"regexp"
	"strings"
	"unicode"
)

// ListStyle represents different types of list markers
type ListStyle uint8

const (
	invalidListStyle ListStyle = iota // Invalid or no list style
	numbersListStyle                  // e.g., "1." or "1:"
	dashListStyle                     // e.g., "- "
	starListStyle                     // e.g., "* "
	plusListStyle                     // e.g., "+ "
)

var (
	// Precompiled regex patterns
	numberPattern = regexp.MustCompile(`^\s*\d+\s*[:.]\s*`)
	dashPattern   = regexp.MustCompile(`^\s*-\s+`)
	starPattern   = regexp.MustCompile(`^\s*\*\s+`)
	plusPattern   = regexp.MustCompile(`^\s*\+\s+`)
)

// ListFromResponse processes a string message and returns a list of suggestions
func ListFromResponse(resp string) []string {
	lines := strings.Split(resp, "\n")
	listKind := detectListStyleKind(lines)

	seen := make(map[string]bool)
	var result []string

	withinList := false
	notWithinListCount := 0

	for _, line := range lines {
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		content := isListItem(line, listKind)
		if content != "" {
			// Check for Chinese characters
			hasHan := false
			for _, r := range content {
				if unicode.Is(unicode.Han, r) {
					hasHan = true
					break
				}
			}
			if hasHan {
				continue
			}

			withinList = true
			notWithinListCount = 0

			// Handle duplicate detection
			seenKey := strings.ToLower(content)
			if seen[seenKey] {
				continue
			}

			seen[seenKey] = true
			result = append(result, content)
		} else if withinList {
			notWithinListCount++
			if notWithinListCount > 2 {
				break
			}
		}
	}

	return result
}

// detectListStyleKind determines the predominant list style in the text
func detectListStyleKind(lines []string) ListStyle {
	styleCounter := make(map[ListStyle]int)

	for _, line := range lines {
		kind := lineListKind(line)
		if kind != invalidListStyle {
			styleCounter[kind]++
		}
	}

	selectedKind := numbersListStyle
	selectedCount := 0

	for kind, count := range styleCounter {
		if count > selectedCount {
			selectedKind = kind
			selectedCount = count
		}
	}

	return selectedKind
}

// lineListKind detects the type of list style for a single line
func lineListKind(line string) ListStyle {
	if numberPattern.MatchString(line) {
		return numbersListStyle
	}

	if dashPattern.MatchString(line) {
		return dashListStyle
	}

	if starPattern.MatchString(line) {
		return starListStyle
	}

	if plusPattern.MatchString(line) {
		return plusListStyle
	}

	return invalidListStyle
}

// isListItem checks if the line is a list item and returns its content
func isListItem(line string, kind ListStyle) string {
	switch kind {
	case numbersListStyle:
		if numberPattern.MatchString(line) {
			return numberPattern.ReplaceAllString(line, "")
		}
	case dashListStyle:
		if dashPattern.MatchString(line) {
			return dashPattern.ReplaceAllString(line, "")
		}
	case starListStyle:
		if starPattern.MatchString(line) {
			return starPattern.ReplaceAllString(line, "")
		}
	case plusListStyle:
		if plusPattern.MatchString(line) {
			return plusPattern.ReplaceAllString(line, "")
		}
	}

	return ""
}
