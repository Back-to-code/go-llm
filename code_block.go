package llm

import "strings"

// ExtractFirstCodeBlock extracts the first code block from markdown content.
// If no code block is found, it returns the original input.
func ExtractFirstCodeBlock(markdown string) string {
	// Split by lines to check each potential code block
	lines := strings.Split(markdown, "\n")

	var blockStart, blockEnd int = -1, -1
	var fenceType string

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Check for start of code block
		if blockStart == -1 {
			if strings.HasPrefix(trimmed, "```") {
				fenceType = "```"
				blockStart = i
			} else if strings.HasPrefix(trimmed, "~~~") {
				fenceType = "~~~"
				blockStart = i
			}
		} else {
			// Look for matching end fence
			if strings.HasPrefix(trimmed, fenceType) {
				blockEnd = i
				break
			}
		}
	}

	// If we found a complete code block, extract its content
	if blockStart != -1 && blockEnd != -1 {
		codeLines := lines[blockStart+1 : blockEnd]
		return strings.Join(codeLines, "\n")
	}

	// No code block found, return original input
	return markdown
}
