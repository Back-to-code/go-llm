package llm

import (
	"encoding/json"
	"strings"
)

type Message struct {
	Role             string          `json:"role" validate:"required|llm_role"` // "user", "assistant", "system", "tool"
	Content          string          `json:"content"`
	ToolCalls        json.RawMessage `json:"-"`
	ToolCallId       string          `json:"-"`
	ThoughtSignature string          `json:"-"`
}

func System(content string) Message {
	return Message{Role: "system", Content: content}
}

func User(content string) Message {
	return Message{Role: "user", Content: content}
}

// UserLines is a wrapper function around User that makes it easy to send multiple line messages
func UserLines(lines ...string) Message {
	return User(strings.Join(lines, "\n"))
}

func Assistant(content string) Message {
	return Message{Role: "assistant", Content: content}
}
