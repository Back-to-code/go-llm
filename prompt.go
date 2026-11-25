package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

type Message struct {
	Role       string          `json:"role" validate:"required|llm_role"` // "user", "assistant", "system", "tool"
	Content    string          `json:"content"`
	Images     []Image         `json:"-"`
	ToolCalls  json.RawMessage `json:"-"`
	ToolCallId string          `json:"-"`
}

// ImageDetail trades tokens for resolution. Only the OpenAI provider sends it.
type ImageDetail string

const (
	ImageDetailAuto ImageDetail = ""
	ImageDetailLow  ImageDetail = "low"
	ImageDetailHigh ImageDetail = "high"
)

type Image struct {
	MimeType string
	Data     []byte
	Detail   ImageDetail
}

func checkImages(messages []Message, provider Provider) error {
	for _, msg := range messages {
		if len(msg.Images) == 0 {
			continue
		}
		if msg.Role != "user" {
			return fmt.Errorf("images can only be sent on user messages, not on %s messages", msg.Role)
		}
		if !provider.SupportsImages() {
			return fmt.Errorf("provider %T does not support images", provider)
		}
	}
	return nil
}

func System(content string) Message {
	return Message{Role: "system", Content: content}
}

func User(content string) Message {
	return Message{Role: "user", Content: content}
}

// UserWithImages sends the images after the text content.
func UserWithImages(content string, images ...Image) Message {
	return Message{Role: "user", Content: content, Images: images}
}

// UserLines is a wrapper function around User that makes it easy to send multiple line messages
func UserLines(lines ...string) Message {
	return User(strings.Join(lines, "\n"))
}

func Assistant(content string) Message {
	return Message{Role: "assistant", Content: content}
}
