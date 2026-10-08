package googleaistudio

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"github.com/Back-to-code/go-llm"
)

// Gemini tool definition types

type GeminiFunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type GeminiTool struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"functionDeclarations"`
}

type ToolConfig struct {
	FunctionCallingConfig FunctionCallingConfig `json:"functionCallingConfig"`
}

type FunctionCallingConfig struct {
	Mode string `json:"mode"`
}

// toolConfig returns nil for auto, which is Gemini's default.
func toolConfig(choice llm.ToolChoice) *ToolConfig {
	var mode string
	switch choice {
	case llm.ToolChoiceRequired:
		mode = "ANY"
	case llm.ToolChoiceNone:
		mode = "NONE"
	default:
		return nil
	}

	return &ToolConfig{FunctionCallingConfig: FunctionCallingConfig{Mode: mode}}
}

// Gemini part types for function calling

type FunctionCall struct {
	Id   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type FunctionResponse struct {
	Id       string                 `json:"id,omitempty"`
	Name     string                 `json:"name"`
	Response FunctionResponseOutput `json:"response"`
}

// callId identifies the call at position among the function calls of one
// model turn. Gemini leaves the id empty on some models, and parallel calls
// can share a name.
func callId(call *FunctionCall, position int) string {
	if call.Id != "" {
		return call.Id
	}
	return call.Name + "#" + strconv.Itoa(position)
}

type FunctionResponseOutput struct {
	Output string `json:"output,omitempty"`
}

// convertTools transforms the common llm.Tool definitions into Gemini's
// functionDeclarations format.
func convertTools(tools []llm.Tool) []GeminiTool {
	if len(tools) == 0 {
		return nil
	}

	declarations := make([]GeminiFunctionDeclaration, len(tools))
	for i, tool := range tools {
		declarations[i] = GeminiFunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		}
	}

	return []GeminiTool{{FunctionDeclarations: declarations}}
}

// convertMessages transforms the common llm.Message slice into Gemini Content
// objects, handling all role types including tool-related messages.
func convertMessages(messages []llm.Message) (contents []Content, systemParts []Part, err error) {
	calls := map[string]*FunctionCall{}
	for _, message := range messages {
		switch message.Role {
		case "system":
			systemParts = append(systemParts, Part{Text: message.Content})

		case "user":
			part := Part{Text: message.Content}
			// Gemini rejects a second user turn right after the function responses.
			if endsWithFunctionResponse(contents) {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, part)
				continue
			}

			contents = append(contents, Content{
				Role:  "user",
				Parts: []Part{part},
			})

		case "assistant":
			parts := []Part{{Text: message.Content}}
			if len(message.ToolCalls) > 0 {
				// This is a model turn that contained function calls.
				// Deserialize the stored tool calls back into Part objects.
				parts = []Part{}
				if err := json.Unmarshal(message.ToolCalls, &parts); err != nil {
					return nil, nil, fmt.Errorf("unmarshaling tool calls: %w", err)
				}

				position := 0
				for _, part := range parts {
					if part.FunctionCall != nil {
						calls[callId(part.FunctionCall, position)] = part.FunctionCall
						position++
					}
				}
			}

			contents = append(contents, Content{
				Role:  "model",
				Parts: parts,
			})
		case "tool":
			// A hand written tool message may carry the function name as its id.
			response := &FunctionResponse{
				Name:     message.ToolCallId,
				Response: FunctionResponseOutput{Output: message.Content},
			}
			if call, ok := calls[message.ToolCallId]; ok {
				response.Id = call.Id
				response.Name = call.Name
			}

			part := Part{FunctionResponse: response}
			if endsWithFunctionResponse(contents) {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, part)
				continue
			}

			contents = append(contents, Content{
				Role:  "user",
				Parts: []Part{part},
			})
		default:
			return nil, nil, errors.New(message.Role + " role currently not supported for gemini")
		}
	}

	return contents, systemParts, nil
}

func endsWithFunctionResponse(contents []Content) bool {
	if len(contents) == 0 {
		return false
	}

	last := contents[len(contents)-1]
	return last.Role == "user" && len(last.Parts) > 0 && last.Parts[len(last.Parts)-1].FunctionResponse != nil
}
