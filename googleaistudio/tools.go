package googleaistudio

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/log"
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

// Gemini part types for function calling

type FunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type FunctionResponse struct {
	Name     string                 `json:"name"`
	Response FunctionResponseOutput `json:"response"`
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

// resolveToolCalls executes the function calls from the model's response and
// returns a slice of FunctionResponse parts to send back. It matches each
// functionCall by name against the provided tools and calls the Resolver.
func resolveToolCalls(calls []Part, tools []llm.Tool) ([]FunctionResponse, error) {
	responses := make([]FunctionResponse, 0, len(calls))

	for _, call := range calls {
		functionCall := call.FunctionCall
		if functionCall == nil {
			continue
		}

		log.Info("llm tool call " + functionCall.Name)

		var matched *llm.Tool
		for i := range tools {
			if tools[i].Function.Name == functionCall.Name {
				matched = &tools[i]
				break
			}
		}

		if matched == nil {
			responses = append(responses, FunctionResponse{
				Name: functionCall.Name,
				Response: FunctionResponseOutput{
					Output: "error: tool not found: " + functionCall.Name,
				},
			})
			continue
		}

		result, err := matched.Resolver(functionCall.Args)
		if err != nil {
			responses = append(responses, FunctionResponse{
				Name: functionCall.Name,
				Response: FunctionResponseOutput{
					Output: "error: " + err.Error(),
				},
			})
			continue
		}

		resultJson, err := json.Marshal(result)
		if err != nil {
			responses = append(responses, FunctionResponse{
				Name: functionCall.Name,
				Response: FunctionResponseOutput{
					Output: "error: " + err.Error(),
				},
			})
			continue
		}

		responses = append(responses, FunctionResponse{
			Name: functionCall.Name,
			Response: FunctionResponseOutput{
				Output: string(resultJson),
			},
		})
	}

	return responses, nil
}

// convertMessages transforms the common llm.Message slice into Gemini Content
// objects, handling all role types including tool-related messages.
func convertMessages(messages []llm.Message) (contents []Content, systemParts []Part, err error) {
	for _, message := range messages {
		switch message.Role {
		case "system":
			systemParts = append(systemParts, Part{Text: message.Content})

		case "user":
			contents = append(contents, Content{
				Role:  "user",
				Parts: []Part{{Text: message.Content}},
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
			}

			contents = append(contents, Content{
				Role:  "model",
				Parts: parts,
			})
		case "tool":
			part := Part{FunctionResponse: &FunctionResponse{
				Name:     message.ToolCallId,
				Response: FunctionResponseOutput{Output: message.Content},
			}}
			if len(contents) > 0 {
				lastContent := contents[len(contents)-1]
				if lastContent.Role == "user" && len(lastContent.Parts) > 0 {
					lastPart := lastContent.Parts[len(lastContent.Parts)-1]
					if lastPart.FunctionResponse != nil {
						lastContent.Parts = append(lastContent.Parts, part)
						contents[len(contents)-1] = lastContent

						continue
					}
				}
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
