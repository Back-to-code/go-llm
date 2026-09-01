package llm

import "encoding/json"

type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`

	// Ignored by the openai provider: strict mode reads additionalProperties
	// from inside Parameters.
	AdditionalProperties bool `json:"additionalProperties"`
}

// Tool defines a tool that can be used by the LLM
// Required fields are Resolver and Function
// The contents of "Function" can be created via the openai website
type Tool struct {
	Resolver func(json.RawMessage) (any, error) `json:"-"`

	Type     string      `json:"type"` // Automatically set to "function" if empty
	Function FunctionDef `json:"function"`
	Strict   bool        `json:"strict"`
}
