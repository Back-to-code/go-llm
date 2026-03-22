package llm

// TokenUsage holds token consumption metrics for a Prompt or PromptSingle call.
type TokenUsage struct {
	InputTokens       int
	OutputTokens      int
	CachedInputTokens int
}

// Response is the return type for Prompt and PromptSingle calls.
type Response struct {
	// Value is the final text content returned by the model.
	Value string

	// Conversation contains all messages up to and including the final
	// assistant response. When tool calls are involved this includes the
	// intermediate assistant tool-call messages and tool response messages.
	Conversation []Message

	// Usage holds the accumulated token usage across all API round-trips
	// that occurred during this call (including tool-call loops).
	Usage TokenUsage
}

// String returns the Value field, making it easy to migrate from the old
// (string, error) return type to (Response, error).
func (r Response) String() string {
	return r.Value
}
