package llm

import "time"

type CallInfo struct {
	Model      string
	Index      int // Position of the model request within its Prompt, from 0
	Attempt    int // 1 for the first try, higher for retries
	ToolChoice ToolChoice
	Messages   []Message // Read only, the conversation that is sent
}

type CallResult struct {
	Value     string // Empty on a tool call turn
	ToolCalls []ToolCall
	Usage     TokenUsage
	Duration  time.Duration
	Err       error
	Retrying  bool // Err is set and the request will be sent again
}

type ToolResult struct {
	Call     ToolCall
	Output   string // What the model receives
	Err      error  // ErrToolNotFound, ErrToolBudgetUsedUp or the resolver's error
	Duration time.Duration
}
