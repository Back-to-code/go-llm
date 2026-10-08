package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/Back-to-code/go-llm/log"
)

var (
	ErrToolNotFound = errors.New("tool not found")

	// ErrToolBudgetUsedUp answers the tool calls past Options.MaxToolCalls.
	// Prompt returns it too, alongside the conversation and usage so far, when
	// the model keeps calling tools after the budget is used up.
	ErrToolBudgetUsedUp = errors.New("tool call budget used up")
)

// A tool message carries the notice: it stays in the returned conversation,
// where a user message would read as the user's words.
const toolBudgetNotice = "The tool call budget is used up, no more tools will run. Answer with what you have."

// maxRoundsPastBudget is how many times the model's tool calls are refused
// after the budget is used up before Prompt gives up. Tool choice none does
// not stop every provider's models from calling tools.
const maxRoundsPastBudget = 1

func (m *Model) runToolLoop(messages []Message, options Options) (Response, error) {
	conversation := slices.Clone(messages)
	var usage TokenUsage

	used := options.toolCallsUsed
	if used == nil {
		used = new(int)
	}
	budgetUsedUp := func() bool { return *used >= options.MaxToolCalls }
	callerChoseNone := options.ToolChoice == ToolChoiceNone
	roundsPastBudget := 0

	for index := 0; ; index++ {
		if budgetUsedUp() {
			// Tools stay in the request: some APIs reject a tool choice without
			// them, and dropping them breaks the prompt cache.
			options.ToolChoice = ToolChoiceNone
		}

		turn, err := m.call(index, conversation, options)
		if err != nil {
			return Response{}, err
		}
		usage.add(turn.Usage)
		conversation = append(conversation, turn.Message)

		if len(turn.ToolCalls) == 0 {
			return Response{
				Value:        turn.Message.Content,
				Conversation: conversation,
				Usage:        usage,
			}, nil
		}
		if callerChoseNone {
			// Answering these would only invite more of them.
			return Response{}, errors.New("model called tools while tool choice was none")
		}

		pastBudget := budgetUsedUp()
		conversation = append(conversation, answerToolCalls(turn.ToolCalls, used, options)...)

		if pastBudget {
			roundsPastBudget++
			if roundsPastBudget > maxRoundsPastBudget {
				return Response{Conversation: conversation, Usage: usage},
					fmt.Errorf("model kept calling tools: %w", ErrToolBudgetUsedUp)
			}
		}
		options.ToolChoice = ToolChoiceAuto
	}
}

// answerToolCalls returns one tool message per call. Every call of a turn
// needs one, so failures are reported to the model, not returned.
func answerToolCalls(calls []ToolCall, used *int, options Options) []Message {
	answers := make([]Message, len(calls))
	for idx, call := range calls {
		start := time.Now()

		output, err := toolOutput(call, used, options)
		if err != nil {
			output = "error: " + err.Error()
		}
		if idx == len(calls)-1 && *used >= options.MaxToolCalls {
			output += "\n\n" + toolBudgetNotice
		}

		if options.OnToolCall != nil {
			options.OnToolCall(ToolResult{
				Call:     call,
				Output:   output,
				Err:      err,
				Duration: time.Since(start),
			})
		}

		answers[idx] = Message{Role: "tool", Content: output, ToolCallId: call.Id}
	}

	return answers
}

func toolOutput(call ToolCall, used *int, options Options) (string, error) {
	if *used >= options.MaxToolCalls {
		return "", ErrToolBudgetUsedUp
	}
	*used++

	idx := slices.IndexFunc(options.Tools, func(tool Tool) bool { return tool.Function.Name == call.Name })
	if idx == -1 {
		return "", ErrToolNotFound
	}

	log.Info("llm tool call " + call.Name)
	result, err := options.Tools[idx].Resolver(call.Arguments)
	if err != nil {
		return "", err
	}

	var output string
	switch result := result.(type) {
	case string:
		output = result
	case json.RawMessage:
		output = string(result)
	default:
		encoded, err := json.Marshal(result)
		if err != nil {
			return "", err
		}
		output = string(encoded)
	}

	if output == "" {
		// Providers drop or reject an empty tool output.
		return `""`, nil
	}
	return output, nil
}
