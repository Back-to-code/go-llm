package llm

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/Back-to-code/go-llm/cache"
	"github.com/Back-to-code/go-llm/log"
)

var DefaultCacheDuration = time.Hour * 24

const DefaultMaxToolCalls = 500

type ResponseFormat string

var (
	ResponseFormatJsonSchema ResponseFormat = "json_schema" // Requires Options.JsonSchema
	ResponseFormatJsonObject ResponseFormat = "json_object"
)

type JsonSchema struct {
	Name   string          `json:"name"` // Defaults to "response"
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict,omitempty"`
}

type ToolChoice string

const (
	ToolChoiceAuto ToolChoice = "auto" // The default
	// ToolChoiceRequired only applies to the first model call of a Prompt,
	// forcing every call would never let the model answer.
	ToolChoiceRequired ToolChoice = "required"
	ToolChoiceNone     ToolChoice = "none"
)

type Thinking uint8

const (
	// AutoThinking sends no thinking parameter, leaving the level to the
	// provider's own default for the model.
	AutoThinking Thinking = iota
	NoThinking
	MinimalThinking
	LowThinking
	MediumThinking
	HighThinking
)

type Options struct {
	// Generically implemented
	Cache   time.Duration // If <= 0, nothing will be cached
	NoRetry bool

	// MaxToolCalls caps the tool calls of one Prompt, 0 means
	// DefaultMaxToolCalls. Calls past the cap are answered with
	// ErrToolBudgetUsedUp instead of being resolved, after which the model has
	// to answer without tools.
	MaxToolCalls int

	// Called synchronously from Prompt, never by Stream. OnCallEnd fires for
	// failed attempts too.
	OnCallStart func(CallInfo)
	OnCallEnd   func(CallInfo, CallResult)
	OnToolCall  func(ToolResult)

	// Implemented by each providers
	Timeout        time.Duration // The request timeout
	MaxTokens      int
	Ctx            context.Context
	ResponseFormat ResponseFormat
	JsonSchema     JsonSchema
	Tools          []Tool
	ToolChoice     ToolChoice
	Thinking       Thinking

	// Shared by the models of a FallbackModel, so they spend one budget.
	toolCallsUsed *int
}

func (o Options) prepare(isStream bool, provider Provider) (Options, error) {
	if isStream && !provider.SupportsStreaming() {
		return o, fmt.Errorf("provider %T does not support streaming", provider)
	}
	if o.ResponseFormat != "" && !provider.SupportsStructuredOutput() {
		return o, fmt.Errorf("provider %T does not support structured output", provider)
	}
	if len(o.Tools) > 0 && !provider.SupportsTools() {
		return o, fmt.Errorf("provider %T does not support tools", provider)
	}

	hasSchema := len(o.JsonSchema.Schema) > 0
	if o.ResponseFormat == ResponseFormatJsonSchema && !hasSchema {
		return o, errors.New("response format json_schema requires a JsonSchema")
	}
	if o.ResponseFormat != ResponseFormatJsonSchema && hasSchema {
		return o, errors.New("JsonSchema requires response format json_schema")
	}
	if o.JsonSchema.Name == "" && hasSchema {
		o.JsonSchema.Name = "response"
	}

	switch o.ToolChoice {
	case "":
		o.ToolChoice = ToolChoiceAuto
	case ToolChoiceAuto, ToolChoiceNone:
	case ToolChoiceRequired:
		if len(o.Tools) == 0 {
			return o, errors.New("tool choice required needs at least one tool")
		}
		if isStream {
			return o, errors.New("tool choice required cannot stream, Stream does not resolve tool calls")
		}
	default:
		return o, fmt.Errorf("unknown tool choice %q", o.ToolChoice)
	}
	if o.MaxToolCalls < 0 {
		return o, errors.New("MaxToolCalls cannot be negative")
	}
	if o.MaxToolCalls == 0 {
		o.MaxToolCalls = DefaultMaxToolCalls
	}

	if o.Timeout <= 0 {
		o.Timeout = time.Second * 30
	}
	for idx, tool := range o.Tools {
		if tool.Resolver == nil {
			return o, fmt.Errorf("tool %s (#%d) is missing a resolver", tool.Function.Name, idx+1)
		}
		if tool.Type == "" {
			tool.Type = "function"
			o.Tools[idx] = tool
		}
	}

	return o, nil
}

type Turn struct {
	// Message is the assistant message to append to the conversation. On a
	// tool call turn it carries the provider's raw tool calls in ToolCalls.
	Message Message

	// ToolCalls are left for the caller to resolve. Empty means
	// Message.Content is the model's answer.
	ToolCalls []ToolCall

	Usage TokenUsage
}

type Provider interface {
	// Call sends exactly one request, it neither retries nor resolves tools.
	Call(model string, messages []Message, options Options) (Turn, error)
	Stream(model string, messages []Message, options Options) (chan string, error)
	SupportsStructuredOutput() bool
	SupportsStreaming() bool
	SupportsTools() bool
}

type Prompter interface {
	Prompt(messages []Message, options Options) (Response, error)
	PromptSingle(message string, options Options) (Response, error)
	Stream(messages []Message, options Options) (chan string, error)
	ModelName() string
}

type Model struct {
	Name     string
	Provider Provider
}

func (m *Model) ModelName() string {
	if m.Name == "" {
		return "<none>"
	}
	return m.Name
}

func (m *Model) Prompt(messages []Message, options Options) (Response, error) {
	var err error
	options, err = options.prepare(false, m.Provider)
	if err != nil {
		return Response{}, err
	}

	var key string
	if options.Cache > 0 {
		key, err = cacheKey(m.Name, messages, options)
		if err == nil {
			cachedResponse, err := cache.Get(key)
			if err == nil && cachedResponse != "" {
				return Response{
					Value:        cachedResponse,
					Conversation: messages,
				}, nil
			}
		}
	}

	resp, err := m.runToolLoop(messages, options)
	if err != nil {
		return resp, err
	}

	if key != "" {
		cache.Set(key, resp.Value, options.Cache)
	}
	return resp, nil
}

// cacheKey covers every input that shapes the answer.
func cacheKey(model string, messages []Message, options Options) (string, error) {
	// Message leaves its tool fields out of its JSON.
	type keyMessage struct {
		Role       string
		Content    string
		ToolCalls  string
		ToolCallId string
	}
	keyMessages := make([]keyMessage, len(messages))
	for idx, msg := range messages {
		keyMessages[idx] = keyMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCalls:  string(msg.ToolCalls),
			ToolCallId: msg.ToolCallId,
		}
	}

	contents, err := json.Marshal(struct {
		Messages       []keyMessage
		MaxTokens      int
		ResponseFormat ResponseFormat
		JsonSchema     JsonSchema
		Tools          []Tool
		ToolChoice     ToolChoice
		MaxToolCalls   int
		Thinking       Thinking
	}{
		Messages:       keyMessages,
		MaxTokens:      options.MaxTokens,
		ResponseFormat: options.ResponseFormat,
		JsonSchema:     options.JsonSchema,
		Tools:          options.Tools,
		ToolChoice:     options.ToolChoice,
		MaxToolCalls:   options.MaxToolCalls,
		Thinking:       options.Thinking,
	})
	if err != nil {
		return "", err
	}

	hash := sha1.Sum(contents)
	return model + ":" + hex.EncodeToString(hash[:]), nil
}

// call sends one model request, retrying it on transient errors.
func (m *Model) call(index int, messages []Message, options Options) (Turn, error) {
	retries := 5
	if options.NoRetry {
		retries = 1
	}

	var err error
	for attempt := 1; attempt <= retries; attempt++ {
		if options.Ctx != nil && options.Ctx.Err() != nil {
			return Turn{}, options.Ctx.Err()
		}

		info := CallInfo{
			Model:      m.Name,
			Index:      index,
			Attempt:    attempt,
			ToolChoice: options.ToolChoice,
			Messages:   messages,
		}
		if options.OnCallStart != nil {
			options.OnCallStart(info)
		}

		start := time.Now()
		var turn Turn
		turn, err = m.Provider.Call(m.Name, messages, options)
		duration := time.Since(start)
		cancelled := options.Ctx != nil && options.Ctx.Err() != nil
		retrying := err != nil && !IsPermanent(err) && !cancelled && attempt < retries

		if options.OnCallEnd != nil {
			options.OnCallEnd(info, CallResult{
				Value:     turn.Message.Content,
				ToolCalls: turn.ToolCalls,
				Usage:     turn.Usage,
				Duration:  duration,
				Err:       err,
				Retrying:  retrying,
			})
		}

		if !retrying {
			return turn, err
		}
		if duration < time.Second {
			time.Sleep(time.Millisecond * 100 * time.Duration(attempt))
		}
	}

	return Turn{}, err
}

// PromptSingle is a wrapper around prompt but only prompt 1 user message
func (m *Model) PromptSingle(message string, options Options) (Response, error) {
	return m.Prompt([]Message{User(message)}, options)
}

func (m *Model) Stream(messages []Message, options Options) (chan string, error) {
	var err error
	options, err = options.prepare(true, m.Provider)
	if err != nil {
		return nil, err
	}

	log.Info("Sending prompt to " + m.Name)
	return m.Provider.Stream(m.Name, messages, options)
}
