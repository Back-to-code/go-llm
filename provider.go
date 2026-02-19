package llm

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"bitbucket.org/teamscript/go-llm/cache"
	"bitbucket.org/teamscript/go-llm/log"
)

var DefaultCacheDuration = time.Hour * 24

type ResponseFormat string

var (
	// ResponseFormatJsonSchema ResponseFormat = "json_schema" // (Unsupported)
	ResponseFormatJsonObject ResponseFormat = "json_object"
)

type Thinking uint8

const (
	NoThinking Thinking = iota
	MinimalThinking
	LowThinking
	MediumThinking
	HighThinking
)

type Options struct {
	// Generically implemented
	Cache   time.Duration // If <= 0, nothing will be cached
	NoRetry bool

	// Implemented by each providers
	Timeout        time.Duration // The request timeout
	MaxTokens      int
	Ctx            context.Context
	ResponseFormat ResponseFormat
	Tools          []Tool
	Thinking       Thinking
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

	if o.Timeout <= 0 {
		o.Timeout = time.Second * 30
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = 4096
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

type Provider interface {
	Prompt(model string, messages []Message, options Options) (string, error)
	Stream(model string, messages []Message, options Options) (chan string, error)
	SupportsStructuredOutput() bool
	SupportsStreaming() bool
	SupportsTools() bool
}

type Model struct {
	Name     string
	Provider Provider
}

func (m *Model) Prompt(messages []Message, options Options) (string, error) {
	var err error
	options, err = options.prepare(false, m.Provider)
	if err != nil {
		return "", err
	}

	retries := 5
	if options.NoRetry {
		retries = 1
	}

	var cacheKey string
	if options.Cache > 0 {
		cacheKeyHashContents, err := json.Marshal(messages)
		if err == nil {
			cacheKeyHash := sha1.New()
			cacheKeyHash.Write(cacheKeyHashContents)
			cacheKey = m.Name + ":" + hex.EncodeToString(cacheKeyHash.Sum(nil))
			cachedResponse, err := cache.Get(cacheKey)
			if err == nil && cachedResponse != "" {
				return cachedResponse, nil
			}
		}
	}

	var resp string
	for i := 0; i < retries; i++ {
		if options.Ctx != nil && options.Ctx.Err() != nil {
			return "", options.Ctx.Err()
		}

		resp, err = m.Provider.Prompt(m.Name, messages, options)
		if err != nil {
			continue
		}

		if cacheKey != "" {
			cache.Set(cacheKey, resp, options.Cache)
		}
		return resp, err
	}

	return resp, err
}

// PromptSingle is a wrapper around prompt but only prompt 1 user message
func (m *Model) PromptSingle(message string, options Options) (string, error) {
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
