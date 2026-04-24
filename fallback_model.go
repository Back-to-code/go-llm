package llm

import (
	"errors"
	"fmt"
)

type FallbackModel struct {
	Models []Prompter
}

func NewFallbackModel(models ...Prompter) *FallbackModel {
	return &FallbackModel{Models: models}
}

func (f *FallbackModel) Prompt(messages []Message, options Options) (Response, error) {
	if len(f.Models) == 0 {
		return Response{}, errors.New("FallbackModel has no models")
	}

	for _, msg := range messages {
		if msg.Role == "tool" {
			// Different providers handle tools differently so we can't allow tool messages when using fallback models
			return Response{}, errors.New("FallbackModel does not support tool messages")
		}
	}

	var lastErr error
	for _, model := range f.Models {
		if options.Ctx != nil && options.Ctx.Err() != nil {
			return Response{}, options.Ctx.Err()
		}
		resp, err := model.Prompt(messages, options)
		if err == nil {
			return resp, nil
		}
		lastErr = err
	}
	return Response{}, fmt.Errorf("all %d fallback models failed, last error: %w", len(f.Models), lastErr)
}

func (f *FallbackModel) PromptSingle(message string, options Options) (Response, error) {
	return f.Prompt([]Message{User(message)}, options)
}

func (f *FallbackModel) Stream(messages []Message, options Options) (chan string, error) {
	if len(f.Models) == 0 {
		return nil, errors.New("FallbackModel has no models")
	}

	for _, msg := range messages {
		if msg.Role == "tool" {
			// Different providers handle tools differently so we can't allow tool messages when using fallback models
			return nil, errors.New("FallbackModel does not support tool messages")
		}
	}

	var lastErr error
	for _, model := range f.Models {
		if options.Ctx != nil && options.Ctx.Err() != nil {
			return nil, options.Ctx.Err()
		}
		ch, err := model.Stream(messages, options)
		if err == nil {
			return ch, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("all %d fallback models stream failed, last error: %w", len(f.Models), lastErr)
}
