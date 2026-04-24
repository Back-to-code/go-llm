package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

// Compile-time assertions: Model and FallbackModel satisfy Prompter.
var (
	_ llm.Prompter = (*llm.Model)(nil)
	_ llm.Prompter = (*llm.FallbackModel)(nil)
)

type stubPrompter struct {
	promptFn func(messages []llm.Message, options llm.Options) (llm.Response, error)
	streamFn func(messages []llm.Message, options llm.Options) (chan string, error)
	calls    atomic.Int32
}

func (s *stubPrompter) Prompt(messages []llm.Message, options llm.Options) (llm.Response, error) {
	s.calls.Add(1)
	return s.promptFn(messages, options)
}

func (s *stubPrompter) PromptSingle(message string, options llm.Options) (llm.Response, error) {
	return s.Prompt([]llm.Message{llm.User(message)}, options)
}

func (s *stubPrompter) Stream(messages []llm.Message, options llm.Options) (chan string, error) {
	s.calls.Add(1)
	return s.streamFn(messages, options)
}

func (s *stubPrompter) ModelName() string {
	return "stub"
}

func okPrompter(value string) *stubPrompter {
	return &stubPrompter{
		promptFn: func(messages []llm.Message, _ llm.Options) (llm.Response, error) {
			return llm.Response{
				Value:        value,
				Conversation: append(messages, llm.Message{Role: "assistant", Content: value}),
			}, nil
		},
	}
}

func errPrompter(err error) *stubPrompter {
	return &stubPrompter{
		promptFn: func([]llm.Message, llm.Options) (llm.Response, error) {
			return llm.Response{}, err
		},
	}
}

func TestFallbackModel_FirstSuccess(t *testing.T) {
	p1 := okPrompter("from-gemini")
	p2 := okPrompter("from-gpt")

	fb := llm.NewFallbackModel(p1, p2)

	resp, err := fb.Prompt([]llm.Message{llm.User("hello")}, llm.Options{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.Value != "from-gemini" {
		t.Errorf("expected value from gemini, got %q", resp.Value)
	}
	if p1.calls.Load() != 1 {
		t.Errorf("expected gemini called once, got %d", p1.calls.Load())
	}
	if p2.calls.Load() != 0 {
		t.Errorf("expected gpt not called, got %d", p2.calls.Load())
	}
}

func TestFallbackModel_FallsBackOnError(t *testing.T) {
	p1 := errPrompter(errors.New("gemini down"))
	p2 := okPrompter("from-gpt")

	fb := llm.NewFallbackModel(p1, p2)

	resp, err := fb.Prompt([]llm.Message{llm.User("hello")}, llm.Options{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.Value != "from-gpt" {
		t.Errorf("expected value from gpt, got %q", resp.Value)
	}
	if p1.calls.Load() != 1 || p2.calls.Load() != 1 {
		t.Errorf("expected each called once, got p1=%d p2=%d", p1.calls.Load(), p2.calls.Load())
	}
}

func TestFallbackModel_AllFail(t *testing.T) {
	sentinel := errors.New("final failure")
	p1 := errPrompter(errors.New("p1 down"))
	p2 := errPrompter(errors.New("p2 down"))
	p3 := errPrompter(sentinel)

	fb := llm.NewFallbackModel(p1, p2, p3)

	_, err := fb.Prompt([]llm.Message{llm.User("hello")}, llm.Options{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("expected wrapped last error, got %v", err)
	}
	if !strings.Contains(err.Error(), "all 3 fallback models failed") {
		t.Errorf("expected 'all 3 fallback models failed' in error, got %q", err.Error())
	}
}

func TestFallbackModel_Empty(t *testing.T) {
	fb := llm.NewFallbackModel()
	_, err := fb.Prompt([]llm.Message{llm.User("hello")}, llm.Options{})
	if err == nil {
		t.Fatal("expected error for empty FallbackModel, got nil")
	}
}

func TestFallbackModel_CtxCancelled(t *testing.T) {
	p1 := errPrompter(errors.New("p1 down"))
	p2 := okPrompter("ok")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fb := llm.NewFallbackModel(p1, p2)

	_, err := fb.Prompt([]llm.Message{llm.User("hello")}, llm.Options{Ctx: ctx})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if p1.calls.Load() != 0 || p2.calls.Load() != 0 {
		t.Errorf("expected no calls on cancelled ctx, got p1=%d p2=%d",
			p1.calls.Load(), p2.calls.Load())
	}
}

func TestFallbackModel_PromptSingle(t *testing.T) {
	p1 := okPrompter("ok")

	fb := llm.NewFallbackModel(p1)

	resp, err := fb.PromptSingle("hello", llm.Options{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.Value != "ok" {
		t.Errorf("expected 'ok', got %q", resp.Value)
	}
}

func TestFallbackModel_AcceptsNestedFallbackModel(t *testing.T) {
	p1 := errPrompter(errors.New("p1 down"))
	p2 := okPrompter("from-inner-p2")
	p3 := okPrompter("from-outer-p3")

	inner := llm.NewFallbackModel(p1, p2)
	outer := llm.NewFallbackModel(inner, p3)

	resp, err := outer.Prompt([]llm.Message{llm.User("hi")}, llm.Options{})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.Value != "from-inner-p2" {
		t.Errorf("expected value from inner p2, got %q", resp.Value)
	}
	if p3.calls.Load() != 0 {
		t.Errorf("expected outer p3 not called, got %d", p3.calls.Load())
	}
}

func TestFallbackModel_AcceptsRealModel(t *testing.T) {
	// Real *llm.Model should satisfy Prompter and work inside FallbackModel.
	// Use stubProvider to avoid hitting network.
	sp := &stubProvider{promptFn: okPromptProvider("from-real-model")}
	realModel := &llm.Model{Name: "gpt-5", Provider: sp}

	fb := llm.NewFallbackModel(realModel)

	resp, err := fb.Prompt([]llm.Message{llm.User("hi")}, llm.Options{NoRetry: true})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if resp.Value != "from-real-model" {
		t.Errorf("expected value from real model, got %q", resp.Value)
	}
}

type stubProvider struct {
	promptFn    func(model string, messages []llm.Message, options llm.Options) (llm.Response, error)
	promptCalls atomic.Int32
}

func (s *stubProvider) Prompt(model string, messages []llm.Message, options llm.Options) (llm.Response, error) {
	s.promptCalls.Add(1)
	return s.promptFn(model, messages, options)
}

func (s *stubProvider) Stream(string, []llm.Message, llm.Options) (chan string, error) {
	return nil, errors.New("not implemented")
}

func (s *stubProvider) SupportsStructuredOutput() bool { return true }
func (s *stubProvider) SupportsStreaming() bool        { return true }
func (s *stubProvider) SupportsTools() bool            { return true }

func okPromptProvider(value string) func(string, []llm.Message, llm.Options) (llm.Response, error) {
	return func(_ string, messages []llm.Message, _ llm.Options) (llm.Response, error) {
		return llm.Response{
			Value:        value,
			Conversation: append(messages, llm.Message{Role: "assistant", Content: value}),
		}, nil
	}
}
