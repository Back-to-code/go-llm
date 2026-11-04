package llm_test

import (
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

type stubDecider struct {
	stubProvider
	decideFn func(model string, input string, questions []llm.Question) (llm.Decisions, error)
	decides  atomic.Int32
}

func (s *stubDecider) Decide(model string, input string, questions []llm.Question, _ llm.DecisionOptions) (llm.Decisions, error) {
	s.decides.Add(1)
	return s.decideFn(model, input, questions)
}

var pickQuestion = []llm.Question{{
	Name:         "pick",
	Type:         llm.QuestionChoice,
	Instructions: "Which fits?",
	Choices:      []llm.Choice{{Value: "a"}, {Value: "b"}},
}}

func TestDecideRetriesATransientErrorAndReportsEveryAttempt(t *testing.T) {
	provider := &stubDecider{}
	provider.decideFn = func(model string, input string, questions []llm.Question) (llm.Decisions, error) {
		if provider.decides.Load() == 1 {
			return llm.Decisions{}, &llm.Err{StatusCode: 500, Body: "boom"}
		}
		return llm.Decisions{
			Answers: []llm.Answer{{Name: "pick", Type: llm.QuestionChoice, Choice: "b", Probabilities: []llm.Probability{
				{Value: "a", Probability: 0.2}, {Value: "b", Probability: 0.8},
			}}},
			Usage: llm.TokenUsage{InputTokens: 12},
		}, nil
	}
	model := &llm.Model{Name: "stub", Provider: provider}

	var started []llm.CallInfo
	var ended []llm.CallResult
	decisions, err := model.Decide("the input", pickQuestion, llm.DecisionOptions{
		OnCallStart: func(info llm.CallInfo) { started = append(started, info) },
		OnCallEnd:   func(_ llm.CallInfo, result llm.CallResult) { ended = append(ended, result) },
	})
	if err != nil {
		t.Fatalf("Decide() error = %v", err)
	}

	answer, ok := decisions.Answer("pick")
	if !ok || answer.Choice != "b" || answer.ProbabilityOf("a") != 0.2 || answer.ProbabilityOf("c") != 0 {
		t.Fatalf("answer = %+v, want choice b with 0.2 for a", answer)
	}
	if provider.decides.Load() != 2 {
		t.Errorf("decide calls = %d, want 2", provider.decides.Load())
	}
	if len(started) != 2 || started[1].Attempt != 2 || started[0].Messages[0].Content != "the input" {
		t.Errorf("started = %+v, want two attempts with the input as message", started)
	}
	if len(ended) != 2 || !ended[0].Retrying || ended[1].Usage.InputTokens != 12 {
		t.Errorf("ended = %+v, want a retried attempt and one with the usage", ended)
	}
}

func TestDecideDoesNotRetryAPermanentError(t *testing.T) {
	provider := &stubDecider{decideFn: func(string, string, []llm.Question) (llm.Decisions, error) {
		return llm.Decisions{}, &llm.Err{StatusCode: 400, Body: "array too short"}
	}}
	model := &llm.Model{Name: "stub", Provider: provider}

	_, err := model.Decide("input", pickQuestion, llm.DecisionOptions{})
	if err == nil || !strings.Contains(err.Error(), "array too short") {
		t.Fatalf("Decide() error = %v, want the bad request", err)
	}
	if provider.decides.Load() != 1 {
		t.Errorf("decide calls = %d, want 1", provider.decides.Load())
	}
}

func TestDecideRefusesAProviderWithoutDecisionsAndNoQuestions(t *testing.T) {
	model := &llm.Model{Name: "stub", Provider: &stubProvider{callFn: okCallProvider("unused")}}
	_, err := model.Decide("input", pickQuestion, llm.DecisionOptions{})
	if !llm.IsPermanent(err) || !strings.Contains(err.Error(), "does not support decisions") {
		t.Fatalf("Decide() error = %v, want a permanent unsupported error", err)
	}

	decider := &stubDecider{decideFn: func(string, string, []llm.Question) (llm.Decisions, error) {
		return llm.Decisions{}, errors.New("must not be called")
	}}
	_, err = (&llm.Model{Name: "stub", Provider: decider}).Decide("input", nil, llm.DecisionOptions{})
	if !llm.IsPermanent(err) || decider.decides.Load() != 0 {
		t.Fatalf("Decide() error = %v after %d calls, want a permanent error before any call", err, decider.decides.Load())
	}
}
