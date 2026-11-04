package llm

import (
	"context"
	"errors"
	"fmt"
	"time"
)

type QuestionType string

const (
	QuestionChoice    QuestionType = "choice"
	QuestionPredicate QuestionType = "predicate"
	QuestionScore     QuestionType = "score"
	// QuestionRefusal is the type of an answer the model refused to give, it carries no probabilities.
	QuestionRefusal QuestionType = "refusal"
)

// Question is answered on its own, independent of the other questions of a Decide call. A choice question needs 2
// to 255 Choices, a score question 2 to 10 Levels from the lowest to the highest, a predicate question neither. Name
// is optional, the answer carries it back.
type Question struct {
	Name         string
	Type         QuestionType
	Instructions string
	Choices      []Choice
	Levels       []Level
}

type Choice struct {
	Value       string
	Description string
}

type Level struct {
	Label       string
	Description string
}

// Answer of a choice question has the chosen value as Choice and a Probability per choice value, of a predicate
// question the Probability that it holds, of a score question the probability weighted mean of the level indexes as
// Score and a Probability per level, with the level index as Value. Confidence is set for choice and score answers.
type Answer struct {
	Name          string
	Type          QuestionType
	Choice        string
	Probability   float64
	Score         float64
	Probabilities []Probability
	Confidence    float64
}

type Probability struct {
	Value       string
	Label       string
	Probability float64
}

// ProbabilityOf is 0 for a value the answer does not have.
func (a Answer) ProbabilityOf(value string) float64 {
	for _, probability := range a.Probabilities {
		if probability.Value == value {
			return probability.Probability
		}
	}
	return 0
}

// Decisions has an answer per question, in the order of the questions.
type Decisions struct {
	Answers []Answer
	Usage   TokenUsage
}

func (d Decisions) Answer(name string) (Answer, bool) {
	for _, answer := range d.Answers {
		if answer.Name == name {
			return answer, true
		}
	}
	return Answer{}, false
}

// DecisionOptions work like the Options of the same name, Timeout defaults to 30 seconds.
type DecisionOptions struct {
	Ctx         context.Context
	Timeout     time.Duration
	NoRetry     bool
	OnCallStart func(CallInfo)
	OnCallEnd   func(CallInfo, CallResult)
}

// Decider is a provider that answers questions with one of their fixed answers instead of with text. Decide sends
// exactly one request, it does not retry.
type Decider interface {
	Decide(model string, input string, questions []Question, options DecisionOptions) (Decisions, error)
}

// Decide answers every question about input in one request, retrying it on transient errors like Prompt. The hooks
// get input as the only message. It fails with a permanent error when the provider of m is no Decider.
func (m *Model) Decide(input string, questions []Question, options DecisionOptions) (Decisions, error) {
	decider, ok := m.Provider.(Decider)
	if !ok {
		return Decisions{}, Permanent(fmt.Errorf("provider %T does not support decisions", m.Provider))
	}
	if len(questions) == 0 {
		return Decisions{}, Permanent(errors.New("decide needs at least one question"))
	}
	if options.Timeout <= 0 {
		options.Timeout = time.Second * 30
	}

	info := CallInfo{Model: m.Name, Messages: []Message{User(input)}}
	hooks := attemptHooks{ctx: options.Ctx, noRetry: options.NoRetry, onStart: options.OnCallStart, onEnd: options.OnCallEnd}

	return withRetries(info, hooks, func() (Decisions, CallResult, error) {
		decisions, err := decider.Decide(m.Name, input, questions, options)
		return decisions, CallResult{Usage: decisions.Usage}, err
	})
}
