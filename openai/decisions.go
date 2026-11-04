package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/Back-to-code/go-llm"
)

var _ llm.Decider = &Provider{}

type decisionChoice struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

type decisionLevel struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

type decisionQuestion struct {
	Name         string           `json:"name,omitempty"`
	Type         llm.QuestionType `json:"type"`
	Instructions string           `json:"instructions"`
	Choices      []decisionChoice `json:"choices,omitempty"`
	Levels       []decisionLevel  `json:"levels,omitempty"`
}

type decisionRequest struct {
	Model     string             `json:"model"`
	Input     string             `json:"input"`
	Questions []decisionQuestion `json:"questions"`
}

type decisionProbability struct {
	Value       json.RawMessage `json:"value"`
	Label       string          `json:"label"`
	Probability float64         `json:"probability"`
}

type decisionAnswer struct {
	Type          llm.QuestionType      `json:"type"`
	Name          string                `json:"name"`
	Choice        json.RawMessage       `json:"choice"`
	Probability   float64               `json:"probability"`
	Score         float64               `json:"score"`
	Probabilities []decisionProbability `json:"probabilities"`
	Confidence    float64               `json:"confidence"`
}

type decisionResponse struct {
	Answers []decisionAnswer `json:"answers"`
	Usage   struct {
		InputTokens        int `json:"input_tokens"`
		OutputTokens       int `json:"output_tokens"`
		InputTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
	} `json:"usage"`
}

// Decide asks POST /v1/decisions.
func (*Provider) Decide(model string, input string, questions []llm.Question, options llm.DecisionOptions) (llm.Decisions, error) {
	body := decisionRequest{Model: model, Input: input, Questions: make([]decisionQuestion, len(questions))}
	for idx, question := range questions {
		body.Questions[idx] = toDecisionQuestion(question)
	}

	resp, err := newRequest("/v1/decisions", body, options.Timeout, options.Ctx)
	if err != nil {
		return llm.Decisions{}, fmt.Errorf("failed to send decisions request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return llm.Decisions{}, llm.NewErr(resp)
	}

	var decoded decisionResponse
	err = json.NewDecoder(resp.Body).Decode(&decoded)
	if err != nil {
		return llm.Decisions{}, fmt.Errorf("failed to decode decisions response: %w", err)
	}
	if len(decoded.Answers) != len(questions) {
		return llm.Decisions{}, fmt.Errorf("decisions response has %d answers for %d questions", len(decoded.Answers), len(questions))
	}

	decisions := llm.Decisions{
		Answers: make([]llm.Answer, len(decoded.Answers)),
		Usage: llm.TokenUsage{
			InputTokens:       decoded.Usage.InputTokens,
			OutputTokens:      decoded.Usage.OutputTokens,
			CachedInputTokens: decoded.Usage.InputTokensDetails.CachedTokens,
		},
	}
	for idx, answer := range decoded.Answers {
		decisions.Answers[idx] = fromDecisionAnswer(answer)
	}
	return decisions, nil
}

func toDecisionQuestion(question llm.Question) decisionQuestion {
	converted := decisionQuestion{Name: question.Name, Type: question.Type, Instructions: question.Instructions}
	for _, choice := range question.Choices {
		converted.Choices = append(converted.Choices, decisionChoice{Value: choice.Value, Description: choice.Description})
	}
	for _, level := range question.Levels {
		converted.Levels = append(converted.Levels, decisionLevel{Label: level.Label, Description: level.Description})
	}
	return converted
}

func fromDecisionAnswer(answer decisionAnswer) llm.Answer {
	converted := llm.Answer{
		Name:        answer.Name,
		Type:        answer.Type,
		Choice:      rawValue(answer.Choice),
		Probability: answer.Probability,
		Score:       answer.Score,
		Confidence:  answer.Confidence,
	}
	for _, probability := range answer.Probabilities {
		converted.Probabilities = append(converted.Probabilities, llm.Probability{
			Value:       rawValue(probability.Value),
			Label:       probability.Label,
			Probability: probability.Probability,
		})
	}
	return converted
}

// rawValue is a choice value, which the API sends as a string or a boolean, or the index of a score level.
func rawValue(raw json.RawMessage) string {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	return strings.TrimSpace(string(raw))
}
