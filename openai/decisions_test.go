package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	llm "github.com/Back-to-code/go-llm"
)

// A real response of the Decisions API, with every kind of answer.
const decisionsResponse = `{
	"model": "gpt-6-luna",
	"answers": [
		{"type": "choice", "name": "document", "choice": "Koks", "probabilities": [
			{"value": "Koks", "probability": 0.97}, {"value": "Geen", "probability": 0.03}
		], "confidence": 0.94},
		{"type": "predicate", "name": "horeca", "probability": 0.99},
		{"type": "score", "name": "niveau", "score": 0.8, "probabilities": [
			{"value": 0, "label": "laag", "probability": 0.33},
			{"value": 1, "label": "midden", "probability": 0.54},
			{"value": 2, "label": "hoog", "probability": 0.13}
		], "confidence": 0.31}
	],
	"usage": {"input_tokens": 385, "input_tokens_details": {"cached_tokens": 128}, "output_tokens": 0}
}`

func decisionsServer(t *testing.T, status int, response string, request *map[string]any) {
	t.Helper()
	os.Setenv("OPENAI_TOKEN", "test-token")
	t.Cleanup(func() { os.Unsetenv("OPENAI_TOKEN") })

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/decisions" {
			t.Errorf("request went to %s, want /v1/decisions", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, request)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)

	prev := BaseURL
	BaseURL = server.URL
	t.Cleanup(func() { BaseURL = prev })
}

var decisionQuestions = []llm.Question{
	{Name: "document", Type: llm.QuestionChoice, Instructions: "Welk document past?", Choices: []llm.Choice{
		{Value: "Koks", Description: "Koks en keukenmedewerkers"}, {Value: "Geen"},
	}},
	{Name: "horeca", Type: llm.QuestionPredicate, Instructions: "Gaat de wens over horeca?"},
	{Name: "niveau", Type: llm.QuestionScore, Instructions: "Welk niveau?", Levels: []llm.Level{
		{Label: "laag"}, {Label: "midden"}, {Label: "hoog", Description: "leidinggevend"},
	}},
}

func TestDecideSendsTheQuestionsAndReadsEveryKindOfAnswer(t *testing.T) {
	var request map[string]any
	decisionsServer(t, http.StatusOK, decisionsResponse, &request)

	decisions, err := (&Provider{}).Decide("gpt-6-luna", "Zoekwens: kok", decisionQuestions, llm.DecisionOptions{})
	if err != nil {
		t.Fatalf("Decide() error = %v", err)
	}

	sent, _ := json.Marshal(request)
	want := `{"input":"Zoekwens: kok","model":"gpt-6-luna","questions":[` +
		`{"choices":[{"description":"Koks en keukenmedewerkers","value":"Koks"},{"value":"Geen"}],"instructions":"Welk document past?","name":"document","type":"choice"},` +
		`{"instructions":"Gaat de wens over horeca?","name":"horeca","type":"predicate"},` +
		`{"instructions":"Welk niveau?","levels":[{"label":"laag"},{"label":"midden"},{"description":"leidinggevend","label":"hoog"}],"name":"niveau","type":"score"}]}`
	if string(sent) != want {
		t.Errorf("request = %s\nwant      %s", sent, want)
	}

	choice, _ := decisions.Answer("document")
	if choice.Type != llm.QuestionChoice || choice.Choice != "Koks" || choice.ProbabilityOf("Geen") != 0.03 || choice.Confidence != 0.94 {
		t.Errorf("choice answer = %+v", choice)
	}
	predicate, _ := decisions.Answer("horeca")
	if predicate.Type != llm.QuestionPredicate || predicate.Probability != 0.99 {
		t.Errorf("predicate answer = %+v", predicate)
	}
	score, _ := decisions.Answer("niveau")
	if score.Score != 0.8 || score.ProbabilityOf("1") != 0.54 || score.Probabilities[2].Label != "hoog" {
		t.Errorf("score answer = %+v", score)
	}
	if decisions.Usage != (llm.TokenUsage{InputTokens: 385, CachedInputTokens: 128}) {
		t.Errorf("usage = %+v", decisions.Usage)
	}
}

func TestDecideReturnsABadRequestAsAPermanentError(t *testing.T) {
	var request map[string]any
	decisionsServer(t, http.StatusBadRequest, `{"error":{"message":"array too short","type":"invalid_request_error"}}`, &request)

	_, err := (&Provider{}).Decide("gpt-6-luna", "input", decisionQuestions[:1], llm.DecisionOptions{})
	if !llm.IsPermanent(err) {
		t.Fatalf("Decide() error = %v, want a permanent error", err)
	}
}

func TestDecideRefusesAResponseWithoutAnAnswerPerQuestion(t *testing.T) {
	var request map[string]any
	decisionsServer(t, http.StatusOK, decisionsResponse, &request)

	_, err := (&Provider{}).Decide("gpt-6-luna", "input", decisionQuestions[:1], llm.DecisionOptions{})
	if err == nil {
		t.Fatal("Decide() error = nil, want one for 3 answers to 1 question")
	}
}
