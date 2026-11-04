package llm_test

import (
	"fmt"
	"log"
	"time"

	llm "github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/aimodels"
)

// A support desk routes a customer question before a slower model writes an answer: send a help article when one
// answers it and hand an angry customer to a person.
func ExampleModel_Decide() {
	const noArticle = "none"
	articles := []llm.Choice{
		{Value: "refunds", Description: "Getting your money back for an order"},
		{Value: "shipping", Description: "Delivery times and tracking a parcel"},
		{Value: "account", Description: "Changing your password, email or address"},
		{Value: noArticle, Description: "No article answers the question"},
	}

	decisions, err := aimodels.ChatGpt6Nano.Decide("Customer question: where is my parcel? I ordered it two weeks ago!", []llm.Question{
		{Name: "article", Type: llm.QuestionChoice, Instructions: "Pick the article that answers the question.", Choices: articles},
		{Name: "angry", Type: llm.QuestionPredicate, Instructions: "Is the customer angry?"},
	}, llm.DecisionOptions{Timeout: 5 * time.Second})
	if err != nil {
		// The decision only saves work, the question still gets an answer without it.
		log.Printf("routing failed, answering in full: %v", err)
		return
	}

	article, _ := decisions.Answer("article")
	switch {
	case article.ProbabilityOf(noArticle) >= 0.5:
		fmt.Println("no article fits, answer in full")
	case article.ProbabilityOf(article.Choice) >= 0.3:
		fmt.Println("send the article", article.Choice)
	default:
		fmt.Println("too unsure, answer in full")
	}

	angry, _ := decisions.Answer("angry")
	if angry.Probability >= 0.7 {
		fmt.Println("hand over to a person")
	}
}
