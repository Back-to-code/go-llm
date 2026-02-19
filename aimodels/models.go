package aimodels

import (
	"github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/googleaistudio"
	"github.com/Back-to-code/go-llm/openai"
)

var models = map[string]*llm.Model{}

func register(name string, provider llm.Provider) *llm.Model {
	model := &llm.Model{Name: name, Provider: provider}
	models[name] = model
	return model
}

func GetModel(name string) *llm.Model {
	return models[name]
}

var (
	// The best models with with the option to think.
	// Should be used if the mini model is not good enough.
	// By deafult the models use the lowest option of thinking (so no thinking in most cases), the level of thinking can be enabled inside llm.Options
	// = PRICY - ULTRA EXPENSIVE
	ChatGpt5   = register("gpt-5.2", &openai.Provider{})
	Gemini3Pro = register("gemini-3-pro-preview", &googleaistudio.Provider{})
	Best       = ChatGpt5 // <- Deafult

	// Mini models.
	// When the nano model is not good enough but the good model is somewhat too expensive
	// This is most of the time a good middleground
	// = CHEAP
	ChatGpt5Mini = register("gpt-5-mini", &openai.Provider{})
	Gemini3Flash = register("gemini-3-flash-preview", &googleaistudio.Provider{})
	Mini         = ChatGpt5Mini // <- Deafult

	// Default nano model.
	// For basic llm tasks mainly smart parttern matching tasks are these models perfect for
	// Or giving simple things a score.
	// = DIRT CHEAP
	ChatGpt5Nano = register("gpt-5-nano", &openai.Provider{})
	Gemini2Flash = register("gemini-2.0-flash", &googleaistudio.Provider{})
	Nano         = ChatGpt5Nano // <- Deafult
)
