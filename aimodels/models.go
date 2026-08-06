package aimodels

import (
	"github.com/Back-to-code/go-llm"
	"github.com/Back-to-code/go-llm/googleaistudio"
	"github.com/Back-to-code/go-llm/inception"
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
	// By default the models use the lowest option of thinking (so no thinking in most cases), the level of thinking can be enabled inside llm.Options
	// = PRICY - ULTRA TURBO EXPENSIVE
	ChatGpt5   = register("gpt-5.6-sol", &openai.Provider{})
	Gemini3Pro = register("gemini-3.1-pro-preview", &googleaistudio.Provider{})
	Best       = ChatGpt5 // <- Default

	// Mini models.
	// When the nano model is not good enough but the good model is somewhat too expensive
	// This is most of the time a good middleground
	// = OKE ISH PRICE
	ChatGpt5Mini     = register("gpt-5.4-mini", &openai.Provider{})
	Gemini3FlashLite = register("gemini-3.5-flash-lite", &googleaistudio.Provider{})
	Mini             = ChatGpt5Mini // <- Default

	// Default nano model.
	// For basic llm tasks mainly smart parttern matching tasks are these models perfect for
	// Or giving simple things a score.
	// = DIRT CHEAP
	ChatGpt5Nano = register("gpt-5-nano", &openai.Provider{})                    // Note that 5.4 nano and 5.6 luna are much more expensive
	Gemini2Flash = register("gemini-2.5-flash-lite", &googleaistudio.Provider{}) // Note that 3.5 flash lite is much more expensive than 2.5 flash lite
	Mercury2     = register("mercury-2", &inception.Provider{})
	Nano         = ChatGpt5Nano // <- Default
)
