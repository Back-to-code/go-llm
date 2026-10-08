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
	// By default the thinking level is left to the provider, it can be set inside llm.Options
	// = PRICY - ULTRA TURBO EXPENSIVE
	ChatGpt6   = register("gpt-6.1-sol", &openai.Provider{})                    // Note that 6 astra is 2.5x the price
	Gemini3Pro = register("gemini-3.1-pro-preview", &googleaistudio.Provider{}) // No newer or stable pro model exists yet
	Best       = ChatGpt6                                                       // <- Default

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
	// No Gemini model is cheap enough for this tier.
	ChatGpt6Nano = register("gpt-6-luna", &openai.Provider{})
	Mercury2     = register("mercury-2.5", &inception.Provider{})
	Nano         = ChatGpt6Nano // <- Default
)
