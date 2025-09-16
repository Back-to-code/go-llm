# Go-LLM

A abstraction layer for communicating with LLM providers.

The following providers and features are supported:

| | [OpenAi](https://openai.com/) | [Google Ai Studio](https://aistudio.google.com/) | [TogetherAi](https://www.together.ai/) |
|---|---|---|---|
| Completions | ✔️ | ✔️ | ✔️ |
| Structured output (json) | ✔️ | ✔️ | ✔️ |
| Structured output (json schema) | | | |
| Streaming | ✔️ | | |
| Tools | ✔️ | | |

## Installation

```bash
go get -u bitbucket.org/teamscript/go-llm@latest
```

## Environment Variables

Before using the library, you need to set up API keys for the providers you want to use:

- **OpenAI**: `OPENAI_TOKEN` - Your OpenAI API key
- **Google AI Studio**: `GOOGLE_AI_STUDIO_KEY` - Your Google AI Studio API key
- **Together AI**: `TOGETHER_AI_TOKEN` - Your Together AI API key

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "os"

    "bitbucket.org/teamscript/go-llm"
    "bitbucket.org/teamscript/go-llm/aimodels"
)

func main() {
	  os.SetEnv("OPENAI_TOKEN", "api-token-here")

	  // Look inside the aimodels package for all models that are pre configured
    model := aimodels.ChatGpt5

    // Simple prompt
    response, err := model.PromptSingle("What is the capital of France?", llm.Options{})
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response)
}
```
