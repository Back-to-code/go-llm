# Go-LLM

A abstraction layer for communicating with LLM providers.

The following providers and features are supported:

|                                 | [OpenAi](https://openai.com/) | [Google Ai Studio](https://aistudio.google.com/) | [TogetherAi](https://www.together.ai/) |
| ------------------------------- | ----------------------------- | ------------------------------------------------ | -------------------------------------- |
| Completions                     | ✔️                            | ✔️                                               | ✔️                                     |
| Structured output (json)        | ✔️                            | ✔️                                               | ✔️                                     |
| Structured output (json schema) |                               |                                                  |                                        |
| Streaming                       | ✔️                            |                                                  |                                        |
| Tools                           | ✔️                            | ✔️                                               |                                        |

DO NOT MAKE THIS REPO PRIVATE! This library can now be easially imported from other go project without having to configured annoying shell variables.

## Installation

```bash
go get -u github.com/Back-to-code/go-llm@latest
```

## Environment Variables

Before using the library, you need to set up API keys for the providers you want to use:

| Provider         | Environment            |
| ---------------- | ---------------------- |
| OpenAI           | `OPENAI_TOKEN`         |
| Google AI Studio | `GOOGLE_AI_STUDIO_KEY` |
| Together AI      | `TOGETHER_AI_TOKEN`    |

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "os"

    "github.com/Back-to-code/go-llm"
    "github.com/Back-to-code/go-llm/aimodels"
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
