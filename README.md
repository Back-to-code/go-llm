# Go-LLM

A abstraction layer for communicating with LLM providers.

The following providers and features are supported:

|                                 | [OpenAi](https://openai.com/) | [Google Ai Studio](https://aistudio.google.com/) | [TogetherAi](https://www.together.ai/) | [Inception](https://www.inceptionlabs.ai/) |
| ------------------------------- | ----------------------------- | ------------------------------------------------ | -------------------------------------- | ------------------------------------------ |
| Completions                     | ✔️                            | ✔️                                               | ✔️                                     | ✔️                                         |
| Structured output (json)        | ✔️                            | ✔️                                               | ✔️                                     | ✔️                                         |
| Structured output (json schema) | ✔️                            | ✔️                                               | ✔️                                     | ✔️                                         |
| Streaming                       | ✔️                            |                                                  |                                        | ✔️                                         |
| Tools                           | ✔️                            | ✔️                                               |                                        | ✔️                                         |
| Tool choice                     | ✔️                            | ✔️                                               |                                        | ✔️                                         |

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
| Inception        | `INCEPTION_API_KEY`    |

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
    model := aimodels.ChatGpt6

    // Simple prompt
    response, err := model.PromptSingle("What is the capital of France?", llm.Options{})
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response)
}
```

## JSON schema output

```go
resp, err := model.PromptSingle("Name a color", llm.Options{
    ResponseFormat: llm.ResponseFormatJsonSchema,
    JsonSchema: llm.JsonSchema{
        Name:   "color",
        Schema: json.RawMessage(`{"type":"object","properties":{"color":{"type":"string"}},"required":["color"],"additionalProperties":false}`),
        Strict: true,
    },
})
```

## Tools

`Prompt` runs the tool loop: it resolves every tool call and sends the results back until the model answers.

```go
resp, err := model.Prompt(messages, llm.Options{
    Tools:        tools,
    ToolChoice:   llm.ToolChoiceRequired, // Only the first model call, later calls use auto
    MaxToolCalls: 10,                     // Defaults to 500, then the model has to answer without tools

    // Per model request, for logging or progress events
    OnCallStart: func(info llm.CallInfo) {},
    OnCallEnd:   func(info llm.CallInfo, result llm.CallResult) {}, // result.Retrying marks a failed attempt that is retried
    OnToolCall:  func(result llm.ToolResult) {},
})
```

A resolver returning a `string` or `json.RawMessage` is sent to the model as is, any other value is JSON encoded. An empty result is sent as `""`.
