package log

import "fmt"

var Logger func(msg string)

func Info(msg string) {
	if Logger != nil {
		Logger(msg)
	} else {
		fmt.Println("go-llm:", msg)
	}
}
