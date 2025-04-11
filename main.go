package main

import (
    "fmt"
    "math/rand"
    "nn/api"
    "os"
    "strings"
    "time"
)

var globalModel *api.Model

func main() {
    rand.Seed(time.Now().UnixNano())
    if len(os.Args) < 2 {
        fmt.Println("Usage: go run main.go [command] [arguments]...")
        return
    }

    args := os.Args[1:]
    for i := 0; i < len(args); {
        command := args[i]
        switch command {
        case "learn":
            if i+1 >= len(args) {
                fmt.Println("Usage: learn <filename.txt>")
                return
            }
            api.Learning(args[i+1])
            i += 2
        case "load":
            if i+1 >= len(args) {
                fmt.Println("Usage: load <filename.bin>")
                return
            }
            var err error
            globalModel, err = api.LoadFromFile(args[i+1])
            if err != nil {
                fmt.Println("Error loading model:", err)
                return
            }
            fmt.Println("Model loaded successfully.")
            i += 2
        case "ask":
            if i+1 >= len(args) {
                fmt.Println("Usage: ask <question>")
                return
            }
            if globalModel == nil {
                fmt.Println("Error: model not loaded.")
                return
            }
            question := strings.Join(args[i+1:], " ")
            api.InputOutput(globalModel, question)
            i = len(args)
        default:
            fmt.Println("Unknown command:", command)
            return
        }
    }
}
