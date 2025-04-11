package main

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"
	"nn/api"
)

var globalModel *api.Model

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) < 2 {
		fmt.Println("Использование: go run main.go [команда] [аргументы]")
		return
	}
	args := os.Args[1:]
	for i := 0; i < len(args); {
		command := args[i]
		switch command {
		case "learn":
			if i+1 >= len(args) {
				fmt.Println("Использование: learn <filename.txt>")
				return
			}
			api.Learning(args[i+1])
			i += 2
		case "load":
			if i+1 >= len(args) {
				fmt.Println("Использование: load <filename.bin>")
				return
			}
			var err error
			globalModel, err = api.LoadFromFile(args[i+1])
			if err != nil {
				fmt.Println("Ошибка загрузки модели:", err)
				return
			}
			fmt.Println("Модель успешно загружена.")
			i += 2
		case "ask":
			if i+1 >= len(args) {
				fmt.Println("Использование: ask <вопрос>")
				return
			}
			if globalModel == nil {
				fmt.Println("Ошибка: модель не загружена.")
				return
			}
			question := strings.Join(args[i+1:], " ")
			api.InputOutput(globalModel, question)
			i = len(args)
		default:
			fmt.Println("Неизвестная команда:", command)
			return
		}
	}
}
