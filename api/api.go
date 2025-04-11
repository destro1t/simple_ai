package api

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

// Pair представляет пару вопрос-ответ
type Pair struct {
	Question string
	Answer   string
}

// TrainingSample содержит входной вектор и целевой индекс ответа
type TrainingSample struct {
	Input  []float64
	Target int
}

// Model описывает структуру нейронной сети
type Model struct {
	Keywords            []string   // Список ключевых слов
	Answers             []string   // Список уникальных ответов
	WeightsInputHidden  [][]float64 // Веса между входным и скрытым слоями
	BiasHidden          []float64  // Смещения скрытого слоя
	WeightsHiddenOutput [][]float64 // Веса между скрытым и выходным слоями
	BiasOutput          []float64  // Смещения выходного слоя
}

// SaveToFile сохраняет модель в бинарный файл
func (m *Model) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	enc := gob.NewEncoder(file)
	return enc.Encode(m)
}

// LoadFromFile загружает модель из бинарного файла
func LoadFromFile(filename string) (*Model, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	dec := gob.NewDecoder(file)
	var m Model
	err = dec.Decode(&m)
	if err != nil {
		return nil, err
	}
	return &m, nil
}

// Learning обучает модель на основе текстового файла
func Learning(filename string) {
	pairs, err := readPairs(filename)
	if err != nil {
		fmt.Println("Ошибка чтения файла:", err)
		return
	}
	answers := extractUniqueAnswers(pairs)
	keywords := extractKeywords(pairs)
	trainingData, err := createTrainingData(pairs, keywords, answers)
	if err != nil {
		fmt.Println("Ошибка создания обучающих данных:", err)
		return
	}
	model := initializeModel(keywords, answers, len(keywords), len(answers))
	trainModel(model, trainingData, 100, 0.1)
	fmt.Println("Обучение завершено. Сохранить прогресс? (y/n)")
	var response string
	fmt.Scanln(&response)
	if strings.ToLower(response) == "y" {
		saveFilename := strings.Replace(filename, ".txt", ".bin", 1)
		err := model.SaveToFile(saveFilename)
		if err != nil {
			fmt.Println("Ошибка сохранения модели:", err)
		} else {
			fmt.Println("Модель сохранена в", saveFilename)
		}
	} else {
		fmt.Println("Выход без сохранения.")
	}
}

// InputOutput обрабатывает входной вопрос и возвращает ответ
func InputOutput(model *Model, input string) {
	if model == nil {
		fmt.Println("Ошибка: модель не загружена.")
		return
	}
	inputVector := createInputVector(input, model.Keywords)
	hiddenInput := addVector(dot(model.WeightsInputHidden, inputVector), model.BiasHidden)
	hiddenOutput := applySigmoid(hiddenInput)
	outputInput := addVector(dot(model.WeightsHiddenOutput, hiddenOutput), model.BiasOutput)
	output := softmaxWithTemperature(outputInput, 1.2) // Температура 1.2 для разнообразия
	answerIndex := sampleFromProbabilities(output)
	answer := model.Answers[answerIndex]
	fmt.Println("Ответ:", answer)
}

// readPairs читает пары вопрос-ответ из файла
func readPairs(filename string) ([]Pair, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var pairs []Pair
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 {
			question := strings.TrimSpace(parts[0])
			answer := strings.TrimSpace(parts[1])
			pairs = append(pairs, Pair{Question: question, Answer: answer})
		}
	}
	return pairs, scanner.Err()
}

// extractUniqueAnswers извлекает уникальные ответы
func extractUniqueAnswers(pairs []Pair) []string {
	answerSet := make(map[string]bool)
	for _, pair := range pairs {
		answerSet[pair.Answer] = true
	}
	answers := make([]string, 0, len(answerSet))
	for answer := range answerSet {
		answers = append(answers, answer)
	}
	return answers
}

// extractKeywords извлекает уникальные ключевые слова из вопросов
func extractKeywords(pairs []Pair) []string {
	wordSet := make(map[string]bool)
	for _, pair := range pairs {
		words := strings.Fields(strings.ToLower(pair.Question))
		for _, word := range words {
			wordSet[word] = true
		}
	}
	keywords := make([]string, 0, len(wordSet))
	for word := range wordSet {
		keywords = append(keywords, word)
	}
	return keywords
}

// createTrainingData создаёт обучающие данные
func createTrainingData(pairs []Pair, keywords []string, answers []string) ([]TrainingSample, error) {
	answerIndex := make(map[string]int)
	for i, answer := range answers {
		answerIndex[answer] = i
	}
	var trainingData []TrainingSample
	for _, pair := range pairs {
		inputVector := createInputVector(pair.Question, keywords)
		target, ok := answerIndex[pair.Answer]
		if !ok {
			return nil, fmt.Errorf("ответ не найден: %s", pair.Answer)
		}
		trainingData = append(trainingData, TrainingSample{Input: inputVector, Target: target})
	}
	return trainingData, nil
}

// initializeModel инициализирует модель с случайными весами
func initializeModel(keywords []string, answers []string, inputSize, outputSize int) *Model {
	hiddenSize := 10
	weightsInputHidden := make([][]float64, inputSize)
	for i := range weightsInputHidden {
		weightsInputHidden[i] = make([]float64, hiddenSize)
		for j := range weightsInputHidden[i] {
			weightsInputHidden[i][j] = rand.Float64()*0.2 - 0.1
		}
	}
	biasHidden := make([]float64, hiddenSize)
	for i := range biasHidden {
		biasHidden[i] = rand.Float64()*0.2 - 0.1
	}
	weightsHiddenOutput := make([][]float64, hiddenSize)
	for i := range weightsHiddenOutput {
		weightsHiddenOutput[i] = make([]float64, outputSize)
		for j := range weightsHiddenOutput[i] {
			weightsHiddenOutput[i][j] = rand.Float64()*0.2 - 0.1
		}
	}
	biasOutput := make([]float64, outputSize)
	for i := range biasOutput {
		biasOutput[i] = rand.Float64()*0.2 - 0.1
	}
	return &Model{
		Keywords:            keywords,
		Answers:             answers,
		WeightsInputHidden:  weightsInputHidden,
		BiasHidden:          biasHidden,
		WeightsHiddenOutput: weightsHiddenOutput,
		BiasOutput:          biasOutput,
	}
}

// trainModel обучает модель с помощью обратного распространения
func trainModel(model *Model, trainingData []TrainingSample, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for _, sample := range trainingData {
			input := sample.Input
			target := sample.Target
			hiddenInput := addVector(dot(model.WeightsInputHidden, input), model.BiasHidden)
			hiddenOutput := applySigmoid(hiddenInput)
			outputInput := addVector(dot(model.WeightsHiddenOutput, hiddenOutput), model.BiasOutput)
			output := softmax(outputInput)
			dOutputInput := make([]float64, len(output))
			copy(dOutputInput, output)
			dOutputInput[target] -= 1
			dWeightsHiddenOutput := outer(dOutputInput, hiddenOutput)
			dBiasOutput := dOutputInput
			dHiddenOutput := dotTranspose(model.WeightsHiddenOutput, dOutputInput)
			dHiddenInput := mulVector(dHiddenOutput, sigmoidDerivativeVector(hiddenInput))
			dWeightsInputHidden := outer(dHiddenInput, input)
			dBiasHidden := dHiddenInput
			model.WeightsInputHidden = subMatrix(model.WeightsInputHidden, scaleMatrix(dWeightsInputHidden, learningRate))
			model.BiasHidden = subVector(model.BiasHidden, scaleVector(dBiasHidden, learningRate))
			model.WeightsHiddenOutput = subMatrix(model.WeightsHiddenOutput, scaleMatrix(dWeightsHiddenOutput, learningRate))
			model.BiasOutput = subVector(model.BiasOutput, scaleVector(dBiasOutput, learningRate))
		}
	}
}

// createInputVector создаёт входной вектор для вопроса
func createInputVector(question string, keywords []string) []float64 {
	words := strings.Fields(strings.ToLower(question))
	wordSet := make(map[string]bool)
	for _, word := range words {
		wordSet[word] = true
	}
	inputVector := make([]float64, len(keywords))
	for i, keyword := range keywords {
		if wordSet[keyword] {
			inputVector[i] = 1.0
		}
	}
	return inputVector
}

// softmaxWithTemperature применяет softmax с температурой
func softmaxWithTemperature(x []float64, temperature float64) []float64 {
	maxVal := x[0]
	for _, val := range x {
		if val > maxVal {
			maxVal = val
		}
	}
	expSum := 0.0
	scaled := make([]float64, len(x))
	for i, val := range x {
		scaled[i] = math.Exp((val - maxVal) / temperature)
		expSum += scaled[i]
	}
	result := make([]float64, len(x))
	for i := range x {
		result[i] = scaled[i] / expSum
	}
	return result
}

// sampleFromProbabilities выбирает индекс ответа на основе вероятностей
func sampleFromProbabilities(probs []float64) int {
	r := rand.Float64()
	sum := 0.0
	for i, p := range probs {
		sum += p
		if r <= sum {
			return i
		}
	}
	return len(probs) - 1 // Возвращаем последний индекс в крайнем случае
}

// Математические вспомогательные функции
func dot(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix[0]))
	for j := 0; j < len(matrix[0]); j++ {
		sum := 0.0
		for i := 0; i < len(vector); i++ {
			sum += matrix[i][j] * vector[i]
		}
		result[j] = sum
	}
	return result
}

func addVector(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func subVector(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func scaleVector(v []float64, scalar float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] * scalar
	}
	return result
}

func subMatrix(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

func scaleMatrix(m [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(m))
	for i := range m {
		result[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

func outer(a, b []float64) [][]float64 {
	result := make([][]float64, len(b))
	for i := range b {
		result[i] = make([]float64, len(a))
		for j := range a {
			result[i][j] = b[i] * a[j]
		}
	}
	return result
}

func dotTranspose(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		sum := 0.0
		for j := 0; j < len(vector); j++ {
			sum += matrix[i][j] * vector[j]
		}
		result[i] = sum
	}
	return result
}

func applySigmoid(v []float64) []float64 {
	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = 1 / (1 + math.Exp(-val))
	}
	return result
}

func sigmoidDerivativeVector(v []float64) []float64 {
	result := make([]float64, len(v))
	for i, val := range v {
		s := 1 / (1 + math.Exp(-val))
		result[i] = s * (1 - s)
	}
	return result
}

func mulVector(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, val := range x {
		if val > maxVal {
			maxVal = val
		}
	}
	expSum := 0.0
	for _, val := range x {
		expSum += math.Exp(val - maxVal)
	}
	result := make([]float64, len(x))
	for i, val := range x {
		result[i] = math.Exp(val - maxVal) / expSum
	}
	return result
}
