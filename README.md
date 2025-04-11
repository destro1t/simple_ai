# Simple Neural Network (simple_nn)

![image](https://github.com/user-attachments/assets/d43a4230-bcc0-4923-a1da-026d6afe9905)

*Why the hell do we need Claude and your other paid neural networks?*

Welcome to **simple_nn**, a no-nonsense neural network built from scratch in Go. Forget bloated APIs and pricey subscriptions—this is a lean, mean, question-answering machine that learns from a simple text file. Whether you're a coder, a tinkerer, or just curious, `simple_nn` lets you dive into AI without the corporate fluff. Train it, tweak it, make it yours.

## Table of Contents

- [Why simple_nn?](#why-simple_nn)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Loading a Model](#loading-a-model)
  - [Asking Questions](#asking-questions)
- [File Formats](#file-formats)
  - [Training File (`text.txt`)](#training-file-texttxt)
  - [Model File (`text.bin`)](#model-file-textbin)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Why simple_nn?

Why fork over cash for Claude or other gated AI models? With `simple_nn`, you get:
- **Freedom**: Open-source, no strings attached.
- **Simplicity**: Runs on your machine with just Go and a text file.
- **Control**: Hack every layer, weight, and answer to fit your vibe.
- **Attitude**: Built for those who want AI without the hype.

We’re here to make AI accessible, transparent, and a little rebellious. `simple_nn` is your ticket to experimenting with neural networks on your terms.

## Features

- **Pure Go Neural Network**: Feedforward architecture with input, hidden (10 neurons), and output layers.
- **Text-Based Learning**: Trains on a `text.txt` file with question-answer pairs (e.g., `What’s up?: Yo, just chilling!`).
- **Creative Responses**: Temperature parameter (default: 1.2) for varied, spicy answers.
- **Model Saving**: Store trained models as `.bin` files for quick reuse.
- **Zero Dependencies**: No external libraries—just Go and grit.
- **Handles Sass**: Ready for casual, edgy, or downright wild questions.
- **Customizable**: Add your own data, tweak the code, make it sing.

---

## Installation

### Prerequisites
- **Go**: Version 1.24.2 or later (works great with 1.24.2).
- A text editor (VSCode, Vim, or whatever you roll with).

### Steps
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/simple_nn.git
   cd simple_nn
   ```

2. **Set Up Go Module**:
   ```bash
   go mod init simple_nn
   ```

3. **Build the Binary**:
   ```bash
   go build -o nn
   ```

4. **Test It Out**:
   ```bash
   ./nn
   ```
   Expected output:
   ```
   Usage: ./nn [command] [arguments]
   ```

---

## Usage

Run simple_nn with commands: `learn`, `load`, or `ask`. You can chain them for smooth workflows (e.g., `./nn load text.bin ask "Yo, what’s good?"`).

### Training the Model
Teach the network with a text file of questions and answers:
```bash
./nn learn text.txt
```

Output:
```
Learning complete. Save progress? (y/n)
```
Enter `y` to save as `text.bin`, or `n` to exit.

### Loading a Model
Load a trained model from a `.bin` file:
```bash
./nn load text.bin
```

Output:
```
Model loaded successfully.
```

### Asking Questions
Fire off a question to get a response:
```bash
./nn load text.bin ask "Who the hell are you?"
```

Example Output:
```
Model loaded successfully.
Response: I’m Neural AI, your digital wingman!
Pro Tip: Try informal or spicy questions—the temperature makes answers pop!
```

### Error Case
If you skip loading the model:
```bash
./nn ask "What’s up?"
```

Output:
```
Error: Model not loaded.
```

---

## File Formats

### Training File (`text.txt`)
A plain text file with question-answer pairs, one per line.

**Format**:
```
question: answer
```

**Sample `text.txt`**:
```text
How are you?: I’m doing great, thanks!
What’s your name?: Neural AI, at your service.
Who the hell are you?: Yo, just an AI with attitude!
What’s the vibe?: Chill, like a breeze in the cloud.
```

**Tips**:
- Use 300+ lines for better learning (sample has 320).
- Separate questions and answers with a colon `:`.
- Avoid empty lines or formatting errors.

### Model File (`text.bin`)
A binary file storing the trained model (weights, biases, keywords, answers).

- **Created**: When you save after training (`y`).
- **Used**: By `load` to restore the model.

---

## How It Works

**Architecture**:
- **Input Layer**: Matches unique keywords from training data.
- **Hidden Layer**: 10 neurons with sigmoid activation.
- **Output Layer**: One neuron per unique answer, with softmax + temperature.

**Training**:
- Backpropagation over 100 epochs (learning rate: 0.1).
- Temperature parameter (default 1.2) adds randomness for creative answers.
- Saves everything (weights, keywords, answers) to `.bin` via Go’s `gob`.

**Why It Rocks**:
- Train on memes, slang, or philosophy — it’s your playground.
- Open-source, hackable, and rebelliously simple.

---

## Contributing

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/yourusername/simple_nn.git
   ```

2. **Create a Branch**:
   ```bash
   git checkout -b feature/your-awesome-idea
   ```

3. **Hack Away**:
   - Add CLI options (e.g., `--temp=1.5` for crazier answers).
   - Improve training data or algorithms.
   - Fix bugs.

4. **Submit a PR**:
   - Push changes and open a PR with clear details.

**Ideas to Explore**:
- Multilingual support
- Deeper network architecture
- Web/GUI frontend
- Smarter keyword parsing

---

## License

Licensed under the [MIT License](LICENSE). Use it, mod it, share it—go wild.

---

## Contact

Got questions or want to collaborate? Reach out to **yourname@example.com** or hit us up on [GitHub Discussions](https://github.com/yourusername/simple_nn/discussions).
