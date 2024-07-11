# Conversational Health Agent Evaluation

## Overview
This project evaluates different conversational health agent workflows using various evaluation metrics. The agents are designed to ask follow-up questions based on initial patient statements and relevant medical cases. The project includes hardcoded workflows, RAG (Retrieval-Augmented Generation) workflows, and ReAct workflows.

## Workflows
- **Hardcoded Workflow**: Fast and simple, but not very intelligent.
- **RAG Workflows**: Use retrieval-augmented generation to generate questions.
  - Basic RAG
  - RAG with Reflection
  - RAG with n-step Chain of Thought (CoT)
  - RAG with n-step CoT-SC
- **ReAct Workflow**: Uses a ReAct agent with original and custom prompts.

## Metrics
- **LLM Interrogation**: Evaluates the agent's questions based on known ground truth using `claude`.
- **Fake Patient Statement + Answer**: Simulates patient answers to the agent's questions using `claude`.
- **Summarization-based Evaluation**: Measures how much information is retrieved after the first question using metrics like ROUGE and BERTScore.

## Setup
1. Clone the repository.
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
3. Place the necessary data files in the `data` directory.

## Running the Evaluation
Run the evaluation script:
```sh
python main.py
