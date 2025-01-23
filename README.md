# Conversational Health Agent Evaluation - HealthQ

## Overview
This repository contains the codebase for evaluating questioning capabilities of conversational health agents using the HealthQ framework. The framework benchmarks how well Large Language Model (LLM) healthcare chains, including Retrieval-Augmented Generation (RAG) and ReAct workflows, generate context-aware and diagnostically relevant questions during patient interactions.

HealthQ provides a **benchmarking suite** to systematically evaluate the questioning capabilities of LLM healthcare chains. By integrating advanced workflows and robust evaluation metrics, this framework sets a new standard for assessing LLM-driven medical conversations.

---

## Highlights

- **State-of-the-Art Benchmarks**: A first-of-its-kind framework to benchmark questioning capabilities in healthcare-focused LLM chains.
- **Advanced LLM Workflows**: Implements and evaluates workflows such as RAG, Chain of Thought (CoT), and ReAct.
- **Multi-Metric Evaluation**: Assesses question quality using metrics like specificity, relevance, usefulness, and fluency.
- **Realistic Simulations**: Uses virtual patient simulators and public datasets (e.g., ChatDoctor, MTS-Dialog) for realistic evaluation scenarios.
- **Customizable Framework**: Flexible architecture for adding new workflows or evaluation criteria.

---

## Workflows

### **Hardcoded Workflow**

A simple rule-based approach that serves as a baseline for comparison. It uses:
- **Named Entity Recognition (NER)** to extract medical terms from patient input.
- **Predefined Templates** to generate follow-up questions based on extracted entities.

This workflow is lightweight and computationally efficient but lacks adaptability and contextual reasoning.

### **RAG Workflows**

Retrieval-Augmented Generation (RAG) workflows combine knowledge retrieval with generative capabilities, enabling more dynamic and context-aware question generation. Variants include:

1. **Basic RAG**:
   - Retrieves relevant information from a vector database.
   - Generates questions by integrating retrieved knowledge with patient context.

2. **RAG with Reflection**:
   - Introduces iterative improvement by allowing the model to reflect on its initial questions.
   - Enhances question quality by re-evaluating the relevance and specificity of generated outputs.

3. **RAG with n-step Chain of Thought (CoT)**:
   - Implements step-by-step reasoning to refine questions.
   - Each reasoning step incorporates additional context from retrieved information, ensuring logical progression.

4. **RAG with n-step CoT-Self-Consistency (CoT-SC)**:
   - Extends CoT by generating multiple reasoning chains and consolidating results.
   - Uses a self-consistency mechanism to ensure robustness and minimize contradictions in final questions.

### **ReAct Workflow**

The ReAct (Reasoning and Acting) workflow employs:
- **Tool-Integrated Reasoning**: Combines LLM reasoning with external tools, such as vector database queries and medical term lookup APIs.
- **Dynamic Decision-Making**: The LLM decides whether to act (e.g., retrieve data or run a tool) or reason further at each step.

This workflow excels in handling complex, multi-step interactions and dynamically adjusting to new information during conversations.

---

## Benchmark Details

HealthQ is designed as a **comprehensive benchmarking suite** for questioning capabilities in LLM healthcare chains. It establishes consistent and reproducible evaluation scenarios to:

1. **Assess Question Quality**:
   - Evaluate generated questions across five key dimensions: specificity, relevance, usefulness, fluency, and coverage.
   - Use GPT-4, GPT-3.5, and Claude as model-agnostic LLM judges.

2. **Measure Information Retrieval**:
   - Assess how well LLM-generated questions elicit critical patient information using ROUGE and NER-based similarity metrics.

3. **Enable Comparisons**:
   - Compare advanced workflows (e.g., RAG with Reflection) against baselines like Hardcoded and ReAct workflows.
   - Provide detailed metrics for reproducibility and transparency.

4. **Dataset-Driven Evaluation**:
   - Leverage two public datasets (ChatDoctor and MTS-Dialog) to simulate diverse and realistic patient scenarios.

---

## Evaluation Metrics

### 1. LLM Judge Interrogation
- **Specificity**: Precision of the question targeting patient symptoms.
- **Usefulness**: Diagnostic value of the generated question.
- **Relevance**: Alignment with the patient’s context and medical history.
- **Fluency**: Clarity and coherence of question phrasing.
- **Coverage**: Breadth of patient information explored.

### 2. Virtual Patient Simulation
- Simulates patient responses to questions, ensuring realistic and diverse conversational scenarios.

### 3. Summarization-Based Evaluation
- Uses ROUGE (Recall-Oriented Understudy for Gisting Evaluation) and NER (Named Entity Recognition)-based similarity metrics to measure information completeness.

---

## Dataset

- **ChatDoctor**:
  - 110,000 anonymized medical conversations covering symptoms, diagnoses, and treatments.
- **MTS-Dialog**:
  - 1,700 doctor-patient dialogues with clinical notes summarizing interactions.

---

## Code Structure

```
├── healthq_framework.py    # Core implementation of HealthQ framework.
├── workflows/             # Contains implementations of RAG, ReAct, and CoT workflows.
├── evaluation/            # Evaluation metrics and utility functions.
├── data/                  # Input datasets and processed files.
├── main.py                # Main script to run evaluations.
├── results/               # Stores generated outputs and logs.
└── README.md              # Documentation.
```

---

## Quick Start

### Step 1: Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Datasets

Place the ChatDoctor and MTS-Dialog datasets in the `data/` directory.

### Step 3: Run Evaluation

Run the main evaluation script:
```bash
python main.py
```

This will evaluate all workflows and generate logs in the `results/` directory.

---

## Outputs

1. **Generated Questions**:
   - Stored in `results/questions/` by workflow and dataset.

2. **Evaluation Metrics**:
   - Includes scores for specificity, relevance, ROUGE, and NER-based metrics.
   - Summary results stored in `results/metrics_summary.json`.

3. **Logs**:
   - Detailed logs for each workflow execution stored in `results/logs/`.

---

## Citation

If you find this work useful, please cite:

```text
@article{wang2024healthq,
  title={HealthQ: Unveiling Questioning Capabilities of LLM Chains in Healthcare Conversations},
  author={Ziyu Wang and others},
  journal={arXiv preprint arXiv:2409.19487},
  year={2024}
}
```

---

## License

This project is released under the MIT License:

```text
MIT License

Copyright (c) 2025 Ziyu Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

For questions or collaboration inquiries, please contact:

**Ziyu Wang**
