import pandas as pd
import time
from workflows import *
from utils import *
from utils import _initialize_tools, _initialize_agent
from groq_local import llm as local_llm

# Load data
test_df = pd.read_csv("data/merged_ground_truth.csv")
os.environ["GROQ_API_KEY"] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

# Initialize tools and agent
tools = _initialize_tools()
agent = _initialize_agent(tools, local_llm)

# Initial state
state_init = {
    'symptoms': {'True': set(), 'False': set(), 'Unsure': set()},
    'medication': {'True': set(), 'False': set(), 'Unsure': set()},
    'processed_entities': [],
    'current_iter': 0,
    'current_subiter': 0,
    'received_answers': [],
    'current_retrieved_results': None,
    'step': "retrieval"
}

# Define solvers
solvers = {
    "hardcoded": lambda statement: hardcoded(statement, copy.deepcopy(state_init), {'max_iter': 3}, merged_case_search)[0],
    "RAG_default_workflow": lambda statement: RAG_default_workflow(statement, copy.deepcopy(state_init), {'max_iter': 3}, qa_chain_default)[0],
    "RAG_workflow": lambda statement: RAG_workflow(statement, copy.deepcopy(state_init), {'max_iter': 3})[0],
    "RAG_workflow_reflection": lambda statement: RAG_workflow_reflection(statement, copy.deepcopy(state_init), {'max_iter': 3}, local_llm)[0],
    "RAG_workflow_reflection_cot": lambda statement: RAG_workflow_reflection_cot(statement, copy.deepcopy(state_init), {'max_iter': 3}, local_llm)[0],
    "RAG_workflow_reflection_cot_sc": lambda statement: RAG_workflow_reflection_cot_sc(statement, copy.deepcopy(state_init), {'max_iter': 3}, local_llm, noisy_llm)[0],
    "ReAct_workflow": lambda statement: ReAct_workflow(statement, copy.deepcopy(state_init), {'max_iter': 3}, qa_chain_default, local_llm, agent)[0],
}

# Evaluate solvers
results_df = evaluate_and_collect_results(test_df, solvers)
results_df.to_csv('results2_new.csv', index=False)
