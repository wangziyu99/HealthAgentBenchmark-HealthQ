import re
import time
import copy
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from df_as_search_db import get_any_case_df_as_vdb as get_db
from fake_patient import ground_truth_to_first_answer
from judge import judge
from _get_metric_dict import _medagent_dict, get_score_dict
from NER_as_tool import extract_medication_and_symptom
from hardcoded_workflow import hard_tf_parser, hardcoded
import os
from functools import partial
from hf_data_no_auth import search_cases_contents
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from medialpy_as_tool import medial_search
from df_as_search_db import lazy_collate as collate_func, instructor_embeddings
from claude import claude

# Define environment variables for API keys
os.environ["GROQ_API_KEY"] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

# Load vector databases
db_lavita = get_db(db_path="lavita_train_lazy_vdb")
db_mts = get_db(db_path="mts_train_lazy_vdb")

# Define the embeddings model and load the vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
db_merged = FAISS.load_local(folder_path="merged_train_vdb", embeddings=embeddings)
merged_case_search = partial(search_cases_contents, db=db_merged, sep='\n', k=10)

def search_cases(query, db=db_merged):
    results = db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
    return results

def format_retrieval_results(results):
    return '\n'.join([doc.page_content for doc in results])

# Retry mechanism for functions
def retry_with_timing(*args, n_retry=3, wait_time=1, **kwargs):
    retry_count = 0
    start_time = time.time()
    while retry_count < n_retry:
        try:
            result = args[0](*args[1:], **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            return result, execution_time, retry_count
        except Exception as e:
            retry_count += 1
            if retry_count >= n_retry:
                end_time = time.time()
                execution_time = end_time - start_time
                return "failed", execution_time, retry_count
            else:
                time.sleep(wait_time)
    end_time = time.time()
    execution_time = end_time - start_time
    return "failed", execution_time, retry_count

# Extract tool and arguments from text
def extract_tool_and_argument(text):
    lines = text.split('\n')
    tool_name = None
    argument = None
    for line in lines:
        if line.startswith('Tool:'):
            tool_name = re.sub(r'\\+', '', line[5:].strip())
        elif line.startswith('Argument:'):
            argument = re.sub(r'\\+', '', line[9:].strip())
        if tool_name and argument:
            break
    return tool_name, argument

# Format tools and tool memory
def format_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

def format_tool_memory(tool_memory):
    return "\n".join([f"Previous Tool: {entry[0]}\nPrevious Argument: {entry[1]}\nOutput: {entry[2]}\n" for entry in tool_memory])

# Evaluation function
def evaluate_and_collect_results(test_df, solvers):
    times = []
    results = []
    retries = []
    fake_patient_statements = []
    fake_first_answers = []
    sources = []
    apps = []
    interrogate_metrics = []
    summarization_metrics = []
    i_s = []

    for i in range(len(test_df)):
        fake_patient_statement = test_df.iloc[i]['Patient_first_statement']
        patient_known_knowledge = test_df.iloc[i]['Patient_known_knowledge']
        source = test_df.iloc[i]["source"]

        for solver_name, solver_func in solvers.items():
            i_s.append(i)
            apps.append(solver_name)
            print(f"Solver: {solver_name}")
            result, execution_time, retry_count = retry_with_timing(solver_func, fake_patient_statement, n_retry=3, wait_time=120)
            results.append(result)
            times.append(execution_time)
            retries.append(retry_count)
            fake_patient_statements.append(fake_patient_statement)
            sources.append(source)
            interrogate_metrics.append(judge(patient_known_knowledge, fake_patient_statement, result, claude))
            fake_first_answer = ground_truth_to_first_answer(patient_known_knowledge, result, claude)['Patient_Answer']
            fake_first_answers.append(fake_first_answer)
            summarization_metrics.append(get_score_dict(result, patient_known_knowledge, flatten=True, s1_tasks={}, s2_tasks=_medagent_dict))
            print(f"Finished\n")
            print(result, execution_time, retry_count)

    interrogate_df = pd.DataFrame(interrogate_metrics)
    summary_df = pd.concat(summarization_metrics)
    results_df = pd.concat((interrogate_df.reset_index(), summary_df.reset_index()), axis=1)
    results_df['source'] = sources
    results_df['app'] = apps
    results_df['fake_patient_statement'] = fake_patient_statement
    results_df['output'] = results
    results_df['fake_first_answer'] = fake_first_answers
    results_df['i'] = i_s
    results_df = results_df.drop('index', axis=1)
    return results_df




# Define tools
def _initialize_tools():
    similar_case_search_tool = Tool(
        name="similar_case_search",
        func=merged_case_search,
        description="Use it to search similar cases for patients in the database",
    )
    medial_search_tool = Tool(
        name="medical_abbreviation_search",
        func=medial_search,
        description="Use it to explain medical abbreviations",
    )
    return [similar_case_search_tool, medial_search_tool]

# Initialize agent
def _initialize_agent(tools, llm):
    token_limit = 64
    memory = ConversationBufferMemory(memory_key="chat_history", token_limit=token_limit)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
    )
    return agent
