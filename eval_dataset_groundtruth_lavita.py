# %% [markdown]
# # Evaluations
# 
#  - use groq open source models for local app implementations(the system itself only use
#  - use Claude 3 to 1. tidy ground truth data, and 2. evaluate as judge

# %%
import os
os.environ["GROQ_API_KEY"] = "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"

os.environ["ANTHROPIC_API_KEY"] = "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"


# %%
from claude import claude
from groq_local import llm as local_llm

# %%
first_run = False
import pandas as pd

if first_run:
    from datasets import load_dataset

    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    df = pd.DataFrame(dataset['train'])
    df.to_csv("lavita_chatdoctor_note.csv", index = False)



    import numpy as np
    np.random.seed(3)   

    from sklearn.model_selection import train_test_split

    train_indices, test_indices = train_test_split(df.index, test_size=64, random_state=3)

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    train_df.to_csv("train_" + "lavita_chatdoctor_note.csv", index = False)
    test_df.to_csv("test_" + "lavita_chatdoctor_note.csv", index = False)

else:
    train_df = pd.read_csv("train_" + "lavita_chatdoctor_note.csv", )
    test_df = pd.read_csv("test_" + "lavita_chatdoctor_note.csv", )

train_df = train_df.reset_index()


# %%
train_df

# %% [markdown]
# Only use train data to build the VDB (here we do not really train anything, but we do not leak test data into out rag or agent system)

# %%
from df_as_search_db import get_any_case_df_as_vdb as get_db
from df_as_search_db import lazy_collate, collate_with_metadata, collate_with_metadata_with_output
from functools import partial

train_w_output = get_db(train_df,
    collate_func=collate_with_metadata_with_output,
    # first_run=True,
    db_path="lavita_train_with_output_vdb")
train = get_db(train_df,
    collate_func=collate_with_metadata,
    # first_run=True,
    db_path="lavita_train_cases_vdb")



train_w_output_lazy = get_db(train_df,
    collate_func=partial(lazy_collate, cols=train_df.columns),
    # first_run=True,
    db_path="lavita_train_w_output_lazy_vdb")

train_lazy = get_db(train_df,
    collate_func=partial(lazy_collate, cols=train_df.columns),
    # first_run=True,
    db_path="lavita_train_lazy_vdb")


# %% [markdown]
#  - get ground truth data from test_df, we use claude to tidy them up.
#  - here use langchain, and create a fallback and retry mechanism.

# %%
test_df.iloc[5,1]

# %%
len(test_df)

# %% [markdown]
# 

# %%
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain import PromptTemplate, LLMChain
from langchain.chains.base import Chain

handler = ConsoleCallbackHandler()  # StdOutCallbackHandler()

# %%
def df_row_to_ground_truth_parse_control(question, llm):


    response_schemas = [
        ResponseSchema(
            name="Patient_known_knowledge",
            description="What patient knows about the symptoms and background information",
            type="string"
        ),
        ResponseSchema(
            name="Doctor_question_statements",
            description="All information doctor asked and stated.",
            type='string'
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    "
    {question}
    " 
    Above is a piece of collated patient note data, complete the task below
    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(input={"question": question}, config={'callbacks': [handler]})

# %%
# df_row_to_ground_truth_parse_control(lazy_collate(train_df.iloc[0,:],  cols=train_df.columns  ), llm=claude)

# %% [markdown]
# this is not bad, we can check hallucination by reflection but it is not discussed here.

# %%
# test_df = test_df.iloc[0:2,:]

# %%
def process_function(i):
    return df_row_to_ground_truth_parse_control(lazy_collate(test_df.iloc[i,:],  cols=test_df.columns  ), llm=claude)

# %%
# def perform_operation(index):
#     # Placeholder function for performing an operation that may raise an exception
#     # Replace this with your actual operation
#     if index % 2 == 0:
#         raise ValueError("Even index not supported")
#     return f"Result for index {index}"

def process_with_retries(indices, n_retries, unit_func , fallback='failed'):
    results = []
    for index in indices:
        retry_count = 0
        while retry_count <= n_retries:
            try:
                result = unit_func(index)
                results.append([index, result])
                break
            except Exception as e:
                retry_count += 1
                if retry_count > n_retries:
                    print((index, "failed"))
                    results.append([index, fallback])
                    break
    return results

n_retries = 3
results = process_with_retries(range(len(test_df)), n_retries, process_function,fallback = {'Patient_known_knowledge': 'failed', 'Doctor_question_statements':'failed'})
print(results[0])

# %%
ground_truth_test_df_claude = pd.DataFrame( [results[i][1] for i in range(len(results))] )
ground_truth_test_df_claude.to_csv('lavita_ground_truth.csv', index = False)

# %%
ground_truth_test_df_claude = pd.read_csv('lavita_ground_truth.csv', )
# ground_truth_test_df_claude

# %%



