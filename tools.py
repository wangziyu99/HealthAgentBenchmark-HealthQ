from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from medialpy_as_tool import medial_search
from df_as_search_db import lazy_collate as collate_func, instructor_embeddings
from functools import partial
from hf_data_no_auth import search_cases_contents

merged_case_search = partial(search_cases_contents, db = db_merged ,sep = '\n', k = 10)

# Define tools
def initialize_tools():
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
def initialize_agent(tools, llm):
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
