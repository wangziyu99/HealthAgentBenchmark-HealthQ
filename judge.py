"""Judge chain


"""

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
import os

os.environ["GROQ_API_KEY"] = "YOUR API KEY"

os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

handler = ConsoleCallbackHandler()  # StdOutCallbackHandler()

def judge(patient_known_knowledge, patient_in, system_question, llm):


    response_schemas = [
        ResponseSchema(
            name="Specificity",
            description="How specific is the question asked? Rate from 1-10",
            type="string"
        ),
        ResponseSchema(
            name="Usefulness",
            description="How useful is the question asked? Rate from 1-10",
            type="string"
        ),
        ResponseSchema(
            name="Relevance",
            description="How relevant is the new question to the patient's symptom? Rate from 1-10",
            type="string"
        ),

        ResponseSchema(
            name="Coverage",
            description="how much this new question cover the information of the full ground truth data? Rate from 1-10",
            type="string"
        ),

        ResponseSchema(
            name="Fluency",
            description="How fluent and human friendly is this question? Rate from 1-10",
            type="string"
        ),
        ResponseSchema(
            name="Comments",
            description=" other comments on the question from the system.",
            type="string"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    "
    {patient_known_knowledge}
    " 
    Above is all the infomation the patient knows.
    
    The patient gives the application this information:
    {patient_in}

    The application comes up with this question:
    {system_question}

     Using these data, make up a text statement to test a clinical application.
     The target is to ask questions useful for diagnosis, not necessarily mentioned by patient.
     If the app asks a question not mentioned by patient, but useful for diagnosis, it should have low coverage but still reasonable usefulness. 
     - relevance: how relevant is the new question to the patient's symptom? 
     - coverage: how much this new question cover the information of the full ground truth data?
     - fluency: is this question written in fluent, human language?    
     {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["patient_known_knowledge"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(input={"patient_known_knowledge": patient_known_knowledge,
     "patient_in": patient_in,"system_question": system_question,}, config={'callbacks': [handler]})
