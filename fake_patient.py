
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


def ground_truth_to_first_question(patient_known_knowledge, llm):
    """Make up patient statement from ground truth"""

    response_schemas = [
        ResponseSchema(
            name="Patient_first_statement",
            description="What patient knows about the symptoms and background information",
            type="string"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    "
    {patient_known_knowledge}
    " 
    Above is all the infomation the patient knows. Using these data, make up a text statement to test a clinical application.
    Imitation of patient: based on all data provided by patient known knowledge, make up the first thing patient may say to the doctor in a clinic.
    Note that one is not likely to dump all the information to the doctor at once. 
    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["patient_known_knowledge"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(input={"patient_known_knowledge": patient_known_knowledge}, config={'callbacks': [handler]})

def ground_truth_to_first_answer(patient_known_knowledge, bot_question, llm):
    """Make up patient statement from ground truth"""

    response_schemas = [
        ResponseSchema(
            name="Patient_Answer",
            description="What the patient can answer about this question based on the patient known knowledge. For the aspects not presented in the patient known information, just say you are not sure about that specific aspect",
            type="string"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """
    The patient knows:
    "
    {patient_known_knowledge}
    " 
    Above is all the infomation the patient knows. Using these data, answer the question the doctor asks below.

    The doctor asks:
    {bot_question}

    Imitation of patient: 
    Note that one is not likely to dump all the information to the doctor at once. 
    {format_instructions}
    """

    prompt = PromptTemplate(
        input_variables=["patient_known_knowledge", "bot_question"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain.invoke(input={"patient_known_knowledge": patient_known_knowledge,"bot_question": bot_question}, config={'callbacks': [handler]})