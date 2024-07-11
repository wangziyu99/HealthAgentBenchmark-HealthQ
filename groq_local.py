
from langchain_groq import ChatGroq
llm = ChatGroq(temperature=0,
    model_name="mixtral-8x7b-32768")


noisy_llm = llm = ChatGroq(temperature=0.3,
    model_name="mixtral-8x7b-32768")

