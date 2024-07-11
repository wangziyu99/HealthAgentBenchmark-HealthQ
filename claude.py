import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
claude = ChatAnthropic(temperature=0, 
                       model_name="claude-3-opus-20240229")
