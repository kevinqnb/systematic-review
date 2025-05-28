from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from systematic_review import *

llm = ChatOllama(
    model="gemma3:27b-it-qat",
    temperature=0
)


prompt_template = PromptTemplate.from_template(
    "<start_of_turn>user\n{instructions}<end_of_turn>\n"
    "<start_of_turn>user\n{query}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

prompt = prompt_template.invoke({
    "instructions": "Only answer like a pirate.",
    "query": "What is the answer to life, the universe, and everything?"
})

print(llm.invoke(prompt))
