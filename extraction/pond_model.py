# Chat Models
from langchain_ollama import ChatOllama
from ollama import chat
from ollama import ChatResponse

# Chat structure
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate

# Typing
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import re


####################################################################################################
# Load the language model and manage prompting and structured responses

'''
LLM = ChatOllama(
    #model="gemma3:27b-it-qat",
    model="olmo2:13b-1124-instruct-q8_0",
    temperature=0,
    num_ctx = 128_000 # Maximum context length for Gemma3
)

prompt_template = PromptTemplate.from_template(
    "<start_of_turn>user\n{instructions}<end_of_turn>\n"
    "<start_of_turn>user\n{context}<end_of_turn>\n"
    "<start_of_turn>user\n{query}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

prompt_template = ChatPromptTemplate([
    ("user", "{instructions}"),
    ("user", "{context}"),
    ("user", "{query}"),
])



class BooleanResponse(BaseModel):
    """
    Manages a structured, boolean response from a language model.
    """
    content : bool = Field(
        description = 
            "Respond with False if the answer is No or Unknown. "
            "Respond True only if the answer is Yes. "
    )

boolean_llm = LLM.with_structured_output(schema = BooleanResponse)
'''

MODEL = "gemma3:27b-it-qat"
#MODEL = "olmo2:13b"

class BooleanResponse(BaseModel):
    """
    Manages a structured, boolean response from a language model.
    """
    content : bool = Field(
        description = 
            "Respond with False if the answer is No or Unknown. "
            "Respond True only if the answer is Yes. "
    )

boolean_format = BooleanResponse.model_json_schema()
response_boolean_formatter = lambda response: BooleanResponse.model_validate_json(
    response.message.content
).content


####################################################################################################
# Define the states and functions to be used in the state graph

class State(TypedDict):
    abstract : str
    text : str
    abstract_bool : bool
    definition_bool : bool
    definition : str
    table_bool : bool

'''
def screen_abstract(state: State):
    """
    Screen the abstract of the current paper for relevance to ponds or lakes.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    # Check that abstract has not already been screened
    if state.get("abstract_bool") is None:
        instructions = (
            "You will be given contextual information from the title and abstract of a "
            "scientific research paper and asked to accurately infer information about "
            "the paper's contents. Your answer should be a boolean value with a value "
            "of False if the answer is No or Unknown and a value of True only if the answer is Yes. "
        )
        context = state["abstract"]
        query = (
            "Does this paper study or discuss freshwater ponds or lakes in some capacity?"
        )
        messages = prompt_template.invoke(
            {"instructions": instructions, "context": context, "query": query}
        )
        response = boolean_llm.invoke(messages)
        return {"abstract_bool": response.content}
    else:
        return state


def screen_definition(state: State):
    """
    Screen the current page for a scientific definition.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole."
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes. "
    )
    context = state["text"]
    query = (
        "Does this page contain a definition for either ponds or lakes?"
        "A definition should specify distinguishing attributes or descriptive characteristics."
        "The definition may be for either ponds or lakes, but not other types of waterbodies."
    )
    messages = prompt_template.invoke(
        {"instructions": instructions, "context": context, "query": query}
    )
    response = boolean_llm.invoke(messages)
    return {"definition_bool": response.content}


def definition_routing(state : State):
    return state['definition_bool']


def extract_definition(state: State):
    """
    Extract a scientific definition from the given page.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole."
    )
    context = state["text"]
    query = (
        "What definition does the context give for either ponds or lakes?"
        "A definition should specify distinguishing attributes or descriptive characteristics."
        "The definition may be for either ponds or lakes, but not other types of waterbodies."
    )
    messages = prompt_template.invoke(
        {"instructions": instructions, "context": context, "query": query}
    )
    response = LLM.invoke(messages)
    return {"definition": response.content}


def screen_table(state: State):
    """
    Screen the current page for tabular data.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole."
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes. "
    )
    context = state["text"]
    query = (
        "Does this page include a table containing data related to "
        "physical, chemical, or biological attributes of individual ponds or lakes?"
        "Data must be reported in a table format, and should only be given for individually "
        "studied ponds or lakes, instead of aggregate statistics for groups of waterbodies. "
        "Examples include but are not limited to depth, surface area, temperature, or pH."
    )

    messages = prompt_template.invoke(
        {"instructions": instructions, "context": context, "query": query}
    )
    response = boolean_llm.invoke(messages)
    return {"table_bool": response.content}


def table_routing(state : State):
    return state['table_bool']
'''


def screen_abstract(state: State):
    """
    Screen the abstract of the current paper for relevance to ponds or lakes.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    # Check that abstract has not already been screened
    if state.get("abstract_bool") is None:
        instructions = (
            "You will be given contextual information from the title and abstract of a "
            "scientific research paper and asked to accurately infer information about "
            "the paper's contents. Your answer should be a boolean value with a value "
            "of False if the answer is No or Unknown and a value of True only if the answer is Yes."
            "Answer in JSON format."
        )
        context = state["abstract"]
        query = (
            "Does this paper study or discuss freshwater ponds or lakes in some capacity?"
        )
        messages = [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': context},
            {'role': 'user', 'content': query}
        ]
        response: ChatResponse = chat(model=MODEL, messages=messages, format=boolean_format)
        return {"abstract_bool": response_boolean_formatter(response)}
    else:
        return state


def screen_definition(state: State):
    """
    Screen the current page for a scientific definition.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole. "
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes."
        "Answer in JSON format."
    )
    context = state["text"]
    query = (
        "Does this page contain a definition for either ponds or lakes? "
        "A definition should specify distinguishing attributes or descriptive characteristics that "
        "generally separate this type of waterbody from others. "
        "A definition should not be simply be a description of an individual pond or lake, or a group "
        "of studied ponds or lakes. "
        "The definition must be specifically for either ponds or lakes, "
        "and not for other types of waterbodies or ecosystems."
    )
    messages = [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': context},
            {'role': 'user', 'content': query}
        ]
    response: ChatResponse = chat(model=MODEL, messages=messages, format=boolean_format)
    return {"definition_bool": response_boolean_formatter(response)}


def definition_routing(state : State):
    return state['definition_bool']


def extract_definition(state: State):
    """
    Extract a scientific definition from the given page.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole."
    )
    context = state["text"]
    query = (
        "What definition does the context give for either ponds or lakes? "
        "A definition should specify distinguishing attributes or descriptive characteristics that "
        "generally separate this type of waterbody from others. "
        "A definition should not be simply be a description of an individual pond or lake, or a group "
        "of studied ponds or lakes. "
        "The definition must be specifically for either ponds or lakes, "
        "and not for other types of waterbodies or ecosystems."
    )
    messages = [
            {'role': 'user', 'content': instructions},
            {'role': 'user', 'content': context},
            {'role': 'user', 'content': query}
        ]
    response: ChatResponse = chat(model=MODEL, messages=messages)
    clean_response = response.message.content
    clean_response = re.sub(
        r'</?start_of_turn>|</?end_of_turn>|</?end_start_of_turn>',
        '',
        clean_response
    ).strip()
    return {"definition": clean_response}


def screen_table(state: State):
    """
    Screen the current page for tabular data.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole. "
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes. "
        "Answer in JSON format."
    )
    context = state["text"]
    query = (
        "Does this page include a table containing data related to "
        "physical, chemical, or biological attributes of individual ponds or lakes? "
        "Data must be reported in a table format, and should only be given for individually "
        "studied ponds or lakes, instead of aggregate statistics for groups of waterbodies. "
        "Examples include but are not limited to depth, surface area, temperature, or pH."
    )

    messages = [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': context},
            {'role': 'user', 'content': query}
        ]
    response: ChatResponse = chat(model=MODEL, messages=messages, format=boolean_format)
    return {"table_bool": response_boolean_formatter(response)}


def table_routing(state : State):
    return state['table_bool']


#####################################################################################################
# Build the state graph

graph_builder = StateGraph(State)
graph_builder.add_node("screen_abstract", screen_abstract)
graph_builder.add_node("screen_definition", screen_definition)
graph_builder.add_node("extract_definition", extract_definition)
graph_builder.add_node("screen_table", screen_table)
graph_builder.add_edge(START, "screen_abstract")
graph_builder.add_conditional_edges(
    "screen_abstract",
    lambda state: state['abstract_bool'] if state.get('text') is not None else False,
    {True: "screen_definition", False: END}
)
graph_builder.add_conditional_edges(
    "screen_definition",
    lambda state: state['definition_bool'],
    {True : "extract_definition", False: "screen_table"}
)
graph_builder.add_edge("extract_definition", "screen_table")
graph_builder.add_edge("screen_table", END)
GRAPH = graph_builder.compile()
