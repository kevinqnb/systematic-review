# Chat Models
from langchain_ollama import ChatOllama

# Chat structure
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, END, StateGraph

# Typing
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


####################################################################################################
# Load the language model and manage prompting and structured responses

LLM = ChatOllama(
    model="gemma3:12b-it-qat",
    temperature=0,
    num_ctx = 128_000 # Maximum context length for Gemma3
)

prompt_template = PromptTemplate.from_template(
    "<start_of_turn>user\n{instructions}<end_of_turn>\n"
    "<start_of_turn>user\n{context}<end_of_turn>\n"
    "<start_of_turn>user\n{query}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

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


####################################################################################################
# Define the states and functions to be used in the state graph

class State(TypedDict):
    context : str
    definition_bool : bool
    definition : str
    table_bool : bool


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
    context = state["context"]
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
    context = state["context"]
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
    context = state["context"]
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


#####################################################################################################
# Build the state graph

graph_builder = StateGraph(State)
graph_builder.add_node("screen_definition", screen_definition)
graph_builder.add_node("extract_definition", extract_definition)
graph_builder.add_node("screen_table", screen_table)
graph_builder.add_edge(START, "screen_definition")
graph_builder.add_conditional_edges(
    "screen_definition",
    definition_routing,
    {True : "extract_definition", False: "screen_table"}
)
graph_builder.add_edge("extract_definition", "screen_table")
graph_builder.add_edge("screen_table", END)
GRAPH = graph_builder.compile()