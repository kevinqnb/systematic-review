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

MEASUREMENT = "total organic carbon (TOC)"
UNITS = "mk/L or mg/kg"


####################################################################################################
# Define the states and functions to be used in the state graph

class State(TypedDict):
    abstract : str
    text : str
    abstract_bool : bool
    definition_bool : bool
    definition : str
    table_bool : bool
    measurement_bool : bool


def screen_abstract(state: State):
    """
    Screen the abstract of the current paper for relevance to coastal ecosystems.

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
            "Does this paper study coastal ecosystems? "
            "Coastal ecosystems may include but are not limited to intertidal zones, estuaries, "
            "lagoons, reefs, mangroves, marshes, seagrass meadows, kelp forests, and coastal wetlands."
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
        "You will be given contextual information from an excerpt of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown in the current excerpt, and not the paper as a whole. "
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes."
        "Answer in JSON format."
    )
    context = state["text"]
    query = (
        "Does this excerpt present a definition for a specific type of coastal ecosystem? "
        "A definition must specify quantitative attributes or descriptive characteristics that "
        "set this type of ecosystem apart from others. "
        "It should not merely quantify or describe a particular instance or example " 
        "of an ecosystem. "
        "Coastal ecosystems may include but are not limited to intertidal zones, estuaries, "
        "lagoons, reefs, mangroves, marshes, seagrass meadows, kelp forests, and coastal wetlands."
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
        "You will be given contextual information from an excerpt of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown in the current excerpt, and not the paper as a whole."
    )
    context = state["text"]
    query = (
        "Which coastal ecosystems are being studied and what definitions are provided for them? "
        "A definition must specify quantitative attributes or descriptive characteristics that "
        "set this type of ecosystem apart from others. "
        "It should not merely quantify or describe a particular instance or example " 
        "of an ecosystem. "
        "Coastal ecosystems may include but are not limited to intertidal zones, estuaries, "
        "lagoons, reefs, mangroves, marshes, seagrass meadows, kelp forests, and coastal wetlands."
    )
    messages = [
            {'role': 'system', 'content': instructions},
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


def screen_measurement(state: State):
    """
    Screen the current page for tabular data.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from an excerpt of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown in the current excerpt, and not the paper as a whole. "
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes. "
        "Answer in JSON format."
    )
    context = state["text"]
    query = (
        f"Does this excerpt include a measurement for {MEASUREMENT} in a coastal ecosystem? "
        f"A measurement should be a specific numeric value or range reported in units of {UNITS}. "
        "Coastal ecosystems may include but are not limited to intertidal zones, estuaries, "
        "lagoons, reefs, mangroves, marshes, seagrass meadows, kelp forests, and coastal wetlands."
    )

    messages = [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': context},
            {'role': 'user', 'content': query}
        ]
    response: ChatResponse = chat(model=MODEL, messages=messages, format=boolean_format)
    return {"measurement_bool": response_boolean_formatter(response)}


def table_routing(state : State):
    return state['table_bool']


def screen_table(state: State):
    """
    Screen the current page for tabular data.

    Args:
        state (State): Current state of the chat.
    Returns:
        state (State): Updated state with generated response.
    """
    instructions = (
        "You will be given contextual information from an excerpt of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown in the current excerpt, and not the paper as a whole. "
        "Your answer should be a boolean value with a value of False if the "
        "answer is No or Unknown and a value of True only if the answer is Yes. "
        "Answer in JSON format."
    )
    context = state["text"]
    query = (
        "Does this excerpt include a table containing data related to "
        "physical, chemical, or biological attributes of coastal ecosystems? "
        "Answer True only if the excerpt includes the actual table, and not just a reference to it. "
        "Coastal ecosystems may include but are not limited to intertidal zones, estuaries, "
        "lagoons, reefs, mangroves, marshes, seagrass meadows, kelp forests, and coastal wetlands."
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
graph_builder.add_node("screen_measurement", screen_measurement)
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
graph_builder.add_edge("screen_table", "screen_measurement")
graph_builder.add_edge("screen_measurement", END)
GRAPH = graph_builder.compile()
