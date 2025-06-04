from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from typing import List, Callable, Union
from typing_extensions import TypedDict
from langchain_core.documents import Document
import pandas as pd
from .document import PdfDocument


class BooleanResponse(BaseModel):
    """
    Manages a structured, boolean response from a language model.
    """
    content : bool = Field(
        description = 
            "Respond with False if the answer is No or Unknown. "
            "Respond True only if the answer is Yes. "
    )


# State for an individual page
class State(TypedDict):
    context : Document
    definition_bool : bool
    definition : str
    table_bool : bool


class PdfChat:
    """
    Base class used for managing a data extraction chat with a language model.
    """
    def __init__(self, llm : Callable):
        """
        Args:
            llm (Callable): Language model to use for the chat.

        Attrs:
            documents (DocumentDB): Document database to use for the chat.
        """
        self.llm = llm
        self.boolean_llm = llm.with_structured_output(BooleanResponse)

        self.prompt_template = PromptTemplate.from_template(
            "<start_of_turn>user\n{instructions}<end_of_turn>\n"
            "<start_of_turn>user\n{context}<end_of_turn>\n"
            "<start_of_turn>user\n{query}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        graph_builder = StateGraph(State)
        graph_builder.add_node("screen_definition", self.screen_definition)
        graph_builder.add_node("extract_definition", self.extract_definition)
        graph_builder.add_node("screen_table", self.screen_table)
        graph_builder.add_edge(START, "screen_definition")
        graph_builder.add_conditional_edges(
            "screen_definition",
            self.definition_routing,
            {True : "extract_definition", False: "screen_table"}
        )
        graph_builder.add_edge("extract_definition", "screen_table")
        graph_builder.add_edge("screen_table", END)
        self.graph = graph_builder.compile()
        self.response = None
        self.data = []


    def screen_definition(self, state: State):
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
        messages = self.prompt_template.invoke(
            {"instructions": instructions, "context": context, "query": query}
        )
        response = self.boolean_llm.invoke(messages)
        return {"definition_bool": response.content}
    

    def definition_routing(self, state : State):
        return state['definition_bool']
    

    def extract_definition(self, state: State):
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
        messages = self.prompt_template.invoke(
            {"instructions": instructions, "context": context, "query": query}
        )
        response = self.llm.invoke(messages)
        return {"definition": response.content}
    

    def screen_table(self, state: State):
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

        messages = self.prompt_template.invoke(
            {"instructions": instructions, "context": context, "query": query}
        )
        response = self.boolean_llm.invoke(messages)
        return {"table_bool": response.content}
    

    def table_routing(self, state : State):
        return state['table_bool']
    

    def fit(self, document: Document, title : str, page_num: int):
        """
        Fit the chat model to a given document.

        Args:
            document (Document): Document to use for the chat.
            title (str): Title of the document. Defaults to None.
            page_num (int): Page number of the document to use for the chat.

        Returns:
            response (dict): Response from the language model.
        """
        self.document = document
        self.title = title
        self.page_num = page_num
        self.response = self.graph.invoke({"context" : self.document.page_content})
        return self.response


    def record(self):
        """
        Record the results for the current page.
        """
        if self.response is None:
            raise ValueError(
                "No response to record. Please run the chat before recording."
            )
        
        definition = None
        if self.response["definition_bool"]:
            definition = self.response["definition"]

        self.data.append([
            self.title,
            self.page_num,
            self.response["definition_bool"],
            definition,
            self.response["table_bool"]
        ])


    def save(self, fname : str):
        """
        Save the screening results.

        Args:
            fname (str): Filename to save the results.
        """
        if len(self.data) == 0:
            raise ValueError("No response data to save. Please run the chat before saving.")
        
        df = pd.DataFrame(
            self.data,
            columns = ["title", "page", "definition_bool", "definition", "table_bool"]
        )
        df.to_csv(fname)


####################################################################################################


class ChatWithHistory:
    """
    Class to manage chat responses from an LLM or a callable Langgraph Graph.

    NOTE: This is designed for usage with models that are called using and .invoke() method.
    For best practice, you should also use a model which returns a consistently structured output
    (e.g. a string or a dictionary with consistent keys).
    """
    def __init__(self, llm : object):
        """
        Attrs:
            llm (object): Language model or graph to use for the chat.
            history (dict[dict[object, object]]): Dictionary with structure 
                {response identifier : response}.
        """
        self.llm = llm
        if not hasattr(llm, 'invoke'):
            raise ValueError(
                "The llm must be a callable with an 'invoke' method."
            )
        self.llm = llm
        self.history = {}


    def invoke(
        self,
        query : Union[str, dict],
        identifier : str = None,
        ignore : List[object] = None
    ) -> Union[str, dict]:
        """
        Run the language model or graph using the given query.

        Args:
            query (Union[str, dict]): Query to send to the model or graph.

            identifier (str, optional): Identifier for the query. If None, a new identifier 
                will be generated based on the current length of the history.

            ignore (List[object], optional): List of objects to ignore in the response. Ignored 
                objects will not be included in the response or history.
                Defaults to None, in which case the full response will be returned and stored.

        Returns:
            Union[str, dict]: Response from the model or graph.
        """
        if identifier is None:
            identifier = "query " + str(len(self.history))

        response = self.llm.invoke(query)
        if ignore is not None:
            # Filter out ignored objects from the response
            if isinstance(response, dict):
                response = {k: v for k, v in response.items() if k not in ignore}
            else:
                raise ValueError("Response must be a dictionary to filter ignored objects.")
            
        self.history[identifier] = response
        return response 
        

    def save(self, fname : str):
        """
        Save the screening results.

        Args:
            fname (str): Filename to save the results.
        """
        if len(self.history) == 0:
            raise ValueError("No response data to save. Please run the chat before saving.")
        
        df = pd.DataFrame.from_dict(
            self.history,
            orient="index"
        )
        df.to_csv(fname)
        return df

    
