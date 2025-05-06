from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Callable
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from .documents import DocumentDB


class StructuredResponse(BaseModel):
    """
    Manages a structured response from a language model.
    """
    success : bool = Field(
        description = 
            "Respond with False for a negative or unknown answer. "
            "Respond True only if the question is answered affirmitavely. "
    )
    
    reason : str = Field(
        description = 
            "Point to a specific location or excerpt "
            "from the text to justify your reasoning. "
            "'None' if nothing extracted."
    )

    '''
    values: list[float] = Field(
        description="Array of all numerical data found for the property. " \
                    "Exctract all values seen in the context. "
    )
    units: list[str] = Field(
        description="Array with units of measurement for each value extracted. "
                    "Include a unit for every data point extracted. " \
                    "This could mean repeating a unit if its values appear multiple times."
    )
    '''



class State(TypedDict):
    retrieval_query : str
    prompt : str
    context : List[Document]
    answer : StructuredResponse


class Chat:
    """
    Base class used for managing a data extraction chat with a language model.
    """
    def __init__(self, llm : Callable, n_documents : int = 5):
        """
        Args:
            llm (Callable): Language model to use for the chat.

        Attrs:
            documents (DocumentDB): Document database to use for the chat.
        """
        self.llm = llm.with_structured_output(StructuredResponse)
        self.n_documents = n_documents
        self.documents = None

        self.instructions = (
            "You will be given contextual information from a scientific research paper "
            "and asked to accurately answer questions about its contents. "
            "If applicable, point to a specific location or excerpt "
            "from the text to justify your reasoning."
        )

        self.prompt_template = ChatPromptTemplate([
            ("system", self.instructions),
            ("human", "Context : {context} \n\n Question : {prompt} \n\n Answer :"),
        ])

        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        self.graph = graph_builder.compile()


    def retrieve(self, state: State):
        retrieved_docs = self.documents.retrieve(state["retrieval_query"], k = self.n_documents)
        return {"context": retrieved_docs}


    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        messages = self.prompt_template.invoke(
            {"prompt": state["prompt"], "context": docs_content}
        )
        response = self.llm.invoke(messages)
        state["answer"] = response
        return state
    

    def fit(self, documents: DocumentDB):
        """
        Fit the chat model to the given documents.

        Args:
            documents (DocumentDB): Document database to use for the chat.
        """
        self.documents = documents


    def screen(self, retreival_query : str, prompt : str):
        """
        Screen the given prompt using the language model.

        Args:
            prompt (str): Prompt to screen.

        Returns:
            str: Response from the language model.
        """
        if self.documents is None:
            raise ValueError("Documents not loaded. Please load documents before screening.")
        output = self.graph.invoke({"retrieval_query": retreival_query, "prompt" : prompt})
        return output
    

    def extract(self):
        """
        Extract data from the documents using the language model.

        Returns:
            str: Extracted data from the documents.
        """
        if self.documents is None:
            raise ValueError("Documents not loaded. Please load documents before extracting.")
        pass
    
