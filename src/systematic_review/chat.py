import pandas as pd
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
    response : bool = Field(
        description = 
            "Respond with False for a negative or unknown answer. "
            "Respond True only if the question is answered affirmitavely. "
    )
    
    reason : str = Field(
        description = 
            "Specify location or excerpt "
            "from the text to justify your reasoning. "
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
    name : str
    retrieval_query : str
    prompt : str
    context : List[Document]
    answer : StructuredResponse


class Chat:
    """
    Base class used for managing a data extraction chat with a language model.
    """
    def __init__(self, llm : Callable, n_documents : int = 1):
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
        self.screening_messages = None
        self.screening_data = []


    def retrieve(self, state: State):
        """
        Retrieve documents from the document database based on the retrieval query.
        NOTE: This function is called by the graph and should not be called directly.

        Args:
            state (State): Current state of the chat.
        Returns:
            state (State): Updated state with retrieved documents.
        """
        print("Retrieving documents...")
        retrieved_docs = self.documents.retrieve(state["retrieval_query"], k = self.n_documents)
        return {"context": retrieved_docs}


    def generate(self, state: State):
        """
        Generate a response using the language model based on the retrieved documents.
        NOTE: This function is called by the graph and should not be called directly.

        Args:
            state (State): Current state of the chat.
        Returns:
            state (State): Updated state with generated response.
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        messages = self.prompt_template.invoke(
            {"prompt": state["prompt"], "context": docs_content}
        )
        print("Generating response...")
        response = self.llm.invoke(messages)
        print("Response generated.")
        return {"answer": response}
    

    def fit(self, documents: DocumentDB, title : str = None):
        """
        Fit the chat model to the given documents.

        Args:
            documents (DocumentDB): Document database to use for the chat.
        """
        self.documents = documents
        self.title = title


    def screen(
            self,
            name : str,
            retrieval_query : str,
            prompt : str
        ):
        """
        Screen the given prompt using the language model.

        Args:
            name (str) : Name for the prompt (for recording purposes).
            retrieval_query (str) : Query to pass to the vector DB retriever.
            prompt (str): Prompt to screen.

        Returns:
            StructuredResponse: Response from the language model.
        """
        if self.documents is None:
            raise ValueError("Documents not loaded. Please load documents before screening.")
        self.screening_messages = self.graph.invoke(
            {
                "name" : name,
                "retrieval_query": retrieval_query,
                "prompt" : prompt,
            }
        )
        return self.screening_messages['answer']
    

    def extract(self):
        """
        Extract data from the documents using the language model.

        Returns:
            str: Extracted data from the documents.
        """
        if self.documents is None:
            raise ValueError("Documents not loaded. Please load documents before extracting.")
        pass


    def screen_record(self):
        """
        Record the screening results.
        """
        if self.screening_messages is None:
            raise ValueError(
                "No screening messages to record. Please run screening before recording."
            )
        question_name = self.screening_messages['name']
        response = self.screening_messages['answer'].response
        reason = self.screening_messages['answer'].reason
        docs = self.screening_messages['context']
        pages = [int(d.metadata['page']) for d in docs]
        self.screening_data.append([self.title, question_name, response, False, reason, pages])


    def screen_save(self, fname : str):
        """
        Save the screening results.

        Args:
            fname (str): Filename to save the results.
        """
        if len(self.screening_data) == 0:
            raise ValueError("No screening data to save. Please run screening before saving.")
        
        df = pd.DataFrame(
            self.screening_data,
            columns = ["title", "question", "response", "truth", "reason", "pages"]
        )
        df.to_csv(fname)
        

    
