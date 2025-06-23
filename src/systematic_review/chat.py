from typing import List, Union
import pandas as pd


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
        self.history = []


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
            identifier = {"query" : len(self.history)}

        response = self.llm.invoke(query)
        if ignore is not None:
            # Filter out ignored objects from the response
            if isinstance(response, dict):
                response = {k: v for k, v in response.items() if k not in ignore}
            else:
                raise ValueError("Response must be a dictionary to filter ignored objects.")
            
        id_response = identifier | response
        self.history.append(id_response)
        return response 
        

    def save(self, fname : str):
        """
        Save the screening results.

        Args:
            fname (str): Filename to save the results.
        """
        if len(self.history) == 0:
            raise ValueError("No response data to save. Please run the chat before saving.")
        
        df = pd.DataFrame(self.history)
        df.to_csv(fname)
        return df


####################################################################################################
