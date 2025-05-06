from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


class DocumentDB:
    """
    Base class used for managing a database of text from pdf documents.
    """
    def __init__(
            self,
            embeddings: HuggingFaceEmbeddings,
            chunks : bool = False
        ):
        """
        Args:
            embeddings (HuggingFaceEmbeddings): Embeddings model to use.
            chunks (bool): Whether to split the document into smaller chunks of text or not.
                Defaults to False, in which case the pdf document is loaded as a collection 
                of pages.
            collection_name (str): Name of the collection in the vector database.
            persist (bool): Whether to persist the database or not.
            persist_path (str): Path to persist the database.

        Attrs:
            database (Chroma): Vector database.
        """
        self.embeddings = embeddings
        self.chunks = chunks
        self.documents = None
        self.document_ids = None
        self.database = InMemoryVectorStore(
            embedding=self.embeddings
        )


    def load(self, filepath: str):
        """
        Load the document from the given path.

        Args:
            filepath (str): Path to the document.

        """
        if self.documents is not None:
            self.clear()

        self.documents = []
        self.document_ids = None
        if not self.chunks:
            # Load as individual pages:
            loader = PyPDFLoader(filepath)
            for page in loader.lazy_load():
                self.documents.append(page)

        else:
            # Load as chunks of parsed text
            loader = UnstructuredLoader(
                file_path=filepath,
                strategy="hi_res"
            )
            
            unfiltered = []
            for chunk in loader.lazy_load():
                unfiltered.append(chunk)

            filter_keys = ['category', 'page_number']

            for chunk in unfiltered:
                mdata = {k: v for k,v in chunk.metadata.items() if k in filter_keys}
                fdoc = Document(page_content = chunk.page_content, metadata = mdata)
                self.documents.append(fdoc)

        self.document_ids = self.database.add_documents(documents=self.documents)


    def clear(self):
        """
        Clear the document(s) from the vector database.

        Returns:
            None
        """
        self.database.adelete(self.document_ids)


    def retrieve(self, query: str, k : int = 1):
        """
        Query the vector database with the given query, retrieve the 
        documents and return them.

        Args:
            query (str): Query string.

            k (int): Number of documents to return. Defaults to 1.

        Returns:
            List[Document]: List of documents matching the query.
        """
        retrieved = self.database.similarity_search(query = query, k = k)
        return retrieved


