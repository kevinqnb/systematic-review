from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List

class PdfDocument:
    """
    Base class used for managing text from pdf files.
    """
    def __init__(self):
        """
        Args:
            chunks (bool): Whether to split the document into smaller chunks of text or not.
                Defaults to False, in which case the pdf document is loaded as a collection 
                of pages.
        Attrs:
            pages (List[Document]): List of pages loaded from the pdf document.
        """
        self.pages = None
        self.chunks = None


    def load(self, filepath: str):
        """
        Load the document from the given path.

        Args:
            filepath (str): Path to the document.

        """
        # Load as individual pages:
        self.pages = []
        loader = PyPDFLoader(filepath)
        for page in loader.lazy_load():
            self.pages.append(page)

        # Load as chunks of parsed text
        self.chunks = []
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
            self.chunks.append(fdoc)


    def trim(self, headings : List[str]):
        """
        Trim the document to only include pages up to and including the first appearance 
        from a set of given headings. For example, if heading is ["References", "Works Cited"] 
        and the document includes a references section labeled with either of these heading 
        values on page 9, then the document will be trimmed to only include the first 9 pages. 

        Args:
            heading (List[str]): List of headings to trim the document by.
        """
        heading_page = -1
        for i, chunk in enumerate(self.chunks):
            if chunk.metadata['category'] == 'Title':
                title = chunk.page_content.lower()
                if any(k in title for k in headings):
                    heading_page = int(chunk.metadata['page_number'])
                    break

        self.pages = self.pages[:heading_page]
        self.chunks = self.chunks[:i]


