from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from lxml import etree
from langchain_text_splitters import RecursiveCharacterTextSplitter
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



class XmlDocument:
    """
    Base class used for managing text from xml files.
    """
    def __init__(self, doi: str = None):
        """
        Attrs:
            doi (str): DOI of the document, if available.
            title (str): Title of the document, if available.
            title_abstract (str): Title and abstract of the document, if available.
            pages (List[str]): List of pages loaded from the xml document, where each page
                is a string containing the text content of the document.
        """
        self.doi = doi
        self.title = ""
        self.title_abstract = ""
        self.pages = [""]


    def parse(self, filepath: str):
        """
        Parses the XML document from the given path and extracts text from its 
        title, abstract, body, and back sections.

        Args:
            filepath (str): Path to the document.
        
        Returns:
            full_text (str): Parsed text content of the document.
        """
        try:
            tree = etree.parse(filepath)
            root = tree.getroot()
            ns = {
                'tei': 'http://www.tei-c.org/ns/1.0',
                'xlink': 'http://www.w3.org/1999/xlink',
                'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            }

            title = root.find('.//tei:title', ns)
            abstract = root.find('.//tei:abstract', ns)
            body = root.find(".//tei:body", namespaces=ns)
            back = root.find(".//tei:back", namespaces=ns)

            title_text = title.text.strip() if (title is not None and title.text is not None) else ""
            self.title = title_text
            title_text = "# " + title_text + "\n"
            abstract_text = ''.join(abstract.itertext()).strip() if (abstract is not None and abstract.itertext() is not None) else "\n"
            abstract_text = "## Abstract\n" + abstract_text + "\n\n"
            self.title_abstract = title_text + abstract_text

            body_text = ""
            if body is not None:
                for div in body.findall(".//tei:div", namespaces=ns):
                    # Header:
                    head = div.find("tei:head", namespaces=ns)
                    header_text = head.text.strip() if head is not None else "Untitled Section"
                    header_num = head.get("n", "") if head is not None else ""
                    count = header_num.count(".")
                    if count == 0:
                        header_style = "##"
                    else:
                        header_style = "#" * (count + 1)
                    body_text += f"{header_style} {header_text}\n"

                    # Get all paragraphs in the section
                    paragraphs = div.findall("tei:p", namespaces=ns)
                    for p in paragraphs:
                        paragraph_text = "".join(p.itertext()).strip() if (p is not None and p.itertext() is not None) else ""
                        body_text += paragraph_text + "\n"

                    if len(paragraphs) > 0:
                        body_text += "\n"  # Add a newline after each section

                for fig in body.findall(".//tei:figure", namespaces=ns):
                    # Get header for the figure
                    head = fig.find("tei:head", namespaces=ns)
                    header_text = head.text.strip() if (head is not None and head.text is not None) else "Untitled Figure"
                    body_text += f"### {header_text}\n"

                    # Get caption for the figure
                    caption = fig.find("tei:figDesc", namespaces=ns)
                    caption_text = caption.text.strip() if (caption is not None and caption.text is not None) else "No caption"
                    body_text += f"**Caption:** {caption_text}\n"

                    body_text += "\n"  # Add a newline after each section

            back_text = ""
            if back is not None:
                for div in back.findall(".//tei:div", namespaces=ns):
                    head = div.find("tei:head", namespaces=ns)
                    if head is not None:
                        header_text = head.text.strip() if head.text is not None else "Untitled Section"
                        header_num = head.get("n", "")
                        count = header_num.count(".")
                        if count == 0:
                            header_style = "##"
                        else:
                            header_style = "#" * (count + 1)
                        back_text += f"{header_style} {header_text}\n"

                        # Get all paragraphs in the section
                        paragraphs = div.findall("tei:p", namespaces=ns)
                        for p in paragraphs:
                            paragraph_text = "".join(p.itertext()).strip()
                            back_text += paragraph_text + "\n"

                        back_text += "\n"  # Add a newline after each section

            full_text = title_text + abstract_text + body_text + back_text
            return full_text
        
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Error while parsing {filepath}: {e}")
        

    def split(self, text, token_size):
        """
        Splits the content into smaller chunks based on the specified token size.

        Args:
            text (str): The text content to be split into chunks of tokens.
            token_size (int): The maximum number of tokens per chunk.
        Returns:
            pages (List[str]): List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=token_size,
            chunk_overlap=0,
            separators = ["\n\n"]
        )
        pages = text_splitter.split_text(text)
        return pages


    def load(self, filepath: str, token_size: int = 4096):
        """
        Loads the XML document from the given path and extracts text from its 
        title, abstract, body, and back sections.

        Args:
            filepath (str): Path to the document.

        """
        full_text = self.parse(filepath)
        self.pages = self.split(full_text, token_size)
