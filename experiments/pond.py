from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_ollama import ChatOllama
from systematic_review import *

# Load the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': False}
)
docdb = DocumentDB(
    embeddings = embeddings,
    chunks = False,
)

# Load the language model
llm = ChatOllama(
    model="gemma3:12b-it-qat",
    temperature=0
)
chatextract = Chat(llm=llm, n_documents = 1)

# Load and embed documents
file_path = 'papers/examples/meteorite.pdf'
docdb.load(file_path)
chatextract.fit(docdb)

# Query and record 
retreival_query = (
    "Data collected and reported for physical or chemical attributes of individually stuided ponds. "
    "Examples include but are not limited to depth, surface area, temperature, or pH."
)
prompt = (
    "Does the paper report data related to physical or chemical attributes of individual ponds? "
    "Examples include but are not limited to depth, surface area, temperature, or pH. "
    "Give reasoning and textual evidence to support your conclusion."
)
output = chatextract.screen(retreival_query, prompt)