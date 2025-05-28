from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from typing_extensions import List, TypedDict
import pandas as pd
from systematic_review import *

# Papers
papers = {
    'Pond 1: Specificity of zooplankton distribution in meteorite craterponds (Morasko, Poland)' :
    'papers/examples/ponds1.pdf',
    'Pond 2: Environmental Conditions and Macrophytes of Karst Ponds' :
    'papers/examples/ponds2.pdf',
    'Pond 3: Drivers of carbon dioxide and methane supersaturation in small, temporary ponds' :
    'papers/examples/ponds3.pdf',
    'Lake 1: Lake metabolism scales with lake morphometry and catchment conditions' :
    'papers/examples/lakes1.pdf',
    'Lake 2: Net Heterotrophy in Small Danish Lakes: A Widespread Feature  Over Gradients in Trophic Status and Land Cover' :
    'papers/examples/lakes2.pdf',
    'Lake 3: Patterns in the Species Composition and Richness of Fish Assemblages in Northern Wisconsin Lakes' :
    'papers/examples/lakes3.pdf',
    'Definition 1: Eutrophication: are mayflies (Ephemeroptera) good bioindicators for ponds?' :
    'papers/examples/definitions1.pdf',
    'Definition 2: The importance of small waterbodies for biodiversity and ecosystem services: implications for policy makers' :
    'papers/examples/definitions2.pdf',
    'Definition 3: Agricultural Freshwater Pond Supports Diverse and Dynamic Bacterial and Viral Populations' :
    'papers/examples/definitions3.pdf',
    'Fake 1: Quantifying saltmarsh vegetation and its effect on wave height dissipation: Results from a UK East coast saltmarsh' :
    'papers/examples/fake1.pdf',
    'Fake 2: Methane and Carbon Dioxide Fluxes in a Temperate Tidal Salt Marsh: Comparisons Between Plot and Ecosystem Measurements' :
    'papers/examples/fake2.pdf',
    'Fake 3: Improving the definition of a coastal habitat: Putting the salt back into saltmarsh' :
    'papers/examples/fake3.pdf',
}
titles = list(papers.keys())

# Load the language model
model_path = hf_hub_download(
    repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
    filename="gemma-3-12b-it-q4_0.gguf",
    #cache_dir = '../huggingface/hub'
)

llm = ChatLlamaCpp(
    model_path=model_path,
    chat_format = "gemma",
    temperature=0,
    max_tokens=1000,
    n_ctx=2000,
    n_gpu_layers = 10,
    n_batch = 500,
    f16_kv=True,  # MUST set to True, on Mac
)

instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole."
        "If applicable, point to a specific location or excerpt "
        "from the text to justify your reasoning."
)

# Specify prompt and response formats
prompt_template = PromptTemplate.from_template(
    "<start_of_turn>user\n{instructions}<end_of_turn>\n"
    "<start_of_turn>user\n{context}<end_of_turn>\n"
    "<start_of_turn>user\n{query}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

class StructuredResponse(BaseModel):
    """
    Manages a structured response from a language model.
    """
    response : bool = Field(
        description = 
            "False if the answer is no or unknown. "
            "True only if the answer is yes. "
    )
    
    reason : str = Field(
        description = 
            "Specify location or excerpt "
            "from the text to justify your reasoning. "
    )

dict_schema = convert_to_openai_tool(StructuredResponse)
structured_llm = llm.with_structured_output(dict_schema)


# Build the conditional graph:
class State(TypedDict):
    title : str
    page_num : int
    context : Document
    definition_screen : bool
    definition_reason : str
    definition : str
    table_screen : bool
    table_reason : str


def screen_for_definition(state : State):
    query = "Does this page contain a scientific definition for classifying either ponds or lakes?"
    messages = prompt_template.invoke(
        {"instructions": instructions, "context": state['context'], "query": query}
    )
    answer = structured_llm.invoke(messages)
    return {"definition_screen": answer['response'], "definition_reason": answer['reason']}


def extract_definition(state : State):
    query = "What is the definition for classifying either ponds or lakes given by the context?"
    messages = prompt_template.invoke(
        {"instructions": instructions, "context": state['context'], "query": query}
    )
    answer = llm.invoke(messages)
    return {"definition": answer.content}


def extract_routing(state : State):
    return state['definition_screen']


def screen_for_table(state : State):
    query = (
        "Does this page contain a table reporting data on "
        "physical, chemical, or biological attributes of individual ponds or lakes?"
    )
    messages = prompt_template.invoke(
        {"instructions": instructions, "context": state['context'], "query": query}
    )
    answer = structured_llm.invoke(messages)
    return {"table_screen": answer['response'], "table_reason": answer['reason']}


graph_builder = StateGraph(State)
graph_builder.add_node("screen_for_definition", screen_for_definition)
graph_builder.add_node("screen_for_table", screen_for_table)
graph_builder.add_node("extract_definition", extract_definition)
graph_builder.add_edge(START, "screen_for_definition")
graph_builder.add_conditional_edges(
    "screen_for_definition",
    extract_routing,
    {True : "extract_definition", False: "screen_for_table"}
)
graph_builder.add_edge("extract_definition", "screen_for_table")
graph_builder.add_edge("screen_for_table", END)
graph = graph_builder.compile()


# Load and process documents as individual pages:
responses  = []
for title, file_path in papers.items():
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    # Load as chunks of parsed text
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res"
    )
    chunks = []
    for doc in loader.lazy_load():
        chunks.append(doc)

    # Filter out pages with references
    references_page = -1
    ref_keywords = ["references", "bibliography", "works cited"]
    for chunk in chunks:
        if chunk.metadata['category'] == 'Title':
            title = chunk.page_content.lower()
            if any(k in title for k in ref_keywords):
                references_page = int(chunk.metadata['page_number'])
                break

    pages = pages[:references_page]

    # Run LLM process on each page:
    for i,page in enumerate(pages):
        response = graph.invoke(
            {
                'title' : title,
                'page_num' : i,
                'context' : page.page_content
            }
        )
        del response['context']
        responses.append(response)


responses = pd.DataFrame(responses)
responses.to_csv("experiments/data/pond_screening2.csv")