from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
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
    repo_id="google/gemma-3-27b-it-qat-q4_0-gguf",
    filename="gemma-3-27b-it-q4_0.gguf",
    cache_dir = '../huggingface/hub'
)

llm = LlamaCpp(
    model_path=model_path,
    chat_format = "gemma",
    temperature=0,
    max_tokens=1000,
    n_ctx=10000
)

prompt_template = PromptTemplate.from_template(
    "<start_of_turn>user\n{instructions}<end_of_turn>\n"
    "<start_of_turn>user\n{context}<end_of_turn>\n"
    "<start_of_turn>user\n{query}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Load documents as individual pages:
paper_pages = {}
for title, file_path in papers.items():
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    paper_pages[title] = pages


instructions = (
        "You will be given contextual information from a page of a scientific research paper "
        "and asked to accurately answer questions about its contents. Please answer only "
        "for the information shown on the current page, and not the paper as a whole."
        "If applicable, point to a specific location or excerpt "
        "from the text to justify your reasoning."
)

context = paper_pages[titles[0]][0].page_content

query1 = (
    "Does this page include any numerical data related to physical or chemical attributes of individual ponds? "
    "Examples may include quantitative measurements of depth, surface area, temperature, or pH. "
)

query2 = (
    "Does this page contain a scientific definition for classifying either ponds or lakes?"
)

prompt = prompt_template.invoke({
    "instructions": instructions,
    "context": context,
    "query": query1
})

answer = llm.invoke(prompt)
print(answer)