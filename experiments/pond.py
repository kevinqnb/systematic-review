from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_ollama import ChatOllama
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

# Load the embeddings model
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': False}
)

# Load the language model
llm = ChatOllama(
    model="gemma3:12b-it-qat",
    temperature=0
)
chat_extract = Chat(llm=llm, n_documents = 2)

# Load and embed documents
for paper_title, file_path in papers.items():
    print(paper_title[:25] + '...')
    docdb = DocumentDB(
        embeddings = embeddings,
        chunks = False,
    )
    docdb.load(file_path)
    chat_extract.fit(docdb, paper_title)

    # Query and record 

    # Pond data:
    retrieval_query1 = (
        "Data collected and reported for physical or chemical attributes of individually stuided ponds. "
        "Examples include but are not limited to depth, surface area, temperature, or pH."
    )
    prompt1 = (
        "Does the paper report data related to physical or chemical attributes of individual ponds? "
        "Examples include but are not limited to depth, surface area, temperature, or pH. "
        "Note that ponds and lakes are distinctly different types of waterbodies. "
        "Answer the current question only for ponds. "
        "Give reasoning and textual evidence to support your conclusion."
    )
    question_name1 = "pond data"
    print("Running query 1...")
    output = chat_extract.screen(question_name1, retrieval_query1, prompt1)
    chat_extract.screen_record()

    # Lake data:
    retrieval_query2 = (
        "Data collected and reported for physical or chemical attributes of individually stuided lakes. "
        "Examples include but are not limited to depth, surface area, temperature, or pH."
    )
    prompt2 = (
        "Does the paper report data related to physical or chemical attributes of individual lakes? "
        "Examples include but are not limited to depth, surface area, temperature, or pH. "
        "Note that ponds and lakes are distinctly different types of waterbodies. "
        "Answer the current question only for lakes. "
        "Give reasoning and textual evidence to support your conclusion."
    )
    question_name2 = "lake data"
    print("Running query 2...")
    output = chat_extract.screen(question_name2, retrieval_query2, prompt2)
    chat_extract.screen_record()


    # Pond definitions:
    retrieval_query3 = (
        "Scientific definition for a pond specifying distinct physical or chemical attributes. "
    )
    prompt3 = (
        "Does the paper report a scientific definition for a pond? "
        "This definition should be specific about certain physical or chemical attributes. "
        "Examples include but are not limited to depth, surface area, temperature, or pH. "
        "Give reasoning and textual evidence to support your conclusion."
    )
    question_name3 = "pond definition"
    print("Running query 3...")
    output = chat_extract.screen(question_name3, retrieval_query3, prompt3)
    chat_extract.screen_record()
    print()




chat_extract.screen_save("experiments/data/pond_screening.csv")