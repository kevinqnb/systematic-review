from langchain_ollama import ChatOllama
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

# Load the language model
llm = ChatOllama(
    model="gemma3:27b-it-qat",
    temperature=0
)
pdf_chat = PdfChat(llm=llm)

# Reference headings to remove
reference_headings = [
    "References",
    "Works Cited",
    "Bibliography"
]

# Load and process documents as individual pages:
for title, file_path in papers.items():
    pdf = PdfDocument()
    pdf.load(file_path)
    pdf.trim(reference_headings)

    for i,page in enumerate(pdf.pages):
        print(f"Processing {title[:25]}... Page {i+1}/{len(pdf.pages)}")
        pdf_chat.fit(page, title, i)
        pdf_chat.record()

outfile = "experiments/data/pond_screening2.csv"
pdf_chat.save(outfile)
