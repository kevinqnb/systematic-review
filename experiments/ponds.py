from model import LLM, GRAPH
from systematic_review import *

chat = ChatWithHistory(llm = GRAPH)
token_size = 1000

# Ideally you'll want to do this so that the files are taken generally from a directory, 
# and the titles are parsed with the XML loader.
papers = {
    'Pond 1: Specificity of zooplankton distribution in meteorite craterponds (Morasko, Poland)' :
    'papers/processed/ponds1.grobid.tei.xml',
    'Pond 2: Environmental Conditions and Macrophytes of Karst Ponds' :
    'papers/processed/ponds2.grobid.tei.xml',
    'Pond 3: Drivers of carbon dioxide and methane supersaturation in small, temporary ponds' :
    'papers/processed/ponds3.grobid.tei.xml',
    'Lake 1: Lake metabolism scales with lake morphometry and catchment conditions' :
    'papers/processed/lakes1.grobid.tei.xml',
    'Lake 2: Net Heterotrophy in Small Danish Lakes: A Widespread Feature  Over Gradients in Trophic Status and Land Cover' :
    'papers/processed/lakes2.grobid.tei.xml',
    'Lake 3: Patterns in the Species Composition and Richness of Fish Assemblages in Northern Wisconsin Lakes' :
    'papers/processed/lakes3.grobid.tei.xml',
    'Definition 1: Eutrophication: are mayflies (Ephemeroptera) good bioindicators for ponds?' :
    'papers/processed/definitions1.grobid.tei.xml',
    'Definition 2: The importance of small waterbodies for biodiversity and ecosystem services: implications for policy makers' :
    'papers/processed/definitions2.grobid.tei.xml',
    'Definition 3: Agricultural Freshwater Pond Supports Diverse and Dynamic Bacterial and Viral Populations' :
    'papers/processed/definitions3.grobid.tei.xml',
    'Fake 1: Quantifying saltmarsh vegetation and its effect on wave height dissipation: Results from a UK East coast saltmarsh' :
    'papers/processed/fake1.grobid.tei.xml',
    'Fake 2: Methane and Carbon Dioxide Fluxes in a Temperate Tidal Salt Marsh: Comparisons Between Plot and Ecosystem Measurements' :
    'papers/processed/fake2.grobid.tei.xml',
    'Fake 3: Improving the definition of a coastal habitat: Putting the salt back into saltmarsh' :
    'papers/processed/fake3.grobid.tei.xml',
}

# Load and process documents as chunks with specified token size:
for title, file_path in papers.items():
    doc = XmlDocument()
    doc.load(file_path, token_size = token_size)

    for i,page in enumerate(doc.pages):
        print(f"Processing {title[:25]}... Page {i+1}/{len(doc.pages)}")
        chat.invoke({'context' : page}, identifier = title, ignore = ['context'])

outfile = "experiments/data/pond_screening4.csv"
chat.save(outfile)
