import os
from extraction.pond_model import MODEL, GRAPH
from systematic_review import *

chat_with_history = ChatWithHistory(llm = GRAPH)
token_size = 1000

directory = "collection/processed/"
papers = get_filenames_in_directory(directory)

# Load and process documents as chunks with specified token size:
for paper in papers:
    try:
        file_path = os.path.join(directory, paper)
        doi = paper.partition(".grobid")[0]
        doi = doi.replace("_", "/")
        doc = XmlDocument(doi = doi)
        doc.load(file_path, token_size = token_size)

        # Screen abstract:
        response = chat_with_history.invoke(
            {'abstract' : doc.title_abstract},
            identifier = {'doi' : doc.doi, 'chunk' : -1}, # -1 indicates abstract
            ignore = ['abstract','text']
        )

        if response["abstract_bool"]:
            # Screen text:
            for i,page in enumerate(doc.pages):
                #print(f"Processing {doc.title[:25]}... Page {i+1}/{len(doc.pages)}")
                response = chat_with_history.invoke(
                    {'text': page, 'abstract_bool': True},
                    identifier = {'doi' : doc.doi, 'chunk' : i},
                    ignore = ['abstract', 'text']
                )
    except:
        print(f"Error processing {paper}. Skipping to next paper.")
        continue

outfile = "extraction/data/coastal/screening.csv"
chat_with_history.save(outfile)