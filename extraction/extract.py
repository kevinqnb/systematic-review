import os
from dotenv import load_dotenv
from model import MODEL, GRAPH
from systematic_review import *
load_dotenv()

chat_with_history = ChatWithHistory(llm = GRAPH)
token_size = 512

directory = os.getenv("PROCESSED_PATH")
papers = get_filenames_in_directory(directory)

# Load and process documents as chunks with specified token size:
for paper in papers[:100]:
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
            ignore = ['abstract']
        )

        if response["abstract_bool"]:
            # Screen text:
            for i,page in enumerate(doc.pages):
                #print(f"Processing {doc.title[:25]}... Page {i+1}/{len(doc.pages)}")
                response = chat_with_history.invoke(
                    {'text': page, 'abstract_bool': True},
                    identifier = {'doi' : doc.doi, 'chunk' : i},
                    ignore = ['abstract']
                )
    except:
        print(f"Error processing {paper}. Skipping to next paper.")
        continue

outfile = "extraction/data/coastal/screening_100_4.csv"
chat_with_history.save(outfile)
