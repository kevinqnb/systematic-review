from semanticscholar import SemanticScholar
import requests
from dotenv import load_dotenv
import os
import requests
import time
load_dotenv()

n_papers = 12000
search_parameters = {
    "query" : "",
    "bulk" : True,
    "sort" : "publicationDate:desc",
    "venue" : ["Limnology and Oceanography"]
}
sch = SemanticScholar()

# Search and get list of up to 10,000,000 DOIs
# https://semanticscholar.readthedocs.io/en/stable/mainclasses/semanticscholar.html#semanticscholar.SemanticScholar.SemanticScholar.search_paper

response = sch.search_paper(**search_parameters)
paper_info = {"doi": [], "title": [], "authors": [], "year": []}
dois = []
for paper in response:
    doi = paper['externalIds'].get('DOI')
    if doi is not None:
        dois.append(doi)
        paper_info["doi"].append(doi)

    paper_info["title"].append(paper['title'])
    paper_info["authors"].append([author['name'] for author in paper['authors']])
    paper_info["year"].append(paper['year'])


# Downloading with Wiley TDM API:
base_url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
api_key = os.getenv("WILEY_API_KEY")
headers = {
    "Wiley-TDM-Client-Token": api_key
}

save_directory = os.getenv("PDF_PATH")

save_count = 0
idx = 0
while save_count < n_papers:
    # Construct the URL for the specific DOI and make the request
    doi = dois[idx]
    url = f"{base_url}{doi}"
    response = requests.get(url, headers=headers, allow_redirects=True)

    # Save PDF
    if response.status_code == 200:
        fname = save_directory + doi.replace('/', '_') + '.pdf'
        with open(fname, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded PDF for DOI: {doi} to {fname}")
        save_count += 1
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
        print("Faied DOI:", doi)
        print(response.text)

    idx += 1
    time.sleep(10)
