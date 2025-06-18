from semanticscholar import SemanticScholar
import requests
from dotenv import load_dotenv
import os
import requests
import time
load_dotenv()

n_papers = 1200
search_parameters = {
    "query" : "",
    "bulk" : True,
    "sort" : "publicationDate:desc",
    "venue" : ["Limnology and Oceanography"]
}
sch = SemanticScholar()

# Search and get list of DOIs
response = sch.search_paper(**search_parameters)
dois = []
for paper in response:
    doi = paper['externalIds'].get('DOI')
    if doi is not None:
        dois.append(doi)


# Downloading with Wiley TDM API:
base_url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
api_key = os.getenv("WILEY_API_KEY")
headers = {
    "Wiley-TDM-Client-Token": api_key
}

save_directory = os.getenv("PDF_PATH")
for doi in dois:
    # Construct the URL for the specific DOI and make the request
    url = f"{base_url}{doi}"
    response = requests.get(url, headers=headers, allow_redirects=True)

    # Save PDF
    if response.status_code == 200:
        fname = save_directory + doi.replace('/', '_') + '.pdf'
        with open(fname, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded PDF for DOI: {doi} to {fname}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
        print("Faied DOI:", doi)
        print(response.text)

    time.sleep(10)
