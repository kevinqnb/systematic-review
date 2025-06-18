from grobid_client.grobid_client import GrobidClient
import os 
from dotenv import load_dotenv
load_dotenv()

client = GrobidClient(config_path="config.json")
client.process(
    service="processFulltextDocument",
    input_path="collection/pdfs",
    output="collection/processed",
    n=3
)
