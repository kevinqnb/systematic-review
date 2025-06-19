from grobid_client.grobid_client import GrobidClient
import os 
from dotenv import load_dotenv
load_dotenv()

inpath = os.getenv("PDF_PATH")
outpath = os.getenv("PROCESSED_PATH")
client = GrobidClient(config_path="config.json")
client.process(
    service="processFulltextDocument",
    input_path=inpath,
    output=outpath,
    n=3
)
