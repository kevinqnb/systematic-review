from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="config.json")
client.process(
    service="processFulltextDocument",
    input_path="../papers/pdfs",
    output="../papers/processed",
    n=1
)