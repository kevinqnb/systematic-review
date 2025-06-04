from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="config.json")
client.process(
    service="processFulltextDocument",
    input_path="collection/pdfs",
    output="collection/processed",
    n=1
)