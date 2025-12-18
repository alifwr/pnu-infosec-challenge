from datasets import load_dataset, logging

logging.set_verbosity(logging.INFO)

# Login using e.g. `huggingface-cli login` to access this dataset
cve_cwe_data = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025")
personal_data = load_dataset("nvidia/Nemotron-Personas-USA")
# Select 'train' split and take first 100 rows
sample_cve_cwe = cve_cwe_data["train"].select(range(100))
sample_personal = personal_data["train"].select(range(100))


def load_cve_cwe_data():
    return sample_cve_cwe


def load_personal_data():
    return sample_personal
