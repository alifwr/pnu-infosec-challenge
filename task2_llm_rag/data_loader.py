from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
cve_cwe_data = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025")
personal_data = load_dataset("nvidia/Nemotron-Personas-USA")


def load_cve_cwe_data():
    return cve_cwe_data


def load_personal_data():
    return personal_data
