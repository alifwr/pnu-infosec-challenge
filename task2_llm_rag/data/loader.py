from datasets import load_dataset, logging

logging.set_verbosity(logging.INFO)

cve_cwe_data = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025")
personal_data = load_dataset("nvidia/Nemotron-Personas-USA")
cve_train = cve_cwe_data["train"]
total_rows = len(cve_train)
indices = list(range(total_rows - 200, total_rows))
sample_cve_cwe = cve_train.select(indices)
sample_personal = personal_data["train"].select(range(100))


def load_cve_cwe_data():
    return sample_cve_cwe


def load_personal_data():
    return sample_personal
