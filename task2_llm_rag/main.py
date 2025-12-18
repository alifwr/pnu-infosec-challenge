from data_loader import load_cve_cwe_data, load_personal_data

cve_cwe_data = load_cve_cwe_data()
personal_data = load_personal_data()

print(cve_cwe_data)
print(personal_data)
