import json
import requests
import os
from typing import List, Dict, Any

# Configuration
TEST_QUERIES_FILE = "test_queries.json"
API_URL = "http://localhost:8000/rag-query"


def load_queries(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def query_rag(query_text: str) -> Dict[str, Any]:
    try:
        response = requests.post(API_URL, json={"query": query_text}, timeout=30)
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
            "status_code": response.status_code,
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def run_tests():
    print(f"Loading queries from {TEST_QUERIES_FILE}...")
    categories = load_queries(TEST_QUERIES_FILE)

    if not categories:
        return

    print(f"Testing RAG endpoint at {API_URL}...\n")

    report = {}

    for category_data in categories:
        category = category_data.get("category", "Unknown")
        queries = category_data.get("queries", [])

        print(f"--- Category: {category} ---")
        category_results = []

        for q in queries:
            print(f"  Querying: '{q}' ... ", end="", flush=True)
            result = query_rag(q)

            if result["success"]:
                retrieved_docs = result["data"]
                num_docs = len(retrieved_docs)
                print(f"✅ Success ({num_docs} docs retrieved)")
                category_results.append(
                    {
                        "query": q,
                        "status": "Success",
                        "retrieved_count": num_docs,
                        "preview": str(retrieved_docs[0]["content"])[:50] + "..."
                        if num_docs > 0
                        else "N/A",
                        "full_result": retrieved_docs,
                    }
                )
            else:
                print(f"❌ Failed ({result.get('error')})")
                category_results.append(
                    {"query": q, "status": "Failed", "error": result.get("error")}
                )

        report[category] = category_results
        print("")

    print("\n=== TEST REPORT ===")

    md_content = "# RAG Endpoint Test Report\n\n"
    md_content += f"**API URL:** {API_URL}\n\n"

    for category, results in report.items():
        print(f"\nCategory: {category}")
        success_count = sum(1 for r in results if r["status"] == "Success")
        total = len(results)
        print(f"  Success Rate: {success_count}/{total}")

        md_content += f"## Category: {category}\n\n"
        md_content += f"**Success Rate:** {success_count}/{total}\n\n"
        md_content += "| Query | Status | Docs | Preview | Full Result |\n"
        md_content += "|---|---|---|---|---|\n"

        print(f"  {'Query':<40} | {'Status':<10} | {'Docs':<5} | {'Preview'}")
        print(f"  {'-' * 40}-|-{'-' * 10}-|-{'-' * 5}-|-{'-' * 20}")
        for r in results:
            q_text = (r["query"][:37] + "...") if len(r["query"]) > 37 else r["query"]
            status = r["status"]
            docs = r.get("retrieved_count", "-")
            preview = r.get("preview", "").replace("\n", " ")
            print(f"  {q_text:<40} | {status:<10} | {docs:<5} | {preview}")

            # Add to MD
            md_q = r["query"].replace("|", "\\|")
            md_preview = r.get("preview", "").replace("\n", " ").replace("|", "\\|")

            full_res = r.get("full_result", [])
            formatted_res = []
            for i, doc in enumerate(full_res):
                c = doc.get("content", "").replace("\n", "<br>")
                m = str(doc.get("metadata", "")).replace("\n", " ")
                formatted_res.append(f"**{i + 1}.** Content: {c}<br>Metadata: {m}")

            md_full_res = "<br><br>".join(formatted_res).replace("|", "\\|")

            md_content += (
                f"| {md_q} | {status} | {docs} | {md_preview} | {md_full_res} |\n"
            )

        md_content += "\n"

    with open("rag_test_report.md", "w") as f:
        f.write(md_content)
    print("\nReport saved to rag_test_report.md")


if __name__ == "__main__":
    run_tests()
