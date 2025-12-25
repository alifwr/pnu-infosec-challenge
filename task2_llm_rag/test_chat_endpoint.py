import json
import requests
import os
from typing import List, Dict, Any

# Configuration
TEST_QUERIES_FILE = "test_queries.json"
API_URL = "http://localhost:8000/chat"
REPORT_FILE = "llm_test_report.md"


def load_queries(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def query_chat(query_text: str) -> Dict[str, Any]:
    try:
        payload = {"messages": [{"role": "user", "content": query_text}]}
        response = requests.post(
            API_URL, json=payload, timeout=60
        )  # Increased timeout for LLM generation
        response.raise_for_status()
        data = response.json()

        # Extract the last assistant message
        messages = data.get("messages", [])
        last_message = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_message = msg
                break

        if last_message:
            return {
                "success": True,
                "data": last_message.get("content", ""),
                "status_code": response.status_code,
            }
        else:
            return {
                "success": False,
                "error": "No assistant message found in response",
                "raw_response": data,
            }

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def run_tests():
    print(f"Loading queries from {TEST_QUERIES_FILE}...")
    categories = load_queries(TEST_QUERIES_FILE)

    if not categories:
        return

    print(f"Testing Chat endpoint at {API_URL}...\n")

    report = {}

    for category_data in categories:
        category = category_data.get("category", "Unknown")
        queries = category_data.get("queries", [])

        print(f"--- Category: {category} ---")
        category_results = []

        for q in queries:
            print(f"  Querying: '{q}' ... ", end="", flush=True)
            result = query_chat(q)

            if result["success"]:
                response_text = result["data"]
                print("✅ Success")
                category_results.append(
                    {
                        "query": q,
                        "status": "Success",
                        "response": response_text,
                        "preview": response_text[:50].replace("\n", " ") + "..."
                        if response_text
                        else "Empty Response",
                    }
                )
            else:
                print(f"❌ Failed ({result.get('error')})")
                category_results.append(
                    {
                        "query": q,
                        "status": "Failed",
                        "error": result.get("error"),
                        "response": result.get("error"),
                    }
                )

        report[category] = category_results
        print("")

    print("\n=== TEST REPORT ===")

    md_content = "# LLM Chat Endpoint Test Report\n\n"
    md_content += f"**API URL:** {API_URL}\n\n"

    for category, results in report.items():
        print(f"\nCategory: {category}")
        success_count = sum(1 for r in results if r["status"] == "Success")
        total = len(results)
        print(f"  Success Rate: {success_count}/{total}")

        md_content += f"## Category: {category}\n\n"
        md_content += f"**Success Rate:** {success_count}/{total}\n\n"
        md_content += "| Query | Status | Response Preview | Full Response |\n"
        md_content += "|---|---|---|---|\n"

        print(f"  {'Query':<40} | {'Status':<10} | {'Preview'}")
        print(f"  {'-' * 40}-|-{'-' * 10}-|-{'-' * 20}")
        for r in results:
            q_text = (r["query"][:37] + "...") if len(r["query"]) > 37 else r["query"]
            status = r["status"]
            preview = r.get("preview", "").replace(
                "|", "\\|"
            )  # Escape pipe for markdown table

            print(f"  {q_text:<40} | {status:<10} | {preview}")

            # Add to MD
            md_q = r["query"].replace("|", "\\|")
            full_resp = r.get("response", "").replace("\n", "<br>").replace("|", "\\|")

            md_content += f"| {md_q} | {status} | {preview} | {full_resp} |\n"

        md_content += "\n"

    with open(REPORT_FILE, "w") as f:
        f.write(md_content)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    run_tests()
