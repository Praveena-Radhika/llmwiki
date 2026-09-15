"""
check_deploy.py
Finds your Azure OpenAI deployment name by listing deployments on your resource.
Fill in ENDPOINT and KEY below, then run:
    .venv\\Scripts\\python.exe check_deploy.py
"""

import requests

ENDPOINT = "https://<your-resource-name>.openai.azure.com"  # <-- paste your endpoint here
KEY = "<your-key>"                                            # <-- paste your key here

api_versions = ["2024-08-01-preview", "2024-02-01", "2023-05-15"]

for version in api_versions:
    url = f"{ENDPOINT.rstrip('/')}/openai/deployments?api-version={version}"
    try:
        r = requests.get(url, headers={"api-key": KEY}, timeout=15)
        print(f"\n--- api-version={version} ---")
        print(f"Status: {r.status_code}")
        print(r.text[:2000])
        if r.status_code == 200:
            print("\n>>> SUCCESS with this api-version. Deployment name(s) are in the 'id' field above. <<<")
            break
    except requests.exceptions.RequestException as e:
        print(f"\n--- api-version={version} ---")
        print(f"Request failed: {e}")

print("\nIf every attempt above 404'd, your ENDPOINT format is likely wrong.")
print("Copy the EXACT endpoint string shown on your resource's 'Keys and Endpoint' page")
print("(it may be *.services.ai.azure.com or *.cognitiveservices.azure.com instead of *.openai.azure.com).")
