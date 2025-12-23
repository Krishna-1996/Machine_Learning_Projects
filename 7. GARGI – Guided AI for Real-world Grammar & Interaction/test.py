import requests

# Your local LanguageTool URL
LANGUAGETOOL_URL = "http://localhost:8081/v2/check"

# The text to check
data = {
    "text": "He go to school every day.",
    "language": "en-US"
}

try:
    response = requests.post(LANGUAGETOOL_URL, data=data)
    response.raise_for_status()  # Raise error if something went wrong
    result = response.json()
    
    # Print grammar matches
    matches = result.get("matches", [])
    if matches:
        print(f"Found {len(matches)} grammar issue(s):")
        for match in matches:
            print(f"- {match['message']}")
            print(f"  Suggestions: {[r['value'] for r in match.get('replacements', [])]}")
    else:
        print("No grammar issues found.")
except requests.exceptions.RequestException as e:
    print(f"Error connecting to LanguageTool: {e}")
