import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Endpoint LM Studio (dietro VPN UNITN)
BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://llm.bears.disi.unitn.it/v1")

# API key da .env
API_KEY = os.getenv("LMSTUDIO_API_KEY", "lmstudio")

# Modello da .env
MODEL = os.getenv("LMSTUDIO_MODEL", "meta/llama-3.3-70b")

def main():
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    print("Sending test request to LM Studio...")
    print(f"Using model: {MODEL}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Reply with a single sentence confirming the API works."}
        ],
        temperature=0.2,
        max_tokens=50
    )

    print("\n--- RESPONSE ---")
    print(response.choices[0].message.content)
    print("----------------")

if __name__ == "__main__":
    main()
