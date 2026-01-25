from openai import OpenAI

# Endpoint LM Studio (dietro VPN UNITN)
BASE_URL = "http://bears.disi.unitn.it:1234/v1"

# LM Studio ignora la key, ma l'SDK la richiede
API_KEY = "lmstudio"

# Modello consigliato
MODEL = "meta-llama-3.1-8b-instruct"

def main():
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    print("Sending test request to LM Studio...")

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
