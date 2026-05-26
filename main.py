from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
# openai_api_base = "http://10.255.58.125:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
    )
def main():
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",  # e.g., "llama-3"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain photosynthesis."}
        ],
        temperature=0.7,
    )

    print(response.choices[0].message.content,)



if __name__ == "__main__":
    main()