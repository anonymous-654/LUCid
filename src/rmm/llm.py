from openai import OpenAI


class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY"):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages, max_tokens: int = 512, temperature: float = 0.7) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},}, 
        )

        msg = completion.choices[0].message
        # print(completion)

        if msg.content is None:
            raise ValueError(
                f"Model returned no final content. "
                f"finish_reason={completion.choices[0].finish_reason}"
            )
        
        content = msg.content.strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # print(content)

        return content