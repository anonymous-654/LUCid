import os
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI
from together import Together

load_dotenv()

azure_openai_endpoint = os.getenv("AZUREOPENAI_ENDPOINT")
deployment = "gpt-4.1"
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")
api_version = "2024-12-01-preview"


gpt_client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=api_version,
)

client = Together(api_key=together_api_key)


def query_llm(sys_prompt: str = None, user_prompt: str = None):
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0,
        seed=42,
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def query_gpt(sys_prompt: str = None, user_prompt: str = None):
    messages = []
    if sys_prompt is not None:
        messages.append({"role": "system", "content": sys_prompt})
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})
    response = gpt_client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0,
        seed=42,
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def query_llm_with_history(user_prompt: str, history: List = []):
    messages = []
    for turn in history:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        elif turn["role"] == "assistant":
            messages.append({"role": "assistant", "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0,
        seed=42,
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )
