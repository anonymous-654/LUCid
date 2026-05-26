import os
import re
from typing import Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from together import Together

# Official SDKs
from anthropic import Anthropic
from google import genai

load_dotenv()

# ===== ENV =====
azure_openai_endpoint = os.getenv("AZUREOPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

# New
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

api_version = "2024-12-01-preview"

# ===== CLIENTS =====

# Azure OpenAI (GPT)
# gpt_client = AzureOpenAI(
#     azure_endpoint=azure_openai_endpoint,
#     api_key=azure_openai_key,
#     api_version=api_version,
# )

gpt_client = OpenAI(api_key=openai_api_key)

# Local OpenAI-compatible (SGLang / vLLM / etc.)
NODE_HOSTNAME = os.getenv("NODE_HOSTNAME")
local_client = OpenAI(
    base_url=f"http://{NODE_HOSTNAME}:8002/v1",
    api_key="EMPTY",
)

# Optional Together client
together_client = Together(api_key=together_api_key) if together_api_key else None

# Anthropic Claude
claude_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

# Google Gemini
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None


# ===== HELPERS =====

def extract_response_from_local_llm(text: str) -> str:
    """
    Removes <think>...</think> blocks and returns only the final model response.
    Useful for Qwen / reasoning models.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _provider_from_model(model_path: str) -> str:
    """
    Simple routing logic by model name.
    """
    model_lower = model_path.lower()

    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or "o4" in model_lower:
        return "openai"

    if "claude" in model_lower:
        return "claude"

    if "gemini" in model_lower:
        return "gemini"

    if "llama" in model_lower and together_client is not None:
        return "together"

    return "local"


def _extract_text_from_claude_response(response) -> str:
    """
    Claude Messages API returns content blocks.
    We concatenate text blocks only.
    """
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _build_plain_prompt(
    sys_prompt: Optional[str],
    user_prompt: Optional[str],
) -> str:
    """
    Gemini generate_content is simplest when given a plain text prompt.
    """
    chunks = []
    if sys_prompt:
        chunks.append(f"System instruction:\n{sys_prompt}")
    if user_prompt:
        chunks.append(f"User:\n{user_prompt}")
    return "\n\n".join(chunks).strip()


# ===== MAIN API =====

def query_llm(
    model_path: str,
    sys_prompt: Optional[str] = None,
    user_prompt: str = "",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    print_tokens: bool = True,
) -> Tuple[str, Optional[int], Optional[int]]:

    provider = _provider_from_model(model_path)

    try:
        # ------------------------------------------------------------------
        # Azure OpenAI
        # ------------------------------------------------------------------
        if provider == "openai":
            messages = []
            if sys_prompt is not None:
                messages.append({"role": "system", "content": sys_prompt})
            if user_prompt is not None:
                messages.append({"role": "user", "content": user_prompt})

            request_kwargs = dict(
                model=model_path,
                messages=messages,
                temperature=temperature,
                seed=42,
            )
            # if max_tokens is not None:
            #     request_kwargs["max_tokens"] = max_tokens

            response = gpt_client.chat.completions.create(**request_kwargs)

            res = response.choices[0].message.content or ""

            prompt_tokens = None
            completion_tokens = None
            if getattr(response, "usage", None):
                prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                completion_tokens = getattr(response.usage, "completion_tokens", None)

            if print_tokens:
                print(
                    f"[llm_client:azure_openai] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}",
                    flush=True,
                )

            return res, prompt_tokens, completion_tokens

        # ------------------------------------------------------------------
        # Claude (Anthropic official SDK)
        # ------------------------------------------------------------------
        if provider == "claude":
            if claude_client is None:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")

            messages = []
            if user_prompt is not None:
                messages.append({"role": "user", "content": user_prompt})

            request_kwargs = dict(
                model=model_path,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 1024,  # required by Claude
            )
            if sys_prompt is not None:
                request_kwargs["system"] = sys_prompt

            response = claude_client.messages.create(**request_kwargs)
            res = _extract_text_from_claude_response(response)

            # Anthropic usage shape differs from OpenAI-style usage
            prompt_tokens = None
            completion_tokens = None
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "input_tokens", None)
                completion_tokens = getattr(usage, "output_tokens", None)

            if print_tokens:
                print(
                    f"[llm_client:claude] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}",
                    flush=True,
                )

            return res, prompt_tokens, completion_tokens

        # ------------------------------------------------------------------
        # Gemini (Google Gen AI SDK)
        # ------------------------------------------------------------------
        if provider == "gemini":
            if gemini_client is None:
                raise RuntimeError("GEMINI_API_KEY is not set")

            prompt = _build_plain_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt)

            # The Google Gen AI SDK supports generate_content for Gemini text generation.
            # response.text is the simplest text accessor.
            config = {}
            if temperature is not None:
                config["temperature"] = temperature
            if max_tokens is not None:
                # In the SDK this maps to output token limit in generation config.
                config["max_output_tokens"] = max_tokens

            response = gemini_client.models.generate_content(
                model=model_path,
                contents=prompt,
                config=config or None,
            )

            res = getattr(response, "text", "") or ""

            # Token usage availability can vary by response/model/SDK version.
            prompt_tokens = None
            completion_tokens = None
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                completion_tokens = getattr(usage, "candidates_token_count", None)

            if print_tokens:
                print(
                    f"[llm_client:gemini] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}",
                    flush=True,
                )

            return res.strip(), prompt_tokens, completion_tokens

        # ------------------------------------------------------------------
        # Together
        # ------------------------------------------------------------------
        if provider == "together":
            messages = []
            if sys_prompt is not None:
                messages.append({"role": "system", "content": sys_prompt})
            if user_prompt is not None:
                messages.append({"role": "user", "content": user_prompt})

            request_kwargs = dict(
                model=model_path,
                messages=messages,
                temperature=temperature,
            )
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens

            response = together_client.chat.completions.create(**request_kwargs)

            res = response.choices[0].message.content or ""

            prompt_tokens = None
            completion_tokens = None
            if getattr(response, "usage", None):
                prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                completion_tokens = getattr(response.usage, "completion_tokens", None)

            if print_tokens:
                print(
                    f"[llm_client:together] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}",
                    flush=True,
                )

            return res, prompt_tokens, completion_tokens

        # ------------------------------------------------------------------
        # Local OpenAI-compatible
        # ------------------------------------------------------------------
        messages = []
        if sys_prompt is not None:
            messages.append({"role": "system", "content": sys_prompt})
        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})

        request_kwargs = dict(
            model=model_path,
            messages=messages,
            temperature=temperature,
            seed=42,
        )
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

        # Qwen-specific params
        if "qwen" in model_path.lower():
            request_kwargs["extra_body"] = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        response = local_client.chat.completions.create(**request_kwargs)
        res = response.choices[0].message.content or ""

        if "qwen" in model_path.lower():
            res = extract_response_from_local_llm(res)

        prompt_tokens = None
        completion_tokens = None
        if getattr(response, "usage", None):
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)

        if print_tokens:
            print(
                f"[llm_client:local] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}",
                flush=True,
            )

        return res, prompt_tokens, completion_tokens

    except Exception as e:
        print(f"[llm_client] ERROR: {repr(e)}", flush=True)
        return "", None, None