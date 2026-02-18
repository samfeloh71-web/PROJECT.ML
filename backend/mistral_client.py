"""
Mistral AI Integration for Market Lens
Using Hugging Face InferenceClient (FREE)

This version uses chat_completion (conversational task) which is
supported by HuggingFace free tier models.
"""

import os
from typing import Dict, List

try:
    from huggingface_hub import InferenceClient
except ImportError:
    raise ImportError(
        "Please install huggingface_hub: pip install --upgrade huggingface_hub"
    )


class MistralClient:
    """Free AI client using Hugging Face InferenceClient with fallback models"""

    # Models that support the 'conversational' / chat_completion task on HF free tier
    FALLBACK_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",       # Latest Mistral instruct
        "mistralai/Mistral-7B-Instruct-v0.2",       # Previous Mistral instruct
        "HuggingFaceH4/zephyr-7b-beta",             # Zephyr chat model
        "microsoft/Phi-3-mini-4k-instruct",          # Phi-3 mini
        "meta-llama/Meta-Llama-3-8B-Instruct",      # Llama 3 8B
        "Qwen/Qwen2.5-7B-Instruct",                 # Qwen 2.5
    ]

    def __init__(self, api_token: str = None, model: str = None):
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.preferred_model = model or os.getenv("MISTRAL_MODEL")
        self.working_model = None  # Will be set when we find one that works

        if not self.api_token:
            raise ValueError(
                "Hugging Face API token not found! "
                "Get your FREE token from https://huggingface.co/settings/tokens "
                "and add it to your .env file as HUGGINGFACE_API_TOKEN"
            )

        self.client = InferenceClient(token=self.api_token)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.45,
        max_tokens: int = 1000,
    ) -> str:
        """Generate chat completion using chat_completion task (conversational)."""

        # Build ordered list of models to try
        models_to_try = []
        if self.working_model:
            models_to_try.append(self.working_model)
        elif self.preferred_model:
            models_to_try.append(self.preferred_model)
        models_to_try.extend(
            [m for m in self.FALLBACK_MODELS if m not in models_to_try]
        )

        last_error = None

        for model in models_to_try:
            try:
                # Use chat_completion — the correct task for instruct/chat models
                response = self.client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                response_text = response.choices[0].message.content

                if response_text and response_text.strip():
                    self.working_model = model
                    print(f"✅ Using model: {model}")
                    return response_text.strip()

            except Exception as e:
                error_msg = str(e).lower()
                last_error = str(e)

                if "loading" in error_msg or "503" in error_msg:
                    return "⏳ The AI model is loading. Please try again in 20–30 seconds!"

                print(f"⚠️ Model {model} failed: {str(e)[:120]}")
                continue

        return self._get_helpful_error_message(last_error)

    def _get_helpful_error_message(self, last_error: str) -> str:
        """Return a user-friendly error message."""
        error_lower = str(last_error).lower() if last_error else ""

        if "unauthorized" in error_lower or "401" in error_lower:
            return (
                "❌ Invalid Hugging Face API token.\n\n"
                "Fix:\n"
                "1. Go to https://huggingface.co/settings/tokens\n"
                "2. Create a new token (Read access)\n"
                "3. Update your .env: HUGGINGFACE_API_TOKEN=hf_your_token\n"
                "4. Restart the app"
            )

        if "rate limit" in error_lower or "429" in error_lower:
            return "⏱️ Rate limit reached. Please wait a few minutes and try again."

        if "forbidden" in error_lower or "403" in error_lower:
            return (
                "❌ Access forbidden.\n\n"
                "1. Go to https://huggingface.co/settings/tokens\n"
                "2. Create a new token with Read access\n"
                "3. Update HUGGINGFACE_API_TOKEN in .env\n"
                "4. Restart the app"
            )

        return (
            "❌ All free AI models are currently unavailable.\n\n"
            "This can happen when:\n"
            "• Models are being updated by Hugging Face\n"
            "• High demand on free tier\n"
            "• Network issues\n\n"
            "Please try again in a few minutes.\n\n"
            f"Error: {str(last_error)[:200] if last_error else 'Unknown'}"
        )


# ── OpenAI-compatible wrappers (app.py uses client.chat.completions.create) ──

class ChatCompletion:
    def __init__(self, client: MistralClient):
        self.client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.45,
        **kwargs,
    ):
        response_text = self.client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return MistralResponse(response_text)


class MistralResponse:
    def __init__(self, text: str):
        self.choices = [MistralChoice(text)]


class MistralChoice:
    def __init__(self, text: str):
        self.message = MistralMessage(text)


class MistralMessage:
    def __init__(self, text: str):
        self.content = text


class MistralClientWrapper:
    """Main client — drop-in replacement, mimics OpenAI's interface."""

    def __init__(self, api_token: str = None):
        mistral_client = MistralClient(api_token=api_token)
        self.chat = type(
            "Chat",
            (),
            {"completions": ChatCompletion(mistral_client)},
        )()
