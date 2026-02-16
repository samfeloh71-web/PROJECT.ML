"""
Mistral AI Integration for Market Lens
Using Hugging Face InferenceClient (FREE)

This version automatically tries multiple FREE models until one works.
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
    
    # List of FREE models to try (in order of preference)
    FALLBACK_MODELS = [
        "microsoft/Phi-3-mini-4k-instruct",  # Microsoft's small but capable model
        "google/flan-t5-large",              # Google's instruction-tuned model
        "HuggingFaceH4/zephyr-7b-beta",     # Zephyr
        "meta-llama/Llama-2-7b-chat-hf",    # Llama 2
        "tiiuae/falcon-7b-instruct",         # Falcon
        "mistralai/Mistral-7B-Instruct-v0.2" # Original Mistral
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
        
        # Initialize the official Hugging Face Inference Client
        self.client = InferenceClient(token=self.api_token)
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.45, max_tokens: int = 1000) -> str:
        """Generate chat completion - tries multiple models until one works"""
        
        # Build list of models to try
        models_to_try = []
        if self.working_model:
            # If we already found a working model, use it first
            models_to_try.append(self.working_model)
        elif self.preferred_model:
            # Try user's preferred model first
            models_to_try.append(self.preferred_model)
        
        # Add fallback models
        models_to_try.extend([m for m in self.FALLBACK_MODELS if m not in models_to_try])
        
        last_error = None
        
        for model in models_to_try:
            try:
                # Format messages into a prompt
                prompt = self._format_messages_to_prompt(messages)
                
                # Try text_generation
                response_text = self.client.text_generation(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    return_full_text=False
                )
                
                if response_text and response_text.strip():
                    # Success! Remember this model for next time
                    self.working_model = model
                    print(f"✅ Using model: {model}")
                    return response_text.strip()
                    
            except Exception as e:
                error_msg = str(e).lower()
                last_error = str(e)
                
                # Check if it's a temporary error (model loading)
                if "model is currently loading" in error_msg or "503" in error_msg:
                    return f"⏳ The AI model is loading. Please try again in 20-30 seconds!"
                
                # Otherwise, try next model
                print(f"⚠️ Model {model} failed: {error_msg[:100]}")
                continue
        
        # If all models failed, return helpful error
        return self._get_helpful_error_message(last_error)
    
    def _get_helpful_error_message(self, last_error: str) -> str:
        """Generate a helpful error message with troubleshooting steps"""
        error_lower = str(last_error).lower() if last_error else ""
        
        if "unauthorized" in error_lower or "401" in error_lower:
            return """❌ Invalid Hugging Face API token.

Fix:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (with 'Read' access)
3. Update your .env file: HUGGINGFACE_API_TOKEN=hf_your_new_token
4. Restart the app"""
        
        elif "rate limit" in error_lower:
            return "⏱️ Rate limit reached. Please wait a few minutes and try again."
        
        elif "forbidden" in error_lower or "403" in error_lower:
            return """❌ Access forbidden.

Your token may need additional permissions:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with 'Read' access
3. Update HUGGINGFACE_API_TOKEN in .env
4. Restart the app"""
        
        else:
            return f"""❌ All free AI models are currently unavailable.

This can happen when:
• Models are being updated by Hugging Face
• High demand on free tier
• Network issues

Please try again in a few minutes, or contact support if the issue persists.

Error: {last_error[:200]}"""
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a simple prompt format that works with most models"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == "user":
                prompt_parts.append(f"Question: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Answer: {content}\n")
        
        prompt_parts.append("Answer:")
        return "\n".join(prompt_parts)


class ChatCompletion:
    """OpenAI-compatible wrapper"""
    
    def __init__(self, client: MistralClient):
        self.client = client
    
    def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.45, **kwargs):
        """OpenAI-compatible create method"""
        response_text = self.client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        return MistralResponse(response_text)


class MistralResponse:
    """OpenAI-compatible response object"""
    def __init__(self, text: str):
        self.choices = [MistralChoice(text)]


class MistralChoice:
    """OpenAI-compatible choice object"""
    def __init__(self, text: str):
        self.message = MistralMessage(text)


class MistralMessage:
    """OpenAI-compatible message object"""
    def __init__(self, text: str):
        self.content = text


class MistralClientWrapper:
    """Main client wrapper that mimics OpenAI's interface"""
    
    def __init__(self, api_token: str = None):
        mistral_client = MistralClient(api_token=api_token)
        self.chat = type('Chat', (), {
            'completions': ChatCompletion(mistral_client)
        })()
