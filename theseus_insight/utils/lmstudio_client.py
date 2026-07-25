"""
LMStudio client cache utility.

The LMStudio SDK uses a singleton pattern for its default client, which can only
be configured once per process. This module provides a caching layer to reuse
LMStudio inference clients across the application, avoiding the 
"Default client is already created" error.
"""

import logging
import os
import re
from typing import Optional, Dict, Any

import httpx
import requests
from openai import OpenAI

from LLMFactory.providers.lmstudio import LMStudioInference

logger = logging.getLogger(__name__)

# Module-level cache for LMStudio inference clients
# Key: (host, model_name) tuple
# Value: LMStudioInference instance
_lmstudio_client_cache: Dict[tuple, Any] = {}

# Track the configured host (LMStudio only allows one default client per process)
_configured_host: Optional[str] = None


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean env var with a safe default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


# Inline reasoning delimiters emitted by various local models when the runtime
# doesn't separate reasoning into a dedicated response field.
_THINK_TAGS = ("think", "thinking", "reason", "reasoning")


def _strip_think_blocks(text: str) -> str:
    """Remove inline reasoning blocks (e.g. <think>...</think>) from model output."""
    if not text:
        return text

    cleaned = text
    for tag in _THINK_TAGS:
        cleaned = re.sub(rf"<{tag}>.*?</{tag}>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Handle partially malformed output that opens a block but never closes it.
        if f"<{tag}>" in cleaned.lower():
            cleaned = re.sub(rf"<{tag}>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


class TheseusLMStudioInference(LMStudioInference):
    """Local wrapper around the LLMFactory LM Studio provider.

    The upstream provider is generally fine, but in this app we want client
    initialization to bypass any ambient proxy/env behavior and talk directly
    to the configured LM Studio host.
    """

    def __init__(
        self,
        *args,
        disable_thinking: Optional[bool] = None,
        request_timeout_sec: Optional[float] = None,
        **kwargs,
    ):
        # Default to disabling model thinking on LM Studio unless explicitly opted out.
        self.disable_thinking = (
            _env_flag("LMSTUDIO_DISABLE_THINKING", True)
            if disable_thinking is None else bool(disable_thinking)
        )
        self.request_timeout_sec = (
            None if request_timeout_sec is None else float(request_timeout_sec)
        )
        super().__init__(*args, **kwargs)

    def _is_qwen_model(self, model_name: Optional[str] = None) -> bool:
        resolved_name = (model_name or self.model_name or "").lower()
        return "qwen" in resolved_name

    def _apply_no_think_directive(self, messages):
        """Inject LM Studio/Qwen's no-think slash command into the latest user turn."""
        patched_messages = [dict(message) for message in messages]

        for index in range(len(patched_messages) - 1, -1, -1):
            message = patched_messages[index]
            if message.get("role") != "user":
                continue

            content = message.get("content", "")
            if isinstance(content, str) and "/no_think" not in content:
                message["content"] = (
                    content if content.rstrip().endswith("/no_think")
                    else f"{content.rstrip()}\n/no_think"
                )
            return patched_messages

        return patched_messages

    def _build_disable_thinking_extra_body(self, extra_body: Optional[dict] = None) -> dict:
        """Merge the LM Studio request-body switches that disable model thinking.

        Several keys are sent because different model templates honor different
        ones, and any a model doesn't recognize is ignored:
          - ``reasoning_effort: "none"`` — LM Studio's switch for reasoning models
            (e.g. Gemma) so they don't spend the token budget thinking.
          - ``enable_thinking: False`` (+ ``chat_template_kwargs``) — the
            Qwen-style chat-template flag.

        Caller-provided values take precedence so an explicit override wins.
        """
        merged_extra_body = dict(extra_body or {})
        chat_template_kwargs = dict(merged_extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        merged_extra_body["chat_template_kwargs"] = chat_template_kwargs
        # Also include the plain keys for clients/runtimes that read them directly.
        merged_extra_body.setdefault("enable_thinking", False)
        merged_extra_body.setdefault("reasoning_effort", "none")
        return merged_extra_body

    def _load_model(self):
        """Initialize the OpenAI-compatible client after a direct connectivity check."""
        session = requests.Session()
        session.trust_env = False

        try:
            response = session.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                f"Ensure LM Studio is running and the server is enabled. Error: {e}"
            )

        # Also disable env/proxy inheritance for the OpenAI/httpx client.
        http_client = httpx.Client(
            trust_env=False,
            timeout=None if self.request_timeout_sec is None else self.request_timeout_sec,
        )
        client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            http_client=http_client,
        )

        if self.disable_thinking:
            self._install_disable_thinking_hook(client)

        return client

    def _install_disable_thinking_hook(self, client: OpenAI) -> None:
        """Force the disable-thinking switches onto every chat completion request.

        Upstream ``LMStudioInference.invoke`` builds the request from a fixed
        kwarg allowlist and discards ``extra_body``, so the client is the only
        reliable place to inject ``reasoning_effort: "none"``. Wrapping
        ``chat.completions.create`` here keeps all of upstream's retry/streaming
        behavior intact while disabling thinking regardless of model family.
        The explicit-thinking path (``use_thinking=True``) uses LM Studio's
        ``/v1/responses`` endpoint directly and is unaffected by this hook.
        """
        completions = client.chat.completions
        original_create = completions.create

        def create_with_thinking_disabled(*args, **kwargs):
            kwargs["extra_body"] = self._build_disable_thinking_extra_body(
                kwargs.get("extra_body")
            )
            return original_create(*args, **kwargs)

        completions.create = create_with_thinking_disabled

    def invoke(
        self,
        messages,
        system_prompt,
        *,
        streaming: bool = False,
        model_name: Optional[str] = None,
        schema=None,
        images=None,
        use_thinking=False,
        return_thinking: bool = False,
        **kwargs,
    ):
        resolved_model_name = model_name or self.model_name

        # The disable-thinking request switches (reasoning_effort etc.) are
        # injected at the client level in _install_disable_thinking_hook; here we
        # only force use_thinking off and, for Qwen chat templates, add the inline
        # /no_think directive. The request still flows through upstream invoke()
        # so its model-reload retry logic is preserved.
        if self.disable_thinking:
            if use_thinking:
                logger.info(
                    "Overriding use_thinking for LM Studio model '%s' because thinking is disabled",
                    resolved_model_name,
                )
            use_thinking = False
            if self._is_qwen_model(resolved_model_name):
                messages = self._apply_no_think_directive(messages)

        response = super().invoke(
            messages=messages,
            system_prompt=system_prompt,
            streaming=streaming,
            model_name=model_name,
            schema=schema,
            images=images,
            use_thinking=use_thinking,
            return_thinking=return_thinking,
            **kwargs,
        )

        # Safety net: strip any inline reasoning the runtime didn't separate into
        # a dedicated field (applies to plain string responses only).
        if self.disable_thinking and isinstance(response, str):
            cleaned_response = _strip_think_blocks(response)
            if cleaned_response != response and any(
                f"<{tag}>" in response.lower() for tag in _THINK_TAGS
            ):
                logger.info(
                    "Stripped inline reasoning from LM Studio response for model '%s'",
                    resolved_model_name,
                )
            return cleaned_response

        return response


def get_lmstudio_client(
    model_name: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    host: str = "localhost:1234",
    context_length: Optional[int] = None,
    disable_thinking: Optional[bool] = None,
    request_timeout_sec: Optional[float] = None,
    **kwargs
):
    """
    Get or create an LMStudio inference client.
    
    This function caches clients to avoid the singleton error when the LMStudio
    SDK tries to reconfigure the default client.
    
    Args:
        model_name: Name of the model to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        host: LMStudio server host (e.g., "localhost:1234")
        context_length: Optional context length
        **kwargs: Additional arguments passed to LLMModelFactory
        
    Returns:
        LMStudioInference instance (cached if available)
        
    Raises:
        ValueError: If trying to use a different host than previously configured
    """
    global _configured_host
    
    cache_key = (host, model_name, request_timeout_sec)
    
    # Check if we have a cached client for this exact configuration
    if cache_key in _lmstudio_client_cache:
        logger.debug(f"Reusing cached LMStudio client for {model_name}@{host}")
        return _lmstudio_client_cache[cache_key]
    
    # Check if we're trying to use a different host than previously configured
    if _configured_host is not None and _configured_host != host:
        # LMStudio SDK limitation: can't change host after initial configuration
        # Try to find any cached client for the same host
        for (cached_host, cached_model), client in _lmstudio_client_cache.items():
            if cached_host == _configured_host:
                logger.warning(
                    f"LMStudio only supports one host per process. "
                    f"Requested host '{host}' differs from configured host '{_configured_host}'. "
                    f"Returning cached client for {cached_model}@{cached_host}"
                )
                return client
        
        raise ValueError(
            f"LMStudio SDK limitation: Cannot change host from '{_configured_host}' to '{host}'. "
            f"Restart the application to use a different host."
        )
    
    create_kwargs = {
        'model_name': model_name,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'host': host,
        'disable_thinking': disable_thinking,
        'request_timeout_sec': request_timeout_sec,
        **kwargs
    }
    
    if context_length is not None:
        create_kwargs['context_length'] = context_length
    
    try:
        client = TheseusLMStudioInference(**create_kwargs)
        _lmstudio_client_cache[cache_key] = client
        _configured_host = host
        logger.info(f"Created LMStudio client for {model_name}@{host}")
        return client
    except Exception as e:
        # Handle the case where default client already exists (race condition or previous error)
        if "already created" in str(e).lower():
            logger.warning(f"LMStudio default client already exists, checking cache...")
            # Return any cached client for this host
            for (cached_host, cached_model), cached_client in _lmstudio_client_cache.items():
                if cached_host == host:
                    logger.info(f"Found cached client for {cached_model}@{cached_host}")
                    _lmstudio_client_cache[cache_key] = cached_client
                    return cached_client
        raise


def clear_lmstudio_cache():
    """
    Clear the LMStudio client cache.
    
    Note: This doesn't reset the LMStudio SDK's internal state. 
    The configured host will still be locked until process restart.
    """
    global _lmstudio_client_cache, _configured_host
    _lmstudio_client_cache.clear()
    # Note: We intentionally don't reset _configured_host because
    # the LMStudio SDK's default client is still configured
    logger.info("LMStudio client cache cleared")


def get_configured_host() -> Optional[str]:
    """Get the currently configured LMStudio host, if any."""
    return _configured_host


def verify_model_loaded(host: str = "localhost:1234", model_name: Optional[str] = None) -> bool:
    """
    Verify that the LM Studio server has models loaded.
    
    This can help detect when LM Studio has auto-unloaded models after
    extended periods of inactivity.
    
    Args:
        host: LM Studio server host
        model_name: Optional specific model name to check for
        
    Returns:
        True if models are loaded (and optionally the specific model), False otherwise
    """
    import requests
    
    try:
        # LM Studio's API endpoint to list models
        url = f"http://{host}/v1/models"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            logger.warning(f"LM Studio models endpoint returned status {response.status_code}")
            return False
        
        data = response.json()
        models = data.get('data', [])
        
        if not models:
            logger.warning("LM Studio reports no models loaded (totalLoadedModels: 0)")
            return False
        
        if model_name:
            # Check if the specific model is loaded
            loaded_model_ids = [m.get('id', '') for m in models]
            if model_name not in loaded_model_ids:
                logger.warning(f"Model '{model_name}' not found in loaded models: {loaded_model_ids}")
                return False
        
        logger.debug(f"LM Studio has {len(models)} model(s) loaded")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to verify LM Studio models: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error checking LM Studio models: {e}")
        return False
