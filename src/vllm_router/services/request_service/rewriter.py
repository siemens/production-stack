# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Request rewriter interface for vLLM router.

This module provides functionality to rewrite requests before they are sent to the backend.
"""

import abc
import json

from vllm_router.log import init_logger
from vllm_router.utils import SingletonABCMeta

logger = init_logger(__name__)


class RequestRewriter(metaclass=SingletonABCMeta):
    """
    Abstract base class for request rewriters.

    Request rewriters can modify the request body before it is sent to the backend.
    This can be used for prompt engineering, model-specific adjustments, or request normalization.
    """

    @abc.abstractmethod
    def rewrite_request(self, request_body: str, model: str, endpoint: str) -> str:
        """
        Rewrite the request body.

        Args:
            request_body: The original request body as string
            model: The model name from the request
            endpoint: The target endpoint of this request

        Returns:
            The rewritten request body as string
        """
        pass


class NoopRequestRewriter(RequestRewriter):
    """
    A request rewriter that does not modify the request.
    """

    def rewrite_request(self, request_body: str, model: str, endpoint: str) -> str:
        """
        Return the request body unchanged.

        Args:
            request_body: The original request body as string
            model: The model name from the request
            endpoint: The target endpoint of this request

        Returns:
            The original request body without any modifications
        """
        return request_body


class MessagesRewriter(RequestRewriter):
    """
    A request rewriter that normalizes messages before forwarding.

    Normalizations:
    - Filters out messages with empty/null content (some backends reject them).
      Assistant messages with ``tool_calls`` are preserved even with null content.
    - For ``/v1/messages``, promotes ``role: "system"`` entries in the messages
      array to the top-level ``system`` parameter (handles the ``mid-conversation-system``
      beta format sent by e.g. Claude Code).
    """

    def rewrite_request(self, request_body: str, model: str, endpoint: str) -> str:
        try:
            body = json.loads(request_body)
        except json.JSONDecodeError:
            return request_body

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return request_body

        # Guard: skip messages with empty content (some backends reject them).
        # Preserve assistant messages with tool_calls even when content is null/empty.
        messages = [m for m in messages if _message_has_content(m)]

        if not messages:
            return request_body

        # For Anthropic Messages API, also promote role: "system" to top-level system param.
        if endpoint == "/v1/messages":
            system_messages = [m for m in messages if m.get("role") == "system"]
            if system_messages:
                system_content = _join_system_content(system_messages)
                body["messages"] = [m for m in messages if m.get("role") != "system"]
                if body.get("system") is not None:
                    existing = body["system"]
                    if isinstance(existing, str):
                        body["system"] = existing + "\n" + system_content
                    elif isinstance(existing, list):
                        body["system"].append({"type": "text", "text": system_content})
                    else:
                        body["system"] = system_content
                else:
                    body["system"] = system_content

                logger.info(
                    "Promoted %d system message(s) from messages array to top-level system param",
                    len(system_messages),
                )
                return json.dumps(body)

            body["messages"] = messages
            return json.dumps(body)

        # For chat completions, just apply the empty-content guard.
        if endpoint in ("/v1/chat/completions", "/chat/completions"):
            body["messages"] = messages
            return json.dumps(body)

        return request_body


def _message_has_content(message: dict) -> bool:
    # Assistant messages with tool_calls, function_call, or refusal are valid
    # even with null/empty content.
    if message.get("tool_calls") or message.get("function_call") or message.get("refusal"):
        return True
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return content.strip() != ""
    if isinstance(content, list):
        return len(content) > 0
    return bool(content)


def _join_system_content(system_messages: list[dict]) -> str:
    parts = []
    for msg in system_messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


# Singleton instance
_request_rewriter_instance = None


def initialize_request_rewriter(rewriter_type: str, **kwargs) -> RequestRewriter:
    """
    Initialize the request rewriter singleton.

    Args:
        rewriter_type: The type of rewriter to initialize
        **kwargs: Additional arguments for the rewriter

    Returns:
        The initialized request rewriter instance
    """
    global _request_rewriter_instance

    if rewriter_type == "messages":
        _request_rewriter_instance = MessagesRewriter()
        logger.info("Initialized MessagesRewriter")
    else:
        _request_rewriter_instance = NoopRequestRewriter()
        logger.info(f"Initialized placeholder request rewriter (type: {rewriter_type})")

    return _request_rewriter_instance


def is_request_rewriter_initialized() -> bool:
    """
    Check if the request rewriter singleton has been initialized.

    Returns:
        bool: True if the request rewriter has been initialized, False otherwise
    """
    global _request_rewriter_instance
    return _request_rewriter_instance is not None


def get_request_rewriter() -> RequestRewriter:
    """
    Get the request rewriter singleton instance.

    Returns:
        The request rewriter instance or MessagesRewriter if not initialized
    """
    global _request_rewriter_instance
    if _request_rewriter_instance is None:
        _request_rewriter_instance = MessagesRewriter()
        logger.info("Initialized default MessagesRewriter")
    return _request_rewriter_instance
