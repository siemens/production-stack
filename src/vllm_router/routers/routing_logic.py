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

from __future__ import annotations

import abc
import asyncio
import concurrent.futures
import enum
import inspect
import math
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from fastapi import HTTPException, Request

try:
    from transformers import AutoTokenizer
except ImportError:
    pass

try:
    from lmcache.v1.cache_controller import controller_manager
    from lmcache.v1.cache_controller.message import (
        LookupMsg,
        QueryInstMsg,
    )
except ImportError:
    pass
from uhashring import HashRing

from vllm_router.log import init_logger
from vllm_router.prefix.hashtrie import HashTrie
from vllm_router.service_discovery import EndpointInfo
from vllm_router.stats.engine_stats import EngineStats
from vllm_router.stats.request_stats import RequestStats
from vllm_router.utils import SingletonABCMeta

logger = init_logger(__name__)


def _serialize_prefix_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _serialize_prefix_content_block(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            serialized = _serialize_prefix_content(part)
            if serialized:
                parts.append(serialized)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _serialize_prefix_content_block(content_block: dict) -> str:
    part_type = str(content_block.get("type", "")).strip()
    if part_type in {"text", "input_text", "output_text"}:
        text = content_block.get("text", "")
        return str(text) if text else ""
    if part_type:
        return f"[{part_type}]"

    text = content_block.get("text", "")
    if text:
        return str(text)
    return str(content_block)


def _build_prefix_message_lines(messages: object) -> list[str]:
    if not isinstance(messages, list):
        return []

    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue

        role = str(message.get("role", "user")).strip().lower() or "user"
        name = str(message.get("name", "")).strip()
        header = f"<|{role}:{name}|>" if name else f"<|{role}|>"
        content = _serialize_prefix_content(message.get("content", ""))
        lines.append(f"{header}\n{content}")
    return lines


def _build_chat_prefix_key(request_json: dict) -> str:
    lines: list[str] = []

    system_content = request_json.get("system")
    if system_content:
        lines.append(f"<|system|>\n{_serialize_prefix_content(system_content)}")

    instructions = request_json.get("instructions")
    if instructions:
        lines.append(f"<|developer|>\n{_serialize_prefix_content(instructions)}")

    lines.extend(_build_prefix_message_lines(request_json.get("messages", [])))

    return "\n<|message-break|>\n".join(lines)


def _build_responses_prefix_key(request_json: dict) -> str:
    lines: list[str] = []

    instructions = request_json.get("instructions")
    if instructions:
        lines.append(f"<|developer|>\n{_serialize_prefix_content(instructions)}")

    response_input = request_json.get("input", "")
    if isinstance(response_input, str):
        lines.append(f"<|input|>\n{response_input}")
    elif isinstance(response_input, list):
        lines.extend(_build_prefix_message_lines(response_input))
    elif response_input is not None:
        lines.append(_serialize_prefix_content(response_input))

    return "\n<|message-break|>\n".join(lines)


def _build_request_prefix_key(request_json: dict) -> str:
    if "messages" in request_json:
        return _build_chat_prefix_key(request_json)

    if "input" in request_json:
        return _build_responses_prefix_key(request_json)

    prompt = request_json.get("prompt", "")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n<|prompt-break|>\n".join(str(item) for item in prompt)
    if prompt is None:
        return ""
    return str(prompt)


class RoutingLogic(str, enum.Enum):
    ROUND_ROBIN = "roundrobin"
    SESSION_BASED = "session"
    KVAWARE = "kvaware"
    PREFIXAWARE = "prefixaware"
    DISAGGREGATED_PREFILL = "disaggregated_prefill"
    DISAGGREGATED_PREFILL_ORCHESTRATED = "disaggregated_prefill_orchestrated"


class RoutingInterface(metaclass=SingletonABCMeta):
    def _qps_routing(
        self, endpoints: List[EndpointInfo], request_stats: Dict[str, RequestStats]
    ) -> str:
        """
        Route the request to the appropriate engine URL based on the QPS of
        each engine

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
        """
        lowest_qps = float("inf")
        ret = None
        for info in endpoints:
            url = info.url
            if url not in request_stats:
                return url  # This engine does not have any requests
            request_stat = request_stats[url]
            if request_stat.qps < lowest_qps:
                lowest_qps = request_stat.qps
                ret = url
        return ret

    def _update_hash_ring(self, endpoints: List["EndpointInfo"]):
        """
        Update the hash ring with the current list of endpoints.
        """
        # Extract endpoint URLs
        endpoint_urls = [endpoint.url for endpoint in endpoints]

        # Get the current nodes in the hash ring
        current_nodes = set(self.hash_ring.get_nodes())

        # Convert the new endpoint URLs to a set for easy comparison
        new_nodes = set(endpoint_urls)

        # Remove nodes that are no longer in the list
        for node in current_nodes - new_nodes:
            self.hash_ring.remove_node(node)

        # Add new nodes that are not already in the hash ring
        for node in new_nodes - current_nodes:
            self.hash_ring.add_node(node)

    def extract_session_id(self, request: Request, request_json: Dict) -> Optional[str]:
        """
        Extract the session id from the request headers or request body.
        """
        session_key = getattr(self, "session_key", None)
        if session_key is None:
            return None
        val = request.headers.get(session_key)
        return val if val is not None else (request_json or {}).get(session_key, None)

    @abc.abstractmethod
    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        raise NotImplementedError


class RoundRobinRouter(RoutingInterface):
    # TODO (ApostaC): when available engines in the endpoints changes, the
    # algorithm may not be "perfectly" round-robin.

    # Upper bound on cached endpoint-set entries to prevent unbounded memory
    # growth when endpoints change dynamically (add / remove / update).
    _MAX_CACHE_SIZE = 1024

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._next_index: dict[tuple[str, ...], int] = {}
        self._sorted_cache: dict[frozenset[str], tuple[str, ...]] = {}
        self._initialized = True

    def _endpoint_key(self, endpoints: List[EndpointInfo]) -> tuple[str, ...]:
        """Return a stable, sorted key for the endpoint set (cached after first sort)."""
        if not endpoints:
            raise ValueError("RoundRobinRouter requires at least one endpoint")

        urls = frozenset(e.url for e in endpoints)
        key = self._sorted_cache.get(urls)
        if key is None:
            if len(self._sorted_cache) >= self._MAX_CACHE_SIZE:
                self._sorted_cache.clear()
            key = tuple(sorted(urls))
            self._sorted_cache[urls] = key
        return key

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route the request to the appropriate engine URL using a simple
        round-robin algorithm

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        endpoint_urls = self._endpoint_key(endpoints)
        idx = self._next_index.get(endpoint_urls, 0)
        if (
            len(self._next_index) >= self._MAX_CACHE_SIZE
            and endpoint_urls not in self._next_index
        ):
            self._next_index.clear()
        self._next_index[endpoint_urls] = idx + 1
        return endpoint_urls[idx % len(endpoint_urls)]


class SessionRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL based on the session key
    in the request headers
    """

    def __init__(self, session_key: str = None):
        if hasattr(self, "_initialized"):
            return
        if session_key is None:
            raise ValueError("SessionRouter must be initialized with a session_key")
        self.session_key = session_key
        self.hash_ring = HashRing()
        self._initialized = True

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route the request to the appropriate engine URL by the 'session id' in
        the request headers or request body.
        If there is no session id in the request header or request body, it will pick a server
        with lowest qps

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_json (Dict): The request body (needed for finding the session id)
        """
        session_id = self.extract_session_id(request, request_json)
        logger.debug(f"Got session id: {session_id}")

        # Update the hash ring with the current list of endpoints
        self._update_hash_ring(endpoints)

        if session_id is None:
            # Route based on QPS if no session ID is present
            url = self._qps_routing(endpoints, request_stats)
        else:
            # Use the hash ring to get the endpoint for the session ID
            url = self.hash_ring.get_node(session_id)

        return url


class KvawareRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL by where the KV cache
    of the longest prefix match is found.
    """

    def __init__(
        self,
        lmcache_controller_port: int,
        session_key: str,
        kv_aware_threshold: int = 2000,
        lmcache_health_check_interval: int = 5,
        lmcache_worker_timeout: int = 30,
        lmcache_controller_reply_port: Optional[int] = None,
        lmcache_controller_heartbeat_port: Optional[int] = None,
        tokenizer_model_names: Optional[Dict[str, str]] = None,
        kv_aware_per_model_thresholds: Optional[Dict[str, int]] = None,
    ):
        self.lmcache_controller_port = lmcache_controller_port
        self.lmcache_controller_reply_port = lmcache_controller_reply_port
        self.lmcache_controller_heartbeat_port = lmcache_controller_heartbeat_port
        logger.info(
            f"Initializing KvawareRouter with port: {self.lmcache_controller_port}, "
            f"reply port: {self.lmcache_controller_reply_port}, "
            f"heartbeat port: {self.lmcache_controller_heartbeat_port}"
        )
        controller_urls = {
            "pull": f"0.0.0.0:{self.lmcache_controller_port}",
            "reply": (
                f"0.0.0.0:{self.lmcache_controller_reply_port}"
                if self.lmcache_controller_reply_port is not None
                else None
            ),
        }
        if self.lmcache_controller_heartbeat_port is not None:
            controller_urls["heartbeat"] = (
                f"0.0.0.0:{self.lmcache_controller_heartbeat_port}"
            )
        self.kv_manager = controller_manager.LMCacheControllerManager(
            controller_urls,
            health_check_interval=lmcache_health_check_interval,
            lmcache_worker_timeout=lmcache_worker_timeout,
        )
        self.req_id = 0
        self.instance_id_to_ip = {}
        self.session_key = session_key
        self.hash_ring = HashRing()
        self.tokenizers = {}
        self.threshold = kv_aware_threshold
        self.tokenizer_model_names = tokenizer_model_names or {}
        self.per_model_thresholds = kv_aware_per_model_thresholds or {}

    def _get_tokenizer(self, tokenizer_name: str):
        tokenizer = self.tokenizers.get(tokenizer_name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            self.tokenizers[tokenizer_name] = tokenizer
        return tokenizer

    def start_kv_manager(self):
        """
        Start the kv manager
        """
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.lmcache_cluster_monitor_task = asyncio.run_coroutine_threadsafe(
            self.kv_manager.start_all(), self.loop
        )

    async def query_manager(self, msg) -> str:
        """
        Get the instance id for the given message
        """
        instance_id = self.kv_manager.handle_orchestration_message(msg)
        if inspect.isawaitable(instance_id):
            instance_id = await instance_id
        return instance_id

    def close(self):
        """Gracefully shutdown the lmcache cluster monitor task."""
        if (
            hasattr(self, "lmcache_cluster_monitor_task")
            and self.lmcache_cluster_monitor_task
        ):
            logger.info("Shutting down lmcache cluster monitor task")
            self.lmcache_cluster_monitor_task.cancel()
            try:
                self.lmcache_cluster_monitor_task.result()
            except concurrent.futures.CancelledError:
                pass
            self.lmcache_cluster_monitor_task = None

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route the request to the appropriate engine URL by where the KV cache
        of the longest prefix match is found.
        If there is no session id in the request header, it will pick a server
        with round robin.

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
               the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
               indicating the request-level performance of each engine
            request (Request): The incoming request
            request_json (Dict): The request body (needed for finding the
            longest prefix match)
        """
        token_ids = None
        _request_json = request_json or {}
        prompt = _build_request_prefix_key(_request_json)
        model_name = _request_json.get("model")
        if not model_name and endpoints:
            model_name = endpoints[0].model_names[0]
        threshold = self.per_model_thresholds.get(model_name, self.threshold)

        # Local-first tokenization, fall back to remote "/tokenize" API on failure
        try:
            tokenizer_name = self.tokenizer_model_names.get(model_name, model_name)
            tokenizer = self._get_tokenizer(tokenizer_name)
            token_ids = tokenizer.encode(prompt)
        except Exception:
            # Remote /tokenize fallback (let errors bubble up to keep behavior simple)
            remote_url = endpoints[0].url + "/tokenize"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": endpoints[0].model_names[0],
                "prompt": prompt,
            }
            body = requests.post(
                remote_url, headers=headers, json=data, timeout=10
            ).json()
            token_ids = body["tokens"]

        event_id = "Lookup" + str(uuid.uuid4())
        msg = LookupMsg(tokens=token_ids, event_id=event_id)
        instance_id = await self.query_manager(msg)
        matched_tokens = math.inf
        logger.debug(f"Lookup return message: {instance_id}")
        if instance_id is not None and len(list(instance_id.layout_info.keys())) > 0:
            matched_instance_id = list(instance_id.layout_info.keys())[
                0
            ]  # Get the first key
            matched_tokens = instance_id.layout_info[matched_instance_id][1]

        if (
            instance_id is None
            or len(instance_id.layout_info) == 0
            or matched_tokens < max(len(token_ids) - threshold, 0)
        ):
            session_id = self.extract_session_id(request, request_json)
            logger.debug(f"Fallback to using session id: {session_id}")
            # Update the hash ring with the current list of endpoints
            self._update_hash_ring(endpoints)
            if session_id is None:
                # Route based on QPS if no session ID is present
                url = self._qps_routing(endpoints, request_stats)
            else:
                # Use the hash ring to get the endpoint for the session ID
                url = self.hash_ring.get_node(session_id)
            return url
        else:
            queried_instance_ids = [info for info in instance_id.layout_info]
            if queried_instance_ids[0] not in self.instance_id_to_ip:
                for endpoint in endpoints:
                    event_id = "QueryInst" + str(uuid.uuid4())
                    query_ip = endpoint.url.split(f":{endpoint.url.split(':')[-1]}")[
                        0
                    ].split("//")[1]
                    query_message = QueryInstMsg(
                        ip=query_ip,
                        event_id=event_id,
                    )
                    endpoint_instance_id = await self.query_manager(query_message)
                    logger.debug(
                        f"Query ip: {query_ip}, return instance id: {endpoint_instance_id}"
                    )
                    self.instance_id_to_ip[endpoint_instance_id.instance_id] = (
                        endpoint.url
                    )
                logger.info(f"Instance id to ip mapping: {self.instance_id_to_ip}")
            logger.info(
                f"Routing request to {queried_instance_ids[0]} found by kvaware router"
            )
            return self.instance_id_to_ip[queried_instance_ids[0]]


@dataclass(slots=True)
class PrefixRouteContext:
    model_name: str
    trie_key: str
    prefix_key: str
    selected_url: str
    used_prefix_match: bool
    match_length: int


@dataclass(frozen=True, slots=True)
class PrefixAwareModelConfig:
    match_threshold: int = 0
    enabled: bool = True


class PrefixAwareRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL by where the longest
    prefix match is found.

    Uses model-scoped HashTries to avoid cross-model prefix contamination.
    For chat completions, builds role-aware prefix keys from message structure.
    Selects among matched endpoints using QPS-based load balancing.
    Defers trie insertion until the backend successfully accepts the request.
    """

    def __init__(
        self,
        prefix_aware_match_threshold: int = 0,
        prefix_aware_per_model_config: Optional[Dict[str, Dict[str, object]]] = None,
    ):
        if hasattr(self, "_initialized"):
            return

        self.global_config = PrefixAwareModelConfig(
            match_threshold=self._validate_match_threshold(
                prefix_aware_match_threshold, "global"
            ),
        )
        self.per_model_config = self._build_per_model_config(
            prefix_aware_per_model_config
        )
        self.hashtries: dict[str, HashTrie] = {}
        self._hashtries_lock = asyncio.Lock()
        self._initialized = True

    @staticmethod
    def _validate_match_threshold(value: object, config_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"Prefix-aware match_threshold for {config_name} must be a non-negative integer"
            )
        return value

    def _build_per_model_config(
        self, per_model_config: Optional[Dict[str, Dict[str, object]]]
    ) -> dict[str, PrefixAwareModelConfig]:
        configs: dict[str, PrefixAwareModelConfig] = {}
        for model_name, overrides in (per_model_config or {}).items():
            if not isinstance(model_name, str) or not model_name:
                raise ValueError(
                    "Prefix-aware per-model config keys must be model names"
                )
            if not isinstance(overrides, dict):
                raise ValueError(
                    f"Prefix-aware per-model config for {model_name} must be an object"
                )

            unknown_keys = set(overrides) - {"match_threshold", "enabled"}
            if unknown_keys:
                raise ValueError(
                    f"Unknown prefix-aware config keys for {model_name}: {sorted(unknown_keys)}"
                )

            enabled = overrides.get("enabled", self.global_config.enabled)
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"Prefix-aware enabled override for {model_name} must be a boolean"
                )

            match_threshold = self._validate_match_threshold(
                overrides.get("match_threshold", self.global_config.match_threshold),
                model_name,
            )
            configs[model_name] = PrefixAwareModelConfig(
                match_threshold=match_threshold,
                enabled=enabled,
            )
        return configs

    def _get_model_config(self, model_name: str) -> PrefixAwareModelConfig:
        return self.per_model_config.get(model_name, self.global_config)

    async def _get_hashtrie(self, model_name: str) -> HashTrie:
        trie = self.hashtries.get(model_name)
        if trie is not None:
            return trie

        async with self._hashtries_lock:
            trie = self.hashtries.get(model_name)
            if trie is None:
                trie = HashTrie()
                self.hashtries[model_name] = trie
            return trie

    @staticmethod
    def _serialize_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return PrefixAwareRouter._serialize_content_block(content)
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                serialized = PrefixAwareRouter._serialize_content(part)
                if serialized:
                    parts.append(serialized)
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _serialize_content_block(content_block: dict) -> str:
        part_type = str(content_block.get("type", "")).strip()
        if part_type in {"text", "input_text", "output_text"}:
            text = content_block.get("text", "")
            return str(text) if text else ""
        if part_type:
            return f"[{part_type}]"

        text = content_block.get("text", "")
        if text:
            return str(text)
        return str(content_block)

    def _build_message_lines(self, messages: object) -> list[str]:
        if not isinstance(messages, list):
            return []

        lines: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                lines.append(str(message))
                continue

            role = str(message.get("role", "user")).strip().lower() or "user"
            name = str(message.get("name", "")).strip()
            header = f"<|{role}:{name}|>" if name else f"<|{role}|>"
            content = self._serialize_content(message.get("content", ""))
            lines.append(f"{header}\n{content}")
        return lines

    def _build_chat_prefix(self, request_json: dict) -> str:
        lines: list[str] = []

        system_content = request_json.get("system")
        if system_content:
            lines.append(f"<|system|>\n{self._serialize_content(system_content)}")

        instructions = request_json.get("instructions")
        if instructions:
            lines.append(f"<|developer|>\n{self._serialize_content(instructions)}")

        lines.extend(self._build_message_lines(request_json.get("messages", [])))

        return "\n<|message-break|>\n".join(lines)

    def _build_responses_prefix(self, request_json: dict) -> str:
        lines: list[str] = []

        instructions = request_json.get("instructions")
        if instructions:
            lines.append(f"<|developer|>\n{self._serialize_content(instructions)}")

        response_input = request_json.get("input", "")
        if isinstance(response_input, str):
            lines.append(f"<|input|>\n{response_input}")
        elif isinstance(response_input, list):
            lines.extend(self._build_message_lines(response_input))
        elif response_input is not None:
            lines.append(self._serialize_content(response_input))

        return "\n<|message-break|>\n".join(lines)

    def _build_prefix_key(self, request_json: dict) -> str:
        if "messages" in request_json:
            return self._build_chat_prefix(request_json)

        if "input" in request_json:
            return self._build_responses_prefix(request_json)

        prompt = request_json.get("prompt", "")
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return "\n<|prompt-break|>\n".join(str(item) for item in prompt)
        if prompt is None:
            return ""
        return str(prompt)

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route the request to the appropriate engine URL by where the longest
        prefix match is found.

        If the request has `prefix_aware_bypass` set (from a prior failed
        prefix-routed attempt), falls back to QPS routing directly.

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
               the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
               indicating the request-level performance of each engine
            request (Request): The incoming request
            request_json (Dict): The request body (needed for finding the
            longest prefix match)
        """
        if not endpoints:
            raise ValueError("PrefixAwareRouter requires at least one endpoint")

        _request_json = request_json or {}

        if getattr(request.state, "prefix_aware_bypass", False):
            logger.warning("Falling back to QPS routing after prefix match failure")
            return self._qps_routing(endpoints, request_stats)

        model_name = _request_json.get("model")
        if not isinstance(model_name, str) or not model_name:
            logger.warning(
                "Missing model for prefix-aware routing, falling back to QPS"
            )
            return self._qps_routing(endpoints, request_stats)

        model_config = self._get_model_config(model_name)
        if not model_config.enabled:
            logger.debug(
                f"Prefix-aware routing disabled for model={model_name}, falling back to QPS"
            )
            return self._qps_routing(endpoints, request_stats)

        raw_prefix_key = self._build_prefix_key(_request_json)
        if not raw_prefix_key:
            logger.debug("Empty prefix key, falling back to QPS routing")
            return self._qps_routing(endpoints, request_stats)
        trie_key = f"{model_name}\n{request.url.path}"
        prefix_key = raw_prefix_key

        trie = await self._get_hashtrie(trie_key)
        available_urls = {ep.url for ep in endpoints}
        match_length, matched_urls = await trie.longest_prefix_match(
            prefix_key,
            available_urls,
        )

        use_prefix_match = (
            match_length > 0
            and bool(matched_urls)
            and match_length >= model_config.match_threshold
        )
        candidate_urls = matched_urls if use_prefix_match else available_urls
        candidate_endpoints = [ep for ep in endpoints if ep.url in candidate_urls]

        if not candidate_endpoints:
            candidate_endpoints = endpoints
            use_prefix_match = False

        selected_url = self._qps_routing(candidate_endpoints, request_stats)

        request.state.prefix_aware_ctx = PrefixRouteContext(
            model_name=model_name,
            trie_key=trie_key,
            prefix_key=prefix_key,
            selected_url=selected_url,
            used_prefix_match=use_prefix_match,
            match_length=match_length,
        )

        logger.info(
            f"Prefix-aware routing: model={model_name}, "
            f"match_length={match_length}, "
            f"match_threshold={model_config.match_threshold}, "
            f"used_prefix_match={use_prefix_match}, "
            f"selected_url={selected_url}"
        )

        return selected_url

    async def record_successful_route(
        self, request: Request, selected_url: str
    ) -> None:
        ctx: Optional[PrefixRouteContext] = getattr(
            request.state, "prefix_aware_ctx", None
        )
        if ctx is None or not ctx.prefix_key:
            return

        trie = await self._get_hashtrie(ctx.trie_key)
        await trie.insert(ctx.prefix_key, selected_url)

        logger.info(
            f"Recorded prefix route: model={ctx.model_name}, "
            f"endpoint={selected_url}, "
            f"prefix_len={len(ctx.prefix_key)}"
        )

    def mark_failed_route(self, request: Request) -> None:
        ctx: Optional[PrefixRouteContext] = getattr(
            request.state, "prefix_aware_ctx", None
        )
        if ctx is not None and ctx.used_prefix_match:
            request.state.prefix_aware_bypass = True
            logger.info(
                f"Prefix route failed for model={ctx.model_name}, "
                f"switching to bypass mode for this request"
            )


class DisaggregatedPrefillRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL by handling prefill and decode operations sequentially.
    First request goes to prefill endpoint, then second request goes to decode endpoint.

    Decode endpoint selection tries kv-aware, then prefix-aware, then fallback.
    """

    def __init__(
        self,
        prefill_model_labels: List[str],
        decode_model_labels: List[str],
        prefix_aware_match_threshold: int = 0,
        prefix_aware_per_model_config: Optional[Dict[str, Dict[str, object]]] = None,
        tokenizer_model_names: Optional[Dict[str, str]] = None,
        lmcache_controller_port: Optional[int] = None,
        kv_aware_threshold: int = 2000,
        kv_aware_per_model_thresholds: Optional[Dict[str, int]] = None,
        lmcache_health_check_interval: int = 5,
        lmcache_worker_timeout: int = 30,
        lmcache_controller_reply_port: Optional[int] = None,
        lmcache_controller_heartbeat_port: Optional[int] = None,
    ):
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels
        self.request_cache = {}  # Cache to store prefill results

        self._prefix_router = PrefixAwareRouter(
            prefix_aware_match_threshold=prefix_aware_match_threshold,
            prefix_aware_per_model_config=prefix_aware_per_model_config,
        )

        if lmcache_controller_port is not None:
            self._kv_router = KvawareRouter(
                lmcache_controller_port=lmcache_controller_port,
                session_key=None,
                kv_aware_threshold=kv_aware_threshold,
                lmcache_health_check_interval=lmcache_health_check_interval,
                lmcache_worker_timeout=lmcache_worker_timeout,
                lmcache_controller_reply_port=lmcache_controller_reply_port,
                lmcache_controller_heartbeat_port=lmcache_controller_heartbeat_port,
                tokenizer_model_names=tokenizer_model_names,
                kv_aware_per_model_thresholds=kv_aware_per_model_thresholds,
            )
            self._kv_router.start_kv_manager()
        else:
            self._kv_router = None

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route the request to appropriate endpoints for prefill and decode operations.
        First request goes to prefill endpoint, then second request goes to decode endpoint.
        """
        # Find prefill and decode endpoints
        is_prefill = (request_json or {}).get("max_tokens", 0) == 1
        if is_prefill:
            logger.info("Prefill request")
        else:
            logger.info("Decode request")

        # Find endpoints with matching model labels
        prefiller_endpoints = [
            e for e in endpoints if e.model_label in self.prefill_model_labels
        ]
        decoder_endpoints = [
            e for e in endpoints if e.model_label in self.decode_model_labels
        ]
        if is_prefill:
            return prefiller_endpoints[0].url
        else:
            # Decode: use kv-aware routing when configured
            selected = await self._select_decode_endpoint(
                decoder_endpoints, engine_stats, request_stats, request, request_json
            )
            return selected.url

    async def _select_decode_endpoint(
        self,
        decoder_endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> EndpointInfo:
        """Select decode endpoint using kv-aware, then prefix-aware fallback."""
        if not decoder_endpoints:
            raise ValueError("No decode endpoints available")

        if self._kv_router is not None:
            try:
                kv_url = await self._kv_router.route_request(
                    decoder_endpoints,
                    engine_stats,
                    request_stats,
                    request,
                    request_json,
                )
                for ep in decoder_endpoints:
                    if ep.url == kv_url:
                        logger.info(f"Decode selected via kv-aware: {kv_url}")
                        return ep
                logger.warning(f"KV-aware returned stale decode endpoint: {kv_url}")
            except Exception as e:
                logger.warning(f"KV-aware routing failed for decode: {e}")

        try:
            prefix_url = await self._prefix_router.route_request(
                decoder_endpoints,
                engine_stats,
                request_stats,
                request,
                request_json,
            )
            ctx = getattr(request.state, "prefix_aware_ctx", None)
            if ctx and ctx.used_prefix_match:
                for ep in decoder_endpoints:
                    if ep.url == prefix_url:
                        logger.info(f"Decode selected via prefix-aware: {prefix_url}")
                        return ep
        except Exception as e:
            logger.warning(f"Prefix-aware routing failed for decode: {e}")

        # Fall back to first endpoint
        logger.info(f"Decode selected via fallback: {decoder_endpoints[0].url}")
        return decoder_endpoints[0]


class DisaggregatedPrefillOrchestratedRouter(RoutingInterface):
    """
    Orchestrates disaggregated inference in a single request by chaining Prefill → Decode.

    Unlike DisaggregatedPrefillRouter (which requires 2 separate client requests),
    this router handles the entire flow internally:
    1. Receives request from client
    2. Forwards to Prefill endpoint with kv_transfer_params to enable disaggregated mode
    3. Gets prefill response with kv_transfer_params containing KV cache metadata
    4. Extracts kv_transfer_params, sets remote_host, and forwards to Decode
    5. Streams decode response back to client

    Load balancing: Uses round-robin across available prefill and decode pods.
    Decode endpoint selection tries kv-aware, then prefix-aware, then round-robin.
    """

    def __init__(
        self,
        prefill_model_labels: List[str],
        decode_model_labels: List[str],
        fallback_routing_logic: Optional[str] = None,
        session_key: Optional[str] = None,
        prefix_aware_match_threshold: int = 0,
        prefix_aware_per_model_config: Optional[Dict[str, Dict[str, object]]] = None,
        tokenizer_model_names: Optional[Dict[str, str]] = None,
        lmcache_controller_port: Optional[int] = None,
        kv_aware_threshold: int = 2000,
        kv_aware_per_model_thresholds: Optional[Dict[str, int]] = None,
        lmcache_health_check_interval: int = 5,
        lmcache_worker_timeout: int = 30,
        lmcache_controller_reply_port: Optional[int] = None,
        lmcache_controller_heartbeat_port: Optional[int] = None,
    ):
        if hasattr(self, "_initialized"):
            return
        self.prefill_model_labels = prefill_model_labels or []
        self.decode_model_labels = decode_model_labels or []
        self.prefill_idx = 0
        self.decode_idx = 0

        if fallback_routing_logic == "session":
            self._fallback_router: RoutingInterface = SessionRouter(session_key)
        elif fallback_routing_logic == "roundrobin":
            self._fallback_router = RoundRobinRouter()
        elif session_key:
            self._fallback_router = SessionRouter(session_key)
        else:
            self._fallback_router = RoundRobinRouter()

        self._prefix_router = PrefixAwareRouter(
            prefix_aware_match_threshold=prefix_aware_match_threshold,
            prefix_aware_per_model_config=prefix_aware_per_model_config,
        )

        if lmcache_controller_port is not None:
            self._kv_router = KvawareRouter(
                lmcache_controller_port=lmcache_controller_port,
                session_key=session_key,
                kv_aware_threshold=kv_aware_threshold,
                lmcache_health_check_interval=lmcache_health_check_interval,
                lmcache_worker_timeout=lmcache_worker_timeout,
                lmcache_controller_reply_port=lmcache_controller_reply_port,
                lmcache_controller_heartbeat_port=lmcache_controller_heartbeat_port,
                tokenizer_model_names=tokenizer_model_names,
                kv_aware_per_model_thresholds=kv_aware_per_model_thresholds,
            )
            self._kv_router.start_kv_manager()
        else:
            self._kv_router = None

        self._initialized = True
        logger.info(
            f"Initialized DisaggregatedPrefillOrchestratedRouter with "
            f"prefill_labels={self.prefill_model_labels}, "
            f"decode_labels={self.decode_model_labels}, "
            f"fallback={type(self._fallback_router).__name__}, "
            f"prefix_aware={True}, "
            f"kv_aware={self._kv_router is not None}"
        )

    def _find_endpoints(self, endpoints: List[EndpointInfo]):
        """Find prefill and decode endpoints based on model labels.

        Raises:
            HTTPException: 503 if prefill or decode endpoints are not available.
                - PREFILL_SERVICE_UNAVAILABLE: No prefill endpoints discovered
                - DECODE_SERVICE_UNAVAILABLE: No decode endpoints discovered
        """
        prefiller_endpoints = [
            e for e in endpoints if e.model_label in self.prefill_model_labels
        ]
        decoder_endpoints = [
            e for e in endpoints if e.model_label in self.decode_model_labels
        ]

        if not prefiller_endpoints:
            logger.warning(
                f"No prefill endpoints found with labels {self.prefill_model_labels}. "
                f"Available endpoints: {[(e.url, e.model_label) for e in endpoints]}"
            )
            raise HTTPException(
                status_code=503,
                detail="PREFILL_SERVICE_UNAVAILABLE: No prefill endpoints discovered",
            )
        if not decoder_endpoints:
            logger.warning(
                f"No decode endpoints found with labels {self.decode_model_labels}. "
                f"Available endpoints: {[(e.url, e.model_label) for e in endpoints]}"
            )
            raise HTTPException(
                status_code=503,
                detail="DECODE_SERVICE_UNAVAILABLE: No decode endpoints discovered",
            )

        return prefiller_endpoints, decoder_endpoints

    def select_prefill_endpoint(
        self, prefiller_endpoints: List[EndpointInfo]
    ) -> EndpointInfo:
        """Select prefill endpoint using round-robin load balancing."""
        if not prefiller_endpoints:
            raise ValueError("No prefill endpoints available")
        # Sort for consistency across requests
        sorted_endpoints = sorted(prefiller_endpoints, key=lambda e: e.url)
        selected = sorted_endpoints[self.prefill_idx % len(sorted_endpoints)]
        self.prefill_idx += 1
        return selected

    async def select_decode_endpoint(
        self,
        decoder_endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> EndpointInfo:
        """Select decode endpoint using kv-aware, then prefix-aware fallback.

        1. Try kv-aware routing when LMCache is configured.
        2. Try prefix-aware routing if no usable kv route is found.
        3. Fall back to round-robin if neither finds a good match.
        """
        if not decoder_endpoints:
            raise ValueError("No decode endpoints available")

        if self._kv_router is not None:
            try:
                kv_url = await self._kv_router.route_request(
                    decoder_endpoints,
                    engine_stats,
                    request_stats,
                    request,
                    request_json,
                )
                for ep in decoder_endpoints:
                    if ep.url == kv_url:
                        logger.info(f"Decode selected via kv-aware: {kv_url}")
                        return ep
                logger.warning(f"KV-aware returned stale decode endpoint: {kv_url}")
            except Exception as e:
                logger.warning(f"KV-aware routing failed for decode: {e}")

        try:
            prefix_url = await self._prefix_router.route_request(
                decoder_endpoints,
                engine_stats,
                request_stats,
                request,
                request_json,
            )
            ctx = getattr(request.state, "prefix_aware_ctx", None)
            if ctx and ctx.used_prefix_match:
                for ep in decoder_endpoints:
                    if ep.url == prefix_url:
                        logger.info(f"Decode selected via prefix-aware: {prefix_url}")
                        return ep
        except Exception as e:
            logger.warning(f"Prefix-aware routing failed for decode: {e}")

        # Fall back to round-robin
        sorted_endpoints = sorted(decoder_endpoints, key=lambda e: e.url)
        selected = sorted_endpoints[self.decode_idx % len(sorted_endpoints)]
        self.decode_idx += 1
        logger.info(f"Decode selected via round-robin: {selected.url}")
        return selected

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Fallback routing for models without prefill/decode endpoints.
        Delegates to the configured fallback router (session or roundrobin).
        P/D models are handled in route_orchestrated_disaggregated_request.
        """
        if not endpoints:
            raise ValueError("No endpoints available")
        return await self._fallback_router.route_request(
            endpoints, engine_stats, request_stats, request, request_json
        )


# Instead of managing a global _global_router, we can define the initialization functions as:
def initialize_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    if routing_logic == RoutingLogic.ROUND_ROBIN:
        logger.info("Initializing round-robin routing logic")
        router = RoundRobinRouter()
    elif routing_logic == RoutingLogic.SESSION_BASED:
        logger.info(f"Initializing session-based routing logic with kwargs: {kwargs}")
        router = SessionRouter(kwargs.get("session_key"))
    elif routing_logic == RoutingLogic.KVAWARE:
        logger.info("Initializing kvaware routing logic")
        router = KvawareRouter(
            lmcache_controller_port=kwargs.get("lmcache_controller_port"),
            session_key=kwargs.get("session_key"),
            kv_aware_threshold=kwargs.get("kv_aware_threshold"),
            lmcache_health_check_interval=kwargs.get("lmcache_health_check_interval"),
            lmcache_worker_timeout=kwargs.get("lmcache_worker_timeout"),
            lmcache_controller_reply_port=kwargs.get("lmcache_controller_reply_port"),
            lmcache_controller_heartbeat_port=kwargs.get(
                "lmcache_controller_heartbeat_port"
            ),
            tokenizer_model_names=kwargs.get("tokenizer_model_names"),
            kv_aware_per_model_thresholds=kwargs.get("kv_aware_per_model_thresholds"),
        )
        router.start_kv_manager()
    elif routing_logic == RoutingLogic.PREFIXAWARE:
        logger.info("Initializing prefix-aware routing logic")
        router = PrefixAwareRouter(
            prefix_aware_match_threshold=kwargs.get("prefix_aware_match_threshold", 0),
            prefix_aware_per_model_config=kwargs.get("prefix_aware_per_model_config"),
        )
    elif routing_logic == RoutingLogic.DISAGGREGATED_PREFILL:
        logger.info("Initializing disaggregated prefill routing logic")
        router = DisaggregatedPrefillRouter(
            kwargs.get("prefill_model_labels"),
            kwargs.get("decode_model_labels"),
            prefix_aware_match_threshold=kwargs.get("prefix_aware_match_threshold", 0),
            prefix_aware_per_model_config=kwargs.get("prefix_aware_per_model_config"),
            tokenizer_model_names=kwargs.get("tokenizer_model_names"),
            lmcache_controller_port=kwargs.get("lmcache_controller_port"),
            kv_aware_threshold=kwargs.get("kv_aware_threshold", 2000),
            kv_aware_per_model_thresholds=kwargs.get("kv_aware_per_model_thresholds"),
            lmcache_health_check_interval=kwargs.get(
                "lmcache_health_check_interval", 5
            ),
            lmcache_worker_timeout=kwargs.get("lmcache_worker_timeout", 30),
            lmcache_controller_reply_port=kwargs.get("lmcache_controller_reply_port"),
            lmcache_controller_heartbeat_port=kwargs.get(
                "lmcache_controller_heartbeat_port"
            ),
        )
    elif routing_logic == RoutingLogic.DISAGGREGATED_PREFILL_ORCHESTRATED:
        logger.info("Initializing disaggregated prefill orchestrated routing logic")
        return DisaggregatedPrefillOrchestratedRouter(
            kwargs.get("prefill_model_labels"),
            kwargs.get("decode_model_labels"),
            fallback_routing_logic=kwargs.get("fallback_routing_logic"),
            session_key=kwargs.get("session_key"),
            prefix_aware_match_threshold=kwargs.get("prefix_aware_match_threshold", 0),
            prefix_aware_per_model_config=kwargs.get("prefix_aware_per_model_config"),
            tokenizer_model_names=kwargs.get("tokenizer_model_names"),
            lmcache_controller_port=kwargs.get("lmcache_controller_port"),
            kv_aware_threshold=kwargs.get("kv_aware_threshold", 2000),
            kv_aware_per_model_thresholds=kwargs.get("kv_aware_per_model_thresholds"),
            lmcache_health_check_interval=kwargs.get(
                "lmcache_health_check_interval", 5
            ),
            lmcache_worker_timeout=kwargs.get("lmcache_worker_timeout", 30),
            lmcache_controller_reply_port=kwargs.get("lmcache_controller_reply_port"),
            lmcache_controller_heartbeat_port=kwargs.get(
                "lmcache_controller_heartbeat_port"
            ),
        )
    else:
        raise ValueError(f"Invalid routing logic {routing_logic}")

    router.max_instance_failover_reroute_attempts = kwargs.get(
        "max_instance_failover_reroute_attempts", 0
    )
    return router


def reconfigure_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    # Remove the existing routers from the singleton registry
    cleanup_routing_logic()
    return initialize_routing_logic(routing_logic, *args, **kwargs)


def get_routing_logic() -> RoutingInterface:
    # Look up in our singleton registry which router (if any) has been created.
    for cls in (
        SessionRouter,
        RoundRobinRouter,
        KvawareRouter,
        PrefixAwareRouter,
        DisaggregatedPrefillRouter,
        DisaggregatedPrefillOrchestratedRouter,
    ):
        if cls in SingletonABCMeta._instances:
            return cls()
    raise ValueError("The global router has not been initialized")


def cleanup_routing_logic():
    """Clean up all routing logic instances."""
    for cls in (
        SessionRouter,
        RoundRobinRouter,
        KvawareRouter,
        PrefixAwareRouter,
        DisaggregatedPrefillRouter,
        DisaggregatedPrefillOrchestratedRouter,
    ):
        if cls in SingletonABCMeta._instances:
            instance = cls()
            if hasattr(instance, "close"):
                instance.close()
            del SingletonABCMeta._instances[cls]
