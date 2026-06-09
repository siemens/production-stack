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

import asyncio
from typing import Generator, Optional, Set, Tuple

import xxhash

from vllm_router.log import init_logger

logger = init_logger(__name__)


class TrieNode:
    def __init__(self):
        self.children = {}
        self.endpoints = set()

        # assign a lock for each trie node.
        # this assures that each node will only be accessed by one co-routine
        # at a time.
        self.lock = asyncio.Lock()


class HashTrie:
    def __init__(self, chunk_size: int = 128):
        """
        Initialize the HashTrie.
        Args:
            chunk_size (int): the string chunk size (in terms of # characters)
        """
        self.root = TrieNode()
        self.chunk_size = chunk_size

    def _chunk_and_hash(self, request: str) -> Generator[Tuple[int, int], None, None]:
        """
        Yield per-character (hash, 1) chunks. This produces one trie node
        per character, enabling correct longest-prefix-match on variable-length
        chat/message payloads.
        """

        for char in request:
            yield xxhash.xxh64(char.encode("utf-8")).intdigest(), 1

    async def insert(self, request: str, endpoint: str) -> None:
        """
        Insert the request and endpoint into the trie.
        Args:
            request (str): The request to insert.
            endpoint (str): The endpoint to insert.
        """
        node = self.root
        async with node.lock:
            node.endpoints.add(endpoint)
        for chunk_hash, _ in self._chunk_and_hash(request):
            async with node.lock:
                if chunk_hash not in node.children:
                    node.children[chunk_hash] = TrieNode()
                node = node.children[chunk_hash]
            async with node.lock:
                node.endpoints.add(endpoint)

    async def longest_prefix_match(
        self,
        request: str,
        available_endpoints: Optional[Set[str]] = None,
    ) -> Tuple[int, Set[str]]:
        """
        Find the longest matching prefix using hashed chunks.
        Args:
            request (str): The request to find the longest matching prefix.
            available_endpoints (Optional[Set[str]]): The endpoints that are
                available. If None, no endpoints are considered and the match
                will always return an empty set.
        """
        node = self.root
        match_length = 0
        selected_endpoints = (
            available_endpoints.copy() if available_endpoints is not None else set()
        )

        for chunk_hash, chunk_len in self._chunk_and_hash(request):
            async with node.lock:
                node = node.children.get(chunk_hash)
            if not node:
                break
            async with node.lock:
                endpoints = node.endpoints.copy()
            intersection = endpoints.intersection(selected_endpoints)
            # reached longest prefix match in currently-available endpoints.
            if not intersection:
                break
            match_length += chunk_len
            selected_endpoints = intersection

        return match_length, selected_endpoints
