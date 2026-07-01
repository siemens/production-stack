import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vllm_router.log import init_logger
from vllm_router.utils import AliasConfig

logger = init_logger(__name__)


@dataclass
class ExternalModelConfig:
    """A single external model configuration."""

    id: str
    type: str = "chat"
    aliases: list[AliasConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.aliases = [self._normalize_alias(alias) for alias in self.aliases]

    @staticmethod
    def _normalize_alias(alias: object) -> AliasConfig:
        if isinstance(alias, AliasConfig):
            return alias
        if isinstance(alias, str):
            return AliasConfig(model=alias)
        if isinstance(alias, dict):
            unknown_keys = set(alias) - {"name", "reasoning_effort"}
            if unknown_keys:
                raise ValueError(
                    "Invalid external model alias: unknown keys "
                    f"{sorted(unknown_keys)}. Supported keys: name, reasoning_effort"
                )
            name = alias.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Invalid external model alias: missing required key 'name'"
                )
            return AliasConfig(
                model=name,
                reasoning_effort=alias.get("reasoning_effort"),
            )
        raise TypeError(
            "Invalid external model alias: expected string or dict, "
            f"got {type(alias).__name__}"
        )

    def alias_names(self) -> list[str]:
        return [alias.model for alias in self.aliases]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExternalModelConfig":
        """Create an ExternalModelConfig from a dictionary (from YAML)."""
        return ExternalModelConfig(
            id=data["id"],
            type=data.get("type", "chat"),
            aliases=data.get("aliases", []),
        )


@dataclass
class ExternalProviderConfig:
    """A single external provider configuration."""

    name: str
    type: str
    api_base: str
    models: list[ExternalModelConfig] = field(default_factory=list)
    api_key_env_var: Optional[str] = None

    # Timeout in seconds applied to socket connect and per-read operations.
    # A total timeout is intentionally not set so long streaming responses are not cut off.
    timeout: float = 30.0
    max_retries: int = 3
    custom_headers: Dict[str, str] = field(default_factory=dict)

    def get_api_key(self, default_env_var: Optional[str] = None) -> Optional[str]:
        """Get the API key for this provider from environment variables."""
        env_var = self.api_key_env_var or default_env_var
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(
                    f"API key for provider '{self.name}' not found "
                    f"in environment variable '{env_var}'"
                )
            return api_key
        return None

    def get_all_model_ids(self) -> list[str]:
        """Get a list of all model IDs for this provider, including aliases."""
        model_ids = []
        for model in self.models:
            model_ids.append(model.id)
            model_ids.extend(model.alias_names())
        return model_ids

    def resolve_model_id(self, requested_model: str) -> Optional[str]:
        """Resolve a requested model name to a valid model ID for this provider."""
        for model in self.models:
            if requested_model == model.id or requested_model in model.alias_names():
                return model.id
        return None

    def resolve_alias_config(self, requested_model: str) -> Optional[AliasConfig]:
        """Resolve a requested alias to its optional request overrides."""
        for model in self.models:
            for alias in model.aliases:
                if requested_model == alias.model:
                    return alias
        return None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExternalProviderConfig":
        """Create an ExternalProviderConfig from a dictionary (from YAML)."""
        models = [ExternalModelConfig.from_dict(m) for m in data.get("models", [])]
        return ExternalProviderConfig(
            name=data["name"],
            type=data["type"],
            api_base=data["api_base"],
            models=models,
            api_key_env_var=data.get("api_key_env_var"),
            timeout=data.get("timeout", 30.0),
            max_retries=data.get("max_retries", 3),
            custom_headers=data.get("custom_headers", {}),
        )
