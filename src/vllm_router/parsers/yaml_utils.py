import json
from typing import Any

import yaml

from vllm_router.log import init_logger
from vllm_router.utils import AliasConfig

logger = init_logger(__name__)


def generate_static_backends(models: dict[str, Any]) -> str:
    static_backends = []
    for _, details in models.items():
        if "static_backends" in details:
            static_backends.extend(details["static_backends"])
    return ",".join(static_backends)


def generate_static_models(models: dict[str, Any]) -> str:
    static_models = []
    for name, details in models.items():
        if "static_backends" in details:
            model_name = details.get("model_name", name)
            static_models.extend([model_name] * len(details["static_backends"]))
    return ",".join(static_models)


_VALID_ALIAS_DICT_KEYS = {"model", "reasoning_effort"}


def _parse_alias_config(alias: str, config: Any) -> AliasConfig:
    if isinstance(config, str):
        return AliasConfig(model=config)
    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid alias config for '{alias}': expected string or dict, got {type(config).__name__}"
        )
    if "model" not in config:
        raise ValueError(f"Alias '{alias}' is missing required key 'model'")

    unknown_keys = set(config.keys()) - _VALID_ALIAS_DICT_KEYS
    if unknown_keys:
        raise ValueError(
            f"Alias '{alias}' contains unknown keys: {sorted(unknown_keys)}. "
            f"Supported keys: {sorted(_VALID_ALIAS_DICT_KEYS)}"
        )

    return AliasConfig(
        model=config["model"],
        reasoning_effort=config.get("reasoning_effort"),
    )


def generate_static_aliases(aliases: dict[str, Any]) -> str:
    parts = []
    for alias, raw_config in aliases.items():
        config = _parse_alias_config(alias, raw_config)
        entry = f"{alias}:{config.model}"
        if config.reasoning_effort is not None:
            entry += f"|reasoning_effort={config.reasoning_effort}"
        parts.append(entry)
    return ",".join(parts)


def generate_static_model_labels(models: dict[str, Any]) -> str:
    static_model_labels = []
    for _, details in models.items():
        if "static_backends" in details:
            label = details.get("model_label", "default")
            static_model_labels.extend([label] * len(details["static_backends"]))
    return ",".join(static_model_labels)


def generate_static_model_types(models: dict[str, Any]) -> str:
    static_model_types = []
    for _, details in models.items():
        if "static_model_type" in details and "static_backends" in details:
            static_model_types.extend(
                [details["static_model_type"]] * len(details["static_backends"])
            )
    return ",".join(static_model_types)


def generate_static_healthcheck_disabled(models: dict[str, Any]) -> str:
    healthcheck_disabled = []
    for _, details in models.items():
        if "static_backends" in details:
            disabled = str(details.get("healthcheck_disabled", False)).lower()
            healthcheck_disabled.extend([disabled] * len(details["static_backends"]))
    return ",".join(healthcheck_disabled)


def generate_static_endpoint_prefixes(models: dict[str, Any]) -> str:
    endpoint_prefixes = []
    for _, details in models.items():
        if "static_backends" in details:
            prefixes = details.get("endpoint_prefixes", {})
            endpoint_prefixes.extend([prefixes] * len(details["static_backends"]))
    return json.dumps(endpoint_prefixes)


def read_and_process_yaml_config_file(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        try:
            yaml_config = yaml.safe_load(f)
            if not yaml_config:
                return {}
            models = yaml_config.pop("static_models", None)
            aliases = yaml_config.pop("static_aliases", None)
            if models:
                yaml_config["static_backends"] = generate_static_backends(models)
                yaml_config["static_models"] = generate_static_models(models)
                yaml_config["static_model_labels"] = generate_static_model_labels(models)
                yaml_config["static_model_types"] = generate_static_model_types(models)
                yaml_config["static_healthcheck_disabled"] = (
                    generate_static_healthcheck_disabled(models)
                )
                yaml_config["static_endpoint_prefixes"] = (
                    generate_static_endpoint_prefixes(models)
                )
            if aliases:
                yaml_config["static_aliases"] = generate_static_aliases(aliases)
            return yaml_config
        except (yaml.YAMLError, AttributeError) as e:
            logger.error(f"Error loading YAML config file: {e}")
            raise
