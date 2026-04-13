# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import inspect
from typing import Any

from llama_stack.core.client import get_client_impl
from llama_stack.core.datatypes import (
    AccessRule,
    Provider,
    StackConfig,
)
from llama_stack.core.distribution import builtin_automatically_routed_apis, plugin_registry
from llama_stack.core.external import load_external_apis
from llama_stack.log import get_logger
from llama_stack_api import (
    LLAMA_STACK_API_V1ALPHA,
    Admin,
    Api,
    Batches,
    Connectors,
    Conversations,
    ExternalApiSpec,
    FileProcessors,
    Files,
    Inference,
    InferenceProvider,
    Inspect,
    Messages,
    Models,
    ModelsProtocolPrivate,
    Prompts,
    ProviderSpec,
    RemoteProviderConfig,
    Responses,
    Safety,
    Shields,
    ShieldsProtocolPrivate,
    ToolGroups,
    ToolGroupsProtocolPrivate,
    ToolRuntime,
    VectorIO,
    VectorStore,
)
from llama_stack_api import (
    Providers as ProvidersAPI,
)

logger = get_logger(name=__name__, category="core")


class InvalidProviderError(Exception):
    """Raised when a provider is invalid or has been deprecated with an error."""

    pass


def api_protocol_map(external_apis: dict[Api, ExternalApiSpec] | None = None) -> dict[Api, Any]:
    """Get a mapping of API types to their protocol classes.

    Args:
        external_apis: Optional dictionary of external API specifications

    Returns:
        Dictionary mapping API types to their protocol classes
    """
    protocols = {
        Api.admin: Admin,
        Api.providers: ProvidersAPI,
        Api.responses: Responses,
        Api.inference: Inference,
        Api.inspect: Inspect,
        Api.batches: Batches,
        Api.vector_io: VectorIO,
        Api.vector_stores: VectorStore,
        Api.models: Models,
        Api.safety: Safety,
        Api.shields: Shields,
        Api.tool_groups: ToolGroups,
        Api.tool_runtime: ToolRuntime,
        Api.files: Files,
        Api.prompts: Prompts,
        Api.conversations: Conversations,
        Api.file_processors: FileProcessors,
        Api.connectors: Connectors,
        Api.messages: Messages,
    }

    if external_apis:
        for api, api_spec in external_apis.items():
            try:
                module = importlib.import_module(api_spec.module)
                api_class = getattr(module, api_spec.protocol)

                protocols[api] = api_class
            except (ImportError, AttributeError):
                logger.exception("Failed to load external API", api_name=api_spec.name)

    return protocols


def api_protocol_map_for_compliance_check(config: Any) -> dict[Api, Any]:
    """Get the API-to-protocol mapping used for provider compliance checks.

    Args:
        config: Stack configuration for loading external APIs.

    Returns:
        Dictionary mapping APIs to their protocol classes, with InferenceProvider replacing Inference.
    """
    external_apis = load_external_apis(config)
    return {
        **api_protocol_map(external_apis),
        Api.inference: InferenceProvider,
    }


def additional_protocols_map() -> dict[Api, Any]:
    """Get the mapping of APIs to their additional private protocol classes for routing table support.

    Returns:
        Dictionary mapping router APIs to tuples of (private_protocol, public_protocol, routing_table_api).
    """
    return {
        Api.inference: (ModelsProtocolPrivate, Models, Api.models),
        Api.tool_groups: (ToolGroupsProtocolPrivate, ToolGroups, Api.tool_groups),
        Api.safety: (ShieldsProtocolPrivate, Shields, Api.shields),
    }


# TODO: make all this naming far less atrocious. Provider. ProviderSpec. ProviderWithSpec. WTF!
class ProviderWithSpec(Provider):
    """A Provider paired with its resolved ProviderSpec for instantiation."""

    spec: ProviderSpec


ProviderRegistry = dict[Api, dict[str, ProviderSpec]]


async def resolve_impls(
    run_config: StackConfig,
    provider_registry: ProviderRegistry,
    dist_registry: Any,
    policy: list[AccessRule],
    internal_impls: dict[Api, Any] | None = None,
) -> dict[Api, Any]:
    """Resolve and instantiate all provider implementations.

    Two-phase approach:
    1. Instantiate all user-facing providers (remote/inline) via ProviderPlugin
    2. Wire up routing infrastructure (routing tables + auto-routers) directly
    """
    impls: dict[Api, Any] = internal_impls.copy() if internal_impls else {}

    # Phase 1: Instantiate user-facing providers in dependency order
    sorted_providers = _prepare_and_sort_providers(run_config, provider_registry)
    for provider in sorted_providers:
        deps = _resolve_deps(provider, impls)
        impl = await _instantiate_provider(provider, deps, run_config)
        impls[provider.spec.api] = impl

    # Phase 2: Wire up routing infrastructure
    await _wire_routing(impls, dist_registry, run_config, policy)

    return impls


def _prepare_and_sort_providers(
    run_config: StackConfig,
    provider_registry: ProviderRegistry,
) -> list[ProviderWithSpec]:
    """Validate, resolve specs, and sort providers in dependency order."""
    providers: list[ProviderWithSpec] = []

    for api_str, configured_providers in run_config.providers.items():
        api = Api(api_str)
        for provider in configured_providers:
            if not provider.provider_id or provider.provider_id == "__disabled__":
                continue

            if provider.provider_type not in provider_registry.get(api, {}):
                raise ValueError(f"Provider `{provider.provider_type}` is not available for API `{api}`")

            spec = provider_registry[api][provider.provider_type]
            if spec.deprecation_error:
                raise InvalidProviderError(spec.deprecation_error)
            if spec.deprecation_warning:
                logger.warning(
                    "Provider is deprecated",
                    provider_type=provider.provider_type,
                    warning=spec.deprecation_warning,
                )

            providers.append(ProviderWithSpec(spec=spec, **provider.model_dump()))

    return _topological_sort(providers)


def _topological_sort(providers: list[ProviderWithSpec]) -> list[ProviderWithSpec]:
    """Sort providers so dependencies are instantiated first."""
    by_api: dict[str, ProviderWithSpec] = {p.spec.api.value: p for p in providers}
    visited: set[str] = set()
    result: list[ProviderWithSpec] = []

    def visit(api_str: str) -> None:
        if api_str in visited or api_str not in by_api:
            return
        visited.add(api_str)
        provider = by_api[api_str]
        for dep in provider.spec.api_dependencies:
            visit(dep.value)
        for dep in provider.spec.optional_api_dependencies:
            visit(dep.value)
        result.append(provider)

    for api_str in by_api:
        visit(api_str)

    return result


def _resolve_deps(provider: ProviderWithSpec, impls: dict[Api, Any]) -> dict[Api, Any]:
    """Resolve required and optional dependencies for a provider."""
    deps: dict[Api, Any] = {}
    for api in provider.spec.api_dependencies:
        if api not in impls:
            raise RuntimeError(
                f"Failed to resolve '{provider.spec.api.value}' provider '{provider.provider_id}': "
                f"required dependency '{api.value}' is not available."
            )
        deps[api] = impls[api]
    for api in provider.spec.optional_api_dependencies:
        if api in impls:
            deps[api] = impls[api]
    return deps


async def _instantiate_provider(
    provider: ProviderWithSpec,
    deps: dict[Api, Any],
    run_config: StackConfig,
) -> Any:
    """Instantiate a provider via its ProviderPlugin."""
    plugin_cls = plugin_registry.get(provider.spec.provider_type)
    if plugin_cls is None:
        raise ValueError(
            f"No ProviderPlugin registered for '{provider.spec.provider_type}'. "
            f"All providers must be distributed as packages with a ProviderPlugin entry point."
        )

    config = plugin_cls.config_class(**provider.config)
    plugin = plugin_cls(config)

    dep_kwargs: dict[str, Any] = {}
    required_apis, optional_apis = plugin_cls.get_dependencies()
    for api in required_apis:
        dep_kwargs[api.name] = deps[api]
    for api in optional_apis:
        if api in deps:
            dep_kwargs[api.name] = deps[api]

    impl = plugin.create(**dep_kwargs)
    if hasattr(impl, "initialize"):
        await impl.initialize()

    object.__setattr__(impl, "__provider_id__", provider.provider_id)
    object.__setattr__(impl, "__provider_spec__", provider.spec)
    object.__setattr__(impl, "__provider_config__", config)

    protocols = api_protocol_map_for_compliance_check(run_config)
    additional_protocols = additional_protocols_map()
    check_protocol_compliance(impl, protocols[provider.spec.api])
    if provider.spec.api in additional_protocols:
        additional_api, _, _ = additional_protocols[provider.spec.api]
        check_protocol_compliance(impl, additional_api)

    return impl


async def _wire_routing(
    impls: dict[Api, Any],
    dist_registry: Any,
    run_config: StackConfig,
    policy: list[AccessRule],
) -> None:
    """Wire up routing tables and auto-routers for routed APIs.

    This is infrastructure, not provider instantiation — routing tables and
    routers are fixed implementations that delegate to the actual providers.
    """
    from llama_stack.core.routers import get_auto_router_impl, get_routing_table_impl

    for info in builtin_automatically_routed_apis():
        if info.router_api not in impls:
            continue

        # Collect provider impls for this routed API
        inner_impls = dict(_collect_provider_impls(impls, info.router_api))

        # Build routing table (e.g., models, shields)
        routing_table = await get_routing_table_impl(info.routing_table_api, inner_impls, impls, dist_registry, policy)
        impls[info.routing_table_api] = routing_table

        # Build auto-router (e.g., inference, safety)
        str_deps = {api.value: impl for api, impl in impls.items()}
        router = await get_auto_router_impl(info.router_api, routing_table, str_deps, run_config, policy)
        impls[info.router_api] = router


def _collect_provider_impls(impls: dict[Api, Any], api: Api) -> list[tuple[str, Any]]:
    """Collect all provider implementations for a routed API."""
    impl = impls.get(api)
    if impl is None:
        return []
    provider_id = getattr(impl, "__provider_id__", api.value)
    return [(provider_id, impl)]


def check_protocol_compliance(obj: Any, protocol: Any) -> None:
    """Verify that a provider implementation correctly implements all required protocol methods.

    Args:
        obj: The provider implementation to check.
        protocol: The protocol class defining required methods.

    Raises:
        ValueError: If the provider is missing required methods or has signature mismatches.
    """
    missing_methods = []

    mro = type(obj).__mro__
    for name, value in inspect.getmembers(protocol):
        if inspect.isfunction(value) and hasattr(value, "__webmethods__"):
            has_alpha_api = False
            for webmethod in value.__webmethods__:
                if webmethod.level == LLAMA_STACK_API_V1ALPHA:
                    has_alpha_api = True
                    break
            # if this API has multiple webmethods, and one of them is an alpha API, this API should be skipped when checking for missing or not callable routes
            if has_alpha_api:
                continue
            if not hasattr(obj, name):
                missing_methods.append((name, "missing"))
            elif not callable(getattr(obj, name)):
                missing_methods.append((name, "not_callable"))
            else:
                # Check if the method signatures are compatible
                obj_method = getattr(obj, name)
                proto_sig = inspect.signature(value)
                obj_sig = inspect.signature(obj_method)

                proto_params = set(proto_sig.parameters)
                proto_params.discard("self")
                obj_params = set(obj_sig.parameters)
                obj_params.discard("self")
                if not (proto_params <= obj_params):
                    logger.error(
                        "Method signature incompatible", method=name, proto_params=proto_params, obj_params=obj_params
                    )
                    missing_methods.append((name, "signature_mismatch"))
                else:
                    # Check if the method has a concrete implementation (not just a protocol stub)
                    # Find all classes in MRO that define this method
                    method_owners = [cls for cls in mro if name in cls.__dict__]

                    # Allow methods from mixins/parents, only reject if ONLY the protocol defines it
                    if len(method_owners) == 1 and method_owners[0].__name__ == protocol.__name__:
                        # Only reject if the method is ONLY defined in the protocol itself (abstract stub)
                        missing_methods.append((name, "not_actually_implemented"))

    if missing_methods:
        raise ValueError(
            f"Provider `{obj.__provider_id__} ({obj.__provider_spec__.api})` does not implement the following methods:\n{missing_methods}"
        )


async def resolve_remote_stack_impls(
    config: RemoteProviderConfig,
    apis: list[str],
) -> dict[Api, Any]:
    """Resolve provider implementations for a remote stack by creating API clients.

    Args:
        config: Remote provider configuration containing the connection URL.
        apis: List of API names to resolve.

    Returns:
        Dictionary mapping APIs to their remote client implementations.
    """
    protocols = api_protocol_map()
    additional_protocols = additional_protocols_map()

    impls = {}
    for api_str in apis:
        api = Api(api_str)
        impls[api] = await get_client_impl(
            protocols[api],
            config,
            {},
        )
        if api in additional_protocols:
            _, additional_protocol, additional_api = additional_protocols[api]
            impls[additional_api] = await get_client_impl(
                additional_protocol,
                config,
                {},
            )

    return impls
