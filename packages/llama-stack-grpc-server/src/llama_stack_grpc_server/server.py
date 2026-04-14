# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Base gRPC provider server.

Loads any inference provider via entry-point discovery and serves it over gRPC.

Usage:
    llama-stack-grpc-server --config provider-server-config.yaml
"""

import argparse
import asyncio
import importlib
import importlib.metadata
import signal
from typing import Any

import grpc
import yaml

from llama_stack.log import get_logger
from llama_stack_api import Api, ProviderSpec, RemoteProviderSpec

from .config import GrpcProviderServerConfig

logger = get_logger(name=__name__, category="server")


def _register_servicer(server: grpc.Server, spec: ProviderSpec, provider: Any) -> None:
    """Register the appropriate gRPC servicer based on the provider's API."""
    if spec.api == Api.inference:
        from llama_stack_api.inference.grpc.generated import inference_pb2_grpc

        from .servicers.inference_servicer import InferenceServiceServicer

        servicer = InferenceServiceServicer(provider)
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    else:
        raise ValueError(f"No gRPC servicer available for API '{spec.api.value}'")


def _find_provider_spec(provider_type: str) -> ProviderSpec:
    """Look up a provider spec via entry points."""
    for ep in importlib.metadata.entry_points(group="llama_stack.providers"):
        try:
            get_spec = ep.load()
            spec_or_specs = get_spec()
            specs = spec_or_specs if isinstance(spec_or_specs, list) else [spec_or_specs]
            for spec in specs:
                if spec.provider_type == provider_type:
                    return spec
        except Exception:
            logger.warning("Failed to load provider entry point", entry_point=ep.name)

    raise ValueError(
        f"Failed to find provider spec for '{provider_type}'. "
        "Ensure the provider package is installed and registers an entry point."
    )


class _PassthroughModelStore:
    """Minimal model store that passes model IDs through without validation.

    The real model_store is injected by the routing table when running inside
    a full Llama Stack server. For standalone gRPC provider servers, this
    passthrough lets providers resolve model names directly.
    """

    async def has_model(self, identifier: str) -> bool:
        return False

    async def get_model(self, identifier: str):
        return None


async def _instantiate_provider(spec: ProviderSpec, provider_config: dict):
    """Instantiate a provider from its spec and config."""
    if spec.module is None:
        raise ValueError(f"Provider spec for '{spec.provider_type}' has no module defined")
    module = importlib.import_module(spec.module)

    config_type_path = spec.config_class
    module_path, class_name = config_type_path.rsplit(".", 1)
    config_module = importlib.import_module(module_path)
    config_type = getattr(config_module, class_name)
    config = config_type(**provider_config)

    if isinstance(spec, RemoteProviderSpec):
        method = "get_adapter_impl"
    else:
        method = "get_provider_impl"

    fn = getattr(module, method)
    impl = await fn(config, {})

    # model_store is normally injected by the routing table infrastructure.
    # Provide a passthrough implementation so providers work standalone.
    if not hasattr(impl, "model_store") or impl.model_store is None:
        impl.model_store = _PassthroughModelStore()

    return impl


async def serve(config: GrpcProviderServerConfig) -> None:
    """Start the gRPC server with the configured provider."""
    logger.info(
        "Starting gRPC provider server",
        provider_type=config.provider_type,
        port=config.port,
    )

    spec = _find_provider_spec(config.provider_type)
    logger.info("Found provider spec", provider_type=config.provider_type, module=spec.module)

    provider = await _instantiate_provider(spec, config.provider_config)
    logger.info("Provider instantiated", provider_type=config.provider_type)

    server = grpc.aio.server()
    _register_servicer(server, spec, provider)

    listen_addr = f"[::]:{config.port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info("gRPC server listening", address=listen_addr)

    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutting down gRPC server")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await shutdown_event.wait()
    await server.stop(grace=5)
    logger.info("gRPC server stopped")


def main():
    parser = argparse.ArgumentParser(description="Llama Stack gRPC Provider Server")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_config = yaml.safe_load(f)

    config = GrpcProviderServerConfig(**raw_config)
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
