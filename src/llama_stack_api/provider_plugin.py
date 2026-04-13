# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import typing
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from llama_stack_api import (
    Api,
    Batches,
    Connectors,
    Conversations,
    FileProcessors,
    Files,
    InferenceProvider,
    Messages,
    Prompts,
    Responses,
    Safety,
    ToolRuntime,
    VectorIO,
)

# Union of all provider protocol types — used for base class create() signature
type ProviderProtocol = (
    InferenceProvider
    | VectorIO
    | Safety
    | Responses
    | Batches
    | ToolRuntime
    | Files
    | FileProcessors
    | Messages
    | Connectors
    | Prompts
    | Conversations
)


class ProviderPlugin[ConfigT: BaseModel](ABC):
    """Minimal base class for type-safe provider plugins.

    Providers are distributed as standalone pip packages. All metadata
    is inferred to minimize boilerplate:

    - api: Inferred from entry point name
    - provider_type: Inferred from entry point name
    - pip_packages: Declared in pyproject.toml dependencies
    - description: Read from pyproject.toml package metadata
    - dependencies: Inferred from create() signature
    - config_class: Inferred from generic parameter ConfigT
    - provider_class: Inferred from create() return type annotation

    Example:
        # Entry point: "inference.remote.ollama"
        class OllamaPlugin(ProviderPlugin[OllamaConfig]):
            def create(self) -> InferenceProvider:
                return OllamaAdapter(self.config)

        # Entry point: "vector_io.inline.faiss"
        class FaissPlugin(ProviderPlugin[FaissConfig]):
            def create(
                self,
                inference: InferenceProvider,      # Required (no default)
                files: Files | None = None,        # Optional (has default)
            ) -> VectorIO:
                return FaissVectorIO(self.config, inference, files)
    """

    # Auto-extracted metadata (set in __init_subclass__)
    config_class: ClassVar[type[BaseModel]]
    provider_class: ClassVar[type]

    def __init__(self, config: ConfigT) -> None:
        """Initialize plugin with configuration.

        Subclasses do not need to override __init__ — just declare config type
        in the Generic parameter and the base class handles initialization.
        """
        self.config: ConfigT = config

    @abstractmethod
    def create(self, **kwargs: ProviderProtocol | None) -> ProviderProtocol:
        """Create provider implementation with API dependencies.

        Pure factory method — returns the provider instance. The resolver
        handles async lifecycle (calling initialize() on the returned impl).

        Concrete implementations override this with explicit typed parameters.
        Dependencies are inferred from the overridden method's signature.
        Core resolver validates signature and injects dependencies by name at runtime.

        Parameter names must match Api enum values (e.g., 'inference' -> Api.inference).
        Parameters without defaults are required; with defaults are optional.
        All parameters must have type annotations.

        The return type annotation on the concrete implementation determines
        provider_class — it is validated at class definition time by __init_subclass__.

        Examples:
            # No dependencies
            def create(self) -> InferenceProvider:
                return MyAdapter(self.config)

            # One required dependency
            def create(self, inference: InferenceProvider) -> VectorIO:
                return MyVectorDB(self.config, inference)

            # Mixed required and optional
            def create(
                self,
                inference: InferenceProvider,      # Required (no default)
                files: Files | None = None,        # Optional (has default)
            ) -> VectorIO:
                return MyVectorDB(self.config, inference, files)
        """
        ...

    @classmethod
    def get_dependencies(cls) -> tuple[list[Api], list[Api]]:
        """Extract API dependencies from create() method signature.

        Dependencies are inferred from create() parameters:
        - Parameter name must match Api enum value (e.g., "inference" -> Api.inference)
        - No default value = required dependency
        - Has default value = optional dependency

        Returns:
            Tuple of (required_apis, optional_apis)
        """
        sig = inspect.signature(cls.create)
        required: list[Api] = []
        optional: list[Api] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            try:
                api = Api[param_name]
            except KeyError:
                raise TypeError(
                    f"{cls.__name__}.create() parameter '{param_name}' does not match any Api enum value"
                ) from None

            if param.default is inspect.Parameter.empty:
                required.append(api)
            else:
                optional.append(api)

        return required, optional

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-extract metadata from generic type parameters and validate."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract intermediate classes
        if ABC in cls.__bases__:
            return

        # 1. Extract ConfigT from generic type parameter
        config_class = None
        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__origin__") and base.__origin__ is ProviderPlugin:
                args = typing.get_args(base)
                if len(args) == 1:
                    config_class = args[0]
                    break

        if config_class is None:
            raise TypeError(f"{cls.__name__} must specify a config type parameter: ProviderPlugin[YourConfig]")

        # 2. Validate config_class is a Pydantic BaseModel
        if not isinstance(config_class, type):
            raise TypeError(f"{cls.__name__}: ConfigT must be a class, got {config_class}")

        if not issubclass(config_class, BaseModel):
            raise TypeError(f"{cls.__name__}: ConfigT must be a Pydantic BaseModel subclass, got {config_class}")

        cls.config_class = config_class

        # 3. Infer provider_class from create() return type annotation
        sig = inspect.signature(cls.create)

        if sig.return_annotation == inspect.Signature.empty:
            raise TypeError(f"{cls.__name__}.create() must have a return type annotation")

        cls.provider_class = sig.return_annotation

        # Validate all parameters (except 'self') have type hints and valid names
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            if param.annotation == inspect.Parameter.empty:
                raise TypeError(f"{cls.__name__}.create() parameter '{param_name}' must have a type hint")

            try:
                Api[param_name]
            except KeyError:
                raise TypeError(
                    f"{cls.__name__}.create() parameter '{param_name}' "
                    f"does not match any Api enum value. "
                    f"Valid API names: {[a.value for a in Api]}"
                ) from None
