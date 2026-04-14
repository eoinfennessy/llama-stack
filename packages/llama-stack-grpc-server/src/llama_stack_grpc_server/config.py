# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field


class GrpcProviderServerConfig(BaseModel):
    """Configuration for the gRPC provider server."""

    provider_type: str = Field(
        ...,
        description="The provider type to load, e.g. 'remote::ollama' or 'inline::transformers'.",
    )
    port: int = Field(
        default=50051,
        description="Port to serve gRPC on.",
    )
    provider_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configuration passed to get_adapter_impl or get_provider_impl.",
    )
