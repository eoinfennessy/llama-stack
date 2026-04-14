# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field


class GrpcInferenceConfig(BaseModel):
    """Configuration for connecting to a gRPC inference provider server."""

    host: str = Field(
        default="localhost",
        description="Hostname of the gRPC provider server.",
    )
    port: int = Field(
        default=50051,
        description="Port of the gRPC provider server.",
    )
    use_tls: bool = Field(
        default=False,
        description="Whether to use TLS for the gRPC connection.",
    )
