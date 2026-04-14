# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import GrpcInferenceConfig


async def get_adapter_impl(config: GrpcInferenceConfig, _deps):
    from .grpc_adapter import GrpcInferenceAdapter

    assert isinstance(config, GrpcInferenceConfig), f"Unexpected config type: {type(config)}"
    impl = GrpcInferenceAdapter(config=config)
    await impl.initialize()
    return impl
