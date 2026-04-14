# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

import grpc

from llama_stack.log import get_logger
from llama_stack_api.inference.api import InferenceProvider
from llama_stack_api.inference.grpc.generated import inference_pb2, inference_pb2_grpc
from llama_stack_api.inference.models import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    RerankRequest,
)

logger = get_logger(name=__name__, category="server")


class InferenceServiceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer that wraps an InferenceProvider implementation."""

    def __init__(self, provider: InferenceProvider) -> None:
        self._provider = provider

    async def Completion(self, request: inference_pb2.InferenceRequest, context: grpc.aio.ServicerContext):
        params = OpenAICompletionRequestWithExtraBody.model_validate_json(request.json_body)
        logger.debug("gRPC Completion", model=params.model, stream=params.stream)

        result = await self._provider.openai_completion(params)
        if isinstance(result, AsyncIterator):
            async for chunk in result:
                yield inference_pb2.InferenceResponse(json_body=chunk.model_dump_json())
        else:
            yield inference_pb2.InferenceResponse(json_body=result.model_dump_json())

    async def ChatCompletion(self, request: inference_pb2.InferenceRequest, context: grpc.aio.ServicerContext):
        params = OpenAIChatCompletionRequestWithExtraBody.model_validate_json(request.json_body)
        logger.debug("gRPC ChatCompletion", model=params.model, stream=params.stream)

        result = await self._provider.openai_chat_completion(params)
        if isinstance(result, AsyncIterator):
            async for chunk in result:
                yield inference_pb2.InferenceResponse(json_body=chunk.model_dump_json())
        else:
            yield inference_pb2.InferenceResponse(json_body=result.model_dump_json())

    async def Embeddings(self, request: inference_pb2.InferenceRequest, context: grpc.aio.ServicerContext):
        params = OpenAIEmbeddingsRequestWithExtraBody.model_validate_json(request.json_body)
        logger.debug("gRPC Embeddings", model=params.model)

        result = await self._provider.openai_embeddings(params)
        return inference_pb2.InferenceResponse(json_body=result.model_dump_json())

    async def Rerank(self, request: inference_pb2.InferenceRequest, context: grpc.aio.ServicerContext):
        params = RerankRequest.model_validate_json(request.json_body)
        logger.debug("gRPC Rerank", model=params.model)

        result = await self._provider.rerank(params)
        return inference_pb2.InferenceResponse(json_body=result.model_dump_json())
