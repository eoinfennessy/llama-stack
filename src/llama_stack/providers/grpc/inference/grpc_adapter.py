# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

import grpc

from llama_stack.log import get_logger
from llama_stack.providers.grpc.inference.config import GrpcInferenceConfig
from llama_stack_api.inference.grpc.generated import inference_pb2, inference_pb2_grpc
from llama_stack_api.inference.models import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankRequest,
    RerankResponse,
)

logger = get_logger(name=__name__, category="providers")


class GrpcInferenceAdapter:
    """Inference adapter that delegates to a remote gRPC provider server."""

    def __init__(self, config: GrpcInferenceConfig) -> None:
        self.config = config
        self._channel: grpc.aio.Channel | None = None
        self._stub: inference_pb2_grpc.InferenceServiceStub | None = None

    async def initialize(self) -> None:
        target = f"{self.config.host}:{self.config.port}"
        if self.config.use_tls:
            self._channel = grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
        else:
            self._channel = grpc.aio.insecure_channel(target)
        self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)
        logger.info("gRPC inference adapter initialized", target=target, tls=self.config.use_tls)

    async def shutdown(self) -> None:
        if self._channel:
            await self._channel.close()

    @property
    def stub(self) -> inference_pb2_grpc.InferenceServiceStub:
        assert self._stub is not None, "Adapter not initialized — call initialize() first"
        return self._stub

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        request = inference_pb2.InferenceRequest(json_body=params.model_dump_json())

        if params.stream:

            async def _stream() -> AsyncIterator[OpenAICompletion]:
                async for resp in self.stub.Completion(request):
                    yield OpenAICompletion.model_validate_json(resp.json_body)

            return _stream()

        # Non-streaming: collect the single response from the server-streaming RPC
        result: OpenAICompletion | None = None
        async for resp in self.stub.Completion(request):
            result = OpenAICompletion.model_validate_json(resp.json_body)
        assert result is not None, "Server returned no response for non-streaming completion"
        return result

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        request = inference_pb2.InferenceRequest(json_body=params.model_dump_json())

        if params.stream:

            async def _stream() -> AsyncIterator[OpenAIChatCompletionChunk]:
                async for resp in self.stub.ChatCompletion(request):
                    yield OpenAIChatCompletionChunk.model_validate_json(resp.json_body)

            return _stream()

        result: OpenAIChatCompletion | None = None
        async for resp in self.stub.ChatCompletion(request):
            result = OpenAIChatCompletion.model_validate_json(resp.json_body)
        assert result is not None, "Server returned no response for non-streaming chat completion"
        return result

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        request = inference_pb2.InferenceRequest(json_body=params.model_dump_json())
        resp = await self.stub.Embeddings(request)
        return OpenAIEmbeddingsResponse.model_validate_json(resp.json_body)

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        grpc_request = inference_pb2.InferenceRequest(json_body=request.model_dump_json())
        resp = await self.stub.Rerank(grpc_request)
        return RerankResponse.model_validate_json(resp.json_body)
