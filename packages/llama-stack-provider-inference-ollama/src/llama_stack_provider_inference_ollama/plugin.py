# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import InferenceProvider, ProviderPlugin

from .config import OllamaImplConfig
from .ollama import OllamaInferenceAdapter


class OllamaPlugin(ProviderPlugin[OllamaImplConfig]):
    """Ollama inference provider plugin for running local models."""

    def create(self) -> InferenceProvider:
        return OllamaInferenceAdapter(config=self.config)
