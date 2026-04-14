from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceRequest(_message.Message):
    __slots__ = ("json_body",)
    JSON_BODY_FIELD_NUMBER: _ClassVar[int]
    json_body: str
    def __init__(self, json_body: str | None = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("json_body",)
    JSON_BODY_FIELD_NUMBER: _ClassVar[int]
    json_body: str
    def __init__(self, json_body: str | None = ...) -> None: ...
