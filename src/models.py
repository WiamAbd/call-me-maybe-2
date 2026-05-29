import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    type: str


class ReturnDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptInput(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


def load_json_file(path: str) -> Any:

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_function_definitions(
    path: str,
) -> list[FunctionDefinition]:

    data = load_json_file(path)

    return [
        FunctionDefinition.model_validate(item)
        for item in data
    ]


def load_prompts(
    path: str,
) -> list[PromptInput]:

    data = load_json_file(path)

    return [
        PromptInput.model_validate(item)
        for item in data
    ]
