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
            f"File not found: {path}"
        )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON in {path}"
        ) from exc


def load_function_definitions(
    path: str,
) -> list[FunctionDefinition]:

    data = load_json_file(path)

    if not isinstance(data, list):
        raise ValueError(
            "functions_definition.json must contain a list"
        )

    functions = []

    for index, item in enumerate(data):

        try:

            functions.append(
                FunctionDefinition.model_validate(
                    item
                )
            )

        except Exception as exc:

            raise ValueError(
                f"Invalid function definition at index {index}"
            ) from exc

    return functions


def load_prompts(
    path: str,
) -> list[PromptInput]:

    data = load_json_file(path)

    if not isinstance(data, list):
        raise ValueError(
            "function_calling_tests.json must contain a list"
        )

    prompts = []

    for index, item in enumerate(data):

        try:

            prompts.append(
                PromptInput.model_validate(
                    item
                )
            )

        except Exception as exc:

            raise ValueError(
                f"Invalid prompt at index {index}"
            ) from exc

    return prompts
