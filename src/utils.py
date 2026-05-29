import json
import math
import time

from src.parser import (
    FunctionDefinition,
)


class PromptCache:

    def __init__(self):

        self.cache: dict[str, list[int]] = {}

    def get_or_create(
        self,
        model,
        text: str,
    ) -> list[int]:

        if text not in self.cache:

            encoded = model.encode(text)

            self.cache[text] = (
                encoded[0].tolist()
            )

        return self.cache[text]


class ConstraintManager:

    def __init__(self, model):

        self.model = model

    def build_token_paths(
        self,
        values: list[str],
    ) -> dict[str, list[int]]:

        token_paths = {}

        for value in values:

            token_ids = self.model.encode(
                value
            )[0].tolist()

            token_paths[value] = token_ids

        return token_paths

    def get_allowed_tokens(
        self,
        generated: list[int],
        token_paths: dict[str, list[int]],
    ) -> set[int]:

        allowed = set()

        for token_path in token_paths.values():

            prefix = token_path[
                :len(generated)
            ]

            if prefix == generated:

                if len(generated) < len(token_path):

                    next_token = token_path[
                        len(generated)
                    ]

                    allowed.add(next_token)

        return allowed


class ConstrainedDecoder:

    def __init__(self, model):

        self.model = model

        self.constraints = ConstraintManager(
            model
        )

    def decode_from_choices(
        self,
        prompt_ids: list[int],
        choices: list[str],
        max_tokens: int = 30,
    ) -> str:

        token_paths = (
            self.constraints.build_token_paths(
                choices
            )
        )

        generated = []

        for _ in range(max_tokens):

            logits = (
                self.model.get_logits_from_input_ids(
                    prompt_ids + generated
                )
            )

            allowed_tokens = (
                self.constraints.get_allowed_tokens(
                    generated,
                    token_paths,
                )
            )

            if not allowed_tokens:
                break

            masked_logits = [
                -math.inf
                for _ in range(len(logits))
            ]

            for token_id in allowed_tokens:

                masked_logits[token_id] = (
                    logits[token_id]
                )

            next_token = max(
                range(len(masked_logits)),
                key=lambda idx: masked_logits[idx],
            )

            generated.append(next_token)

            for value, token_path in (
                token_paths.items()
            ):

                if generated == token_path:
                    return value

        raise ValueError(
            "No valid generation"
        )

    def generate_parameter_values(
        self,
        prompt_ids: list[int],
        parameter_types: list[str],
        max_tokens: int = 60,
    ) -> str:

        generated = []

        current_parameter_index = 0

        while len(generated) < max_tokens:

            logits = (
                self.model.get_logits_from_input_ids(
                    prompt_ids + generated
                )
            )

            current_type = (
                parameter_types[
                    min(
                        current_parameter_index,
                        len(parameter_types) - 1,
                    )
                ]
            )

            allowed_tokens = set()

            for token_id in range(len(logits)):

                token = self.model.decode(
                    [token_id]
                )

                if current_type == "number":

                    valid_chars = (
                        "0123456789.-|"
                    )

                    if all(
                        char in valid_chars
                        for char in token
                    ):
                        allowed_tokens.add(token_id)

                else:

                    clean_token = token.replace(
                        "\n",
                        ""
                    )

                    forbidden_patterns = [
                        "Output:",
                        "Example:",
                        "Examples:",
                        "Input:",
                    ]

                    if (
                        len(clean_token) > 0
                        and not any(
                            pattern in clean_token
                            for pattern in forbidden_patterns
                        )
                    ):
                        allowed_tokens.add(token_id)

            masked_logits = [
                -math.inf
                for _ in range(len(logits))
            ]

            for token_id in allowed_tokens:

                masked_logits[token_id] = (
                    logits[token_id]
                )

            next_token = max(
                range(len(masked_logits)),
                key=lambda idx: masked_logits[idx],
            )

            generated.append(next_token)

            decoded = self.model.decode(
                generated
            )

            if "\n" in decoded:

                decoded = decoded.split(
                    "\n"
                )[0]

                break

            current_parameter_index = (
                decoded.count("|")
            )

            if (
                current_parameter_index
                >= len(parameter_types)
            ):
                break

        return self.model.decode(
            generated
        ).strip()


def validate_function_name(
    function_name: str,
    functions: list[FunctionDefinition],
) -> None:

    valid_names = [
        function.name
        for function in functions
    ]

    if function_name not in valid_names:

        raise ValueError(
            f"Invalid function: "
            f"{function_name}"
        )


def validate_parameters(
    parameters: dict,
    function: FunctionDefinition,
) -> None:

    required = set(
        function.parameters.keys()
    )

    given = set(
        parameters.keys()
    )

    if required != given:

        raise ValueError(
            "Invalid parameters"
        )


def write_results(
    path: str,
    results,
) -> None:

    from pathlib import Path

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                result.model_dump()
                for result in results
            ],
            file,
            indent=2,
            ensure_ascii=False,
        )


class Timer:

    def __init__(self):

        self.start_time = 0.0

    def start(self):

        self.start_time = time.time()

    def stop(self) -> float:

        return (
            time.time() - self.start_time
        )
