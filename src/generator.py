from src.parser import (
    FunctionDefinition,
    FunctionCall,
    ReturnDefinition,
)

from src.prompts import (
    build_function_prompt,
    build_parameter_prompt,
)

from src.utils import (
    PromptCache,
    ConstrainedDecoder,
    validate_function_name,
    validate_parameters,
)


class FunctionCallingGenerator:

    def __init__(self, model):

        self.model = model

        self.cache = PromptCache()

        self.decoder = ConstrainedDecoder(
            model
        )

        self.no_match_function = (
            FunctionDefinition(
                name="fn_no_match",
                description=(
                    "No matching function "
                    "for the request."
                ),
                parameters={},
                returns=ReturnDefinition(
                    type="null"
                ),
            )
        )

    def process_prompt(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionCall:

        available_functions = (
            functions + [self.no_match_function]
        )

        function_name = (
            self.select_function(
                prompt,
                available_functions,
            )
        )

        validate_function_name(
            function_name,
            available_functions,
        )

        if function_name == "fn_no_match":

            return FunctionCall(
                prompt=prompt,
                name=function_name,
                parameters={},
            )

        selected_function = next(
            function
            for function in functions
            if function.name == function_name
        )

        parameters = (
            self.extract_parameters(
                prompt,
                selected_function,
            )
        )

        validate_parameters(
            parameters,
            selected_function,
        )

        return FunctionCall(
            prompt=prompt,
            name=function_name,
            parameters=parameters,
        )

    def select_function(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> str:

        full_prompt = build_function_prompt(
            prompt,
            functions,
        )

        prompt_ids = self.cache.get_or_create(
            self.model,
            full_prompt,
        )

        choices = [
            function.name
            for function in functions
        ]

        return (
            self.decoder.decode_from_choices(
                prompt_ids,
                choices,
            )
        )

    def extract_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict:

        full_prompt = build_parameter_prompt(
            prompt,
            function,
        )

        prompt_ids = self.cache.get_or_create(
            self.model,
            full_prompt,
        )

        parameter_names = list(
            function.parameters.keys()
        )

        parameter_types = [
            parameter.type
            for parameter in (
                function.parameters.values()
            )
        ]

        decoded = (
            self.decoder.generate_parameter_values(
                prompt_ids,
                parameter_types,
            )
        )

        values = decoded.split("|")

        parameters = {}

        for index, param_name in enumerate(
            parameter_names
        ):

            if index >= len(values):
                break

            value = values[index].strip()

            param_type = (
                function.parameters[
                    param_name
                ].type
            )

            if param_type == "number":

                try:
                    value = float(value)

                except ValueError:
                    value = 0.0

            parameters[param_name] = value

        return parameters
