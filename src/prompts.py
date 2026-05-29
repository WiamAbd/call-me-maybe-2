from src.models import FunctionDefinition


FUNCTION_SELECTION_PROMPT = '''
You are a function selection engine.

Choose the SINGLE BEST matching function.

If no function matches semantically,
select fn_no_match.

Return ONLY the function name.

No explanations.
No extra text.
'''


PARAMETER_EXTRACTION_PROMPT = '''
You are a parameter extraction engine.

Extract parameter values for the selected function.

Rules:
- Return ONLY parameter values
- Separate values using |
- Respect parameter types
- No explanations
- No extra text

Examples:

Input:
What is the sum of 2 and 3?
Output:
2|3

Input:
Greet john
Output:
john
'''


def build_function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:

    lines = [
        FUNCTION_SELECTION_PROMPT,
        "",
        f"User Request: {user_prompt}",
        "",
        "Available Functions:",
    ]

    for function in functions:

        lines.append(
            f"- {function.name}: "
            f"{function.description}"
        )

    lines.append("")
    lines.append(
        "Return only the function name."
    )

    return "\n".join(lines)


def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:

    lines = [
        PARAMETER_EXTRACTION_PROMPT,
        "",
        f"User Request: {user_prompt}",
        "",
        f"Selected Function: {function.name}",
        "",
        f"Description: {function.description}",
        "",
        "Required Parameters:",
    ]

    for param_name, param in (
        function.parameters.items()
    ):

        lines.append(
            f"- {param_name}: {param.type}"
        )

    lines.append("")
    lines.append(
        "Return parameter values separated by |"
    )

    return "\n".join(lines)
