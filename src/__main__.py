import argparse
import sys

from llm_sdk import Small_LLM_Model

from src.models import (
    load_function_definitions,
    load_prompts,
)

from src.generator import (
    FunctionCallingGenerator,
)

from src.utils import (
    write_results,
    Timer,
)


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default=(
            "data/input/functions_definition.json"
        ),
    )

    parser.add_argument(
        "--input",
        default=(
            "data/input/function_calling_tests.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/output/"
            "function_calling_results.json"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    try:

        functions = (
            load_function_definitions(
                args.functions_definition
            )
        )

        prompts = load_prompts(
            args.input
        )

        model = Small_LLM_Model()

        generator = (
            FunctionCallingGenerator(model)
        )

        timer = Timer()

        timer.start()

        results = []

        total_prompts = len(prompts)

        for index, prompt in enumerate(
            prompts,
            start=1,
        ):

            print(
                f"[{index}/{total_prompts}] "
                f"Processing: "
                f"{prompt.prompt}"
            )

            result = (
                generator.process_prompt(
                    prompt.prompt,
                    functions,
                )
            )

            print(
                f"    Function: "
                f"{result.name}"
            )

            print(
                f"    Parameters: "
                f"{result.parameters}"
            )

            results.append(result)

        write_results(
            args.output,
            results,
        )

        elapsed = timer.stop()

        print(
            f"Processing completed "
            f"in {elapsed:.2f} seconds"
        )

        print(
            f"Results written to "
            f"{args.output}"
        )

    except Exception as exc:

        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
