import json
import argparse
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Tuple


class TestPromptSchema(BaseModel):
    prompt: str


class Parameter(BaseModel):
    type: str


class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]


class OutputSchema(BaseModel):
    pass


def load_input_files(
) -> Tuple[List[FunctionSchema], List[TestPromptSchema], str, str]:
    parser = argparse.ArgumentParser()
    try:
        try:
            parser.add_argument("--function_defs",
                                default="data/input/functions_definition.json")
            parser.add_argument(
                "--input",
                default="data/input/function_calling_tests.json"
            )
            parser.add_argument(
                "--output",
                default="data/output/function_calling_results.json"
            )
            parser.add_argument(
                "--model",
                default="Qwen/Qwen3-0.6B"
            )
            args = parser.parse_args()
        except Exception as e:
            raise ValueError(e)

        try:
            with open(args.function_defs, "r") as file:
                functions_raw = json.load(file)
            functions = [FunctionSchema(**func) for func in functions_raw]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading function definitions: {e}")
        except ValidationError as e:
            raise ValueError(
                f"Function definition file failed schema validation: {e}"
            )

        try:
            with open(args.input, "r") as file:
                input_raw = json.load(file)
            prompts = [TestPromptSchema(**i) for i in input_raw]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading function definitions: {e}")
        except ValidationError as e:
            raise ValueError(
                f"Function definition file failed schema validation: {e}"
            )
    except Exception as e:
        raise ValueError(
            f"\033[0;31mPARSING ERROR:\033[0m {e}"
        )

    return functions, prompts, args.output, args.model
