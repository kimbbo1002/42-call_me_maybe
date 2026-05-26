import json
import argparse
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List, Tuple


class TestPrompt:
    prompt: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, str]


def load_inpput_files() -> Tuple[List, List]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function_defs",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    args = parser.parse_args()

    try:
        with open(args.function_defs, "r") as file:
            functions_raw = json.load(file)
        tool_functions = [FunctionDef(**func) for func in functions_raw]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading function definitions: {e}")
    except ValidationError as e:
        raise ValueError(
            f"Function definition file failed schema validation: {e}"
        )
    
    try:
        with open(args.input, "r") as file:
            input_raw = json.load(file)
        prompts = [TestPrompt(**i) for i in input_raw]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading function definitions: {e}")
    except ValidationError as e:
        raise ValueError(
            f"Function definition file failed schema validation: {e}"
        )

    return tool_functions, prompts
