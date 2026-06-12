from llm_sdk.llm_sdk import Small_LLM_Model
import json
import os
import numpy as np
from .models import TestPromptSchema, FunctionSchema
from typing import List, Dict, Any


class Engine:
    def __init__(self, model: str) -> None:
        self.llm = Small_LLM_Model(model_name=model)
        self.functions: List[FunctionSchema] = []
        self.function_names: List[str] = []
        self.output_file = ""
        self.output: List[Dict[str, Any]] = []
        self.id_to_token: Dict[int, str] = {}

        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.token_to_id = json.load(file)
        self.id_to_token = {}
        for tkn, id in self.token_to_id.items():
            self.id_to_token[id] = tkn

    def find_function_by_name(self, fn_name: str) -> FunctionSchema | None:
        for fn in self.functions:
            if fn_name == fn.name:
                return fn
        return None

    def ft_decode(self, id: int) -> str:
        token_str = self.id_to_token[id]
        return token_str.replace('Ġ', ' ').replace('Ċ', '\n')

    def select_function(self, prompt: str) -> FunctionSchema | None:
        fn_defs = ""
        for i, fn in enumerate(self.functions):
            fn_defs += f"{i}: {fn.name}: {fn.description}\n"
        fn_prompt = (
            f"<|im_start|>system\n"
            f"You are a function selection assistant. "
            f"Choose the function that best matches the user's request.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Available functions:\n{fn_defs}\n"
            f"User request: {prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"The most appropriate function name is: "
        )

        token_ids = self.llm.encode(fn_prompt).tolist()[0]
        generated = ""

        while generated not in self.function_names:
            logits = self.llm.get_logits_from_input_ids(token_ids)

            masked_logits = np.full(len(logits), float('-inf'))
            for _, token_id in self.token_to_id.items():
                clean_token_str = self.ft_decode(token_id)
                for fn_name in self.function_names:
                    if fn_name.startswith(generated + clean_token_str):
                        masked_logits[token_id] = logits[token_id]
                        break

            next_token_id = int(np.argmax(masked_logits))
            next_token_str = self.ft_decode(next_token_id)
            print(next_token_str, end="", flush=True)
            token_ids.append(next_token_id)
            generated += next_token_str

        return self.find_function_by_name(generated)

    def select_parameter(
            self, prompt: str,
            fn: FunctionSchema
    ) -> Dict[str, Any]:
        param_defs = ""
        for p_name, p_type in fn.parameters.items():
            param_defs += f"{p_name} (type: {p_type.type})\n"
        param_prompt = (
            f"<|im_start|>system\nExtract parameter values "
            "exactly as they appear "
            f"in the user prompt. Preserve negative signs "
            "and exact values.<|im_end|>\n"
            f"<|im_start|>user\nFunction: {fn.name}\n"
            f"Parameters:\n{param_defs}\n"
            f"User prompt: {prompt}<|im_end|>\n"
            f"<|im_start|>assistant\nExtracted values:\n"
        )

        params: Dict[str, Any] = {}
        for p_name, p_type in fn.parameters.items():
            param_prompt += f"{p_name} = "
            token_ids = self.llm.encode(param_prompt).tolist()[0]
            generated = ""
            quote_char = None

            for _ in range(30):
                logits = self.llm.get_logits_from_input_ids(token_ids)

                masked_logits = np.full(len(logits), float('-inf'))
                for _, token_id in self.token_to_id.items():
                    token_str = self.ft_decode(token_id)
                    if p_type.type == "number":
                        if (
                            token_str.isdigit()
                            or token_str in [".", "-"]
                        ):
                            masked_logits[token_id] = logits[token_id]
                    else:
                        masked_logits[token_id] = logits[token_id]

                next_token_id = int(np.argmax(masked_logits))
                next_token_str = self.ft_decode(next_token_id)

                if "\n" in next_token_str:
                    break

                if p_type.type == "number":
                    if (
                        generated and "." in generated
                        and next_token_str.isdigit()
                    ):
                        generated += next_token_str
                        token_ids.append(next_token_id)
                        break
                elif p_type.type == "string":
                    if (
                        quote_char is None
                        and next_token_str in ["'", '"']
                    ):
                        quote_char = next_token_str
                        token_ids.append(next_token_id)
                        continue
                    elif (
                        quote_char
                        and next_token_str.strip() == quote_char
                    ):
                        break
                token_ids.append(next_token_id)
                generated += next_token_str

            param_prompt += generated + '\n'
            if p_type.type == "number":
                if f"-{generated}" in prompt:
                    params[p_name] = -1 * float(generated)
                else:
                    params[p_name] = float(generated)
            else:
                try:
                    int(generated)
                    if f"-{generated}" in prompt:
                        params[p_name] = -1 * int(generated)
                    else:
                        params[p_name] = int(generated)
                except ValueError:
                    params[p_name] = generated.strip().strip("'\"").strip()
        for k, v in params.items():
            print(f"{k}: {v}", flush=True)
        return params

    def generate_output(
            self, prompt: TestPromptSchema,
            fn: FunctionSchema, params: Dict[str, Any]
    ) -> None:
        prompt_output = {
            "prompt": prompt.prompt,
            "name": fn.name,
            "parameters": params
        }
        self.output.append(prompt_output)

    def write_output(self) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w") as file:
            json.dump(self.output, file, indent="\t")

    def start_engine(
            self, functions: List[FunctionSchema],
            prompts: List[TestPromptSchema],
            output_file: str
    ) -> None:
        try:
            print(
                f"\n\033[1;32m=== Launching Call-Me-Maybe "
                f"with {self.llm._model_name} ===\033[0m", end=""
            )
            self.functions = functions
            for fn in self.functions:
                self.function_names.append(fn.name)
            self.output_file = output_file

            for p in prompts:
                print(f"\n\n\033[1;34mPrompt:\033[0m {p.prompt}")
                print("\033[1;34mFunction:\033[0m", end="")
                func: FunctionSchema | None = self.select_function(p.prompt)
                if func:
                    print("\n\033[1;34mParams: \n\033[0m", end="")
                    params = self.select_parameter(p.prompt, func)
                    self.generate_output(p, func, params)
            self.write_output()
            print(
                f"\n\n\033[1;32m=== Call-Me-Maybe finished — "
                f"{len(prompts)} prompt(s) written to "
                f"{self.output_file} ===\033[0m\n"
            )
        except Exception as e:
            raise ValueError(
                f"\033[0;31mGENERATION ERROR:\033[0m {e}"
            )
