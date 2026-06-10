from llm_sdk.llm_sdk import Small_LLM_Model
import json
import os
import numpy as np
from .models import TestPromptSchema, FunctionSchema
from typing import List, Dict, Any


class Engine:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()
        self.functions: List[FunctionSchema] = []
        self.function_names: List[str] = []
        self.output_file = ""
        self.output: List[Dict[str, Any]] = []

        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.token_to_id = json.load(file)
        self.id_to_token = {}
        for tkn, id in self.token_to_id.items():
            self.id_to_token[id] = tkn
        print(len(self.token_to_id))
        print(len(self.id_to_token))

    def find_function_by_name(self, fn_name: str) -> FunctionSchema | None:
        for fn in self.functions:
            if fn_name == fn.name:
                return fn
        return None

    def select_function(self, prompt: str) -> FunctionSchema | None:
        fn_defs = ""
        for i, fn in enumerate(self.functions):
            fn_defs += f"{i}: {fn.name}: {fn.description}\n"
        fn_prompt = (
            f"Functions:\n{fn_defs}\n"
            f"User Prompt: {prompt}\n"
            f"Which function best matches the user prompt?"
            f"The answer is: "
        )

        token_ids = self.llm.encode(fn_prompt).tolist()[0]
        logits = self.llm.get_logits_from_input_ids(token_ids)
        generated = ""
        while generated not in self.function_names:
            logits_cpy = logits.copy()

            # constrained decoding to get function names
            for token_id, token_str in self.id_to_token.items():
                logits[token_id] = float('-inf')
                for fn_name in self.function_names:
                    if fn_name.startswith(generated + token_str):
                        logits[token_id] = logits_cpy[token_id]
                        break
            masked_logits = np.full(len(logits), float('-inf'))
            for token_id in self.id_to_token:
                masked_logits[token_id] = logits[token_id]
            logits = list(masked_logits)

            # adding to tokens and recalculating logits
            next_token_id = np.argmax(logits)
            next_token_str = self.id_to_token[next_token_id]
            generated += next_token_str
            token_ids.append(next_token_id)
            logits = self.llm.get_logits_from_input_ids(token_ids)

        return self.find_function_by_name(generated)

    def select_parameter(
            self, prompt: str,
            fn: FunctionSchema
    ) -> Dict[str, Any]:
        param_defs = ""
        for p_name, p_type in fn.parameters.items():
            param_defs += f"{p_name} (type: {p_type.type})\n"
        param_prompt = (
            f"User prompt: {prompt}\n"
            f"Called Function: {fn.name}\n"
            f"All Function Parameters:\n{param_defs}\n"
            f"Extract the parameter values from the user prompt.\n"
        )

        params: Dict[str, Any] = {}
        for p_name, p_type in fn.parameters.items():
            param_prompt += f"{p_name} = "
            token_ids = self.llm.encode(param_prompt).tolist()[0]
            logits = self.llm.get_logits_from_input_ids(token_ids)
            generated = ""
            quote_char = None

            for _ in range(30):

                masked_logits = np.full(len(logits), float('-inf'))
                for token_id in self.id_to_token:
                    masked_logits[token_id] = logits[token_id]
                logits = list(masked_logits)

                # constrained decoding for types of parameters
                for token_id, token_str in self.id_to_token.items():

                    if p_type.type == "number":
                        if (
                            not token_str.isdigit()
                            and token_str not in [".", "-"]
                        ):
                            logits[token_id] = float('-inf')

                next_token_id = np.argmax(logits)
                next_token_str = self.id_to_token[next_token_id]

                if "Ċ" in next_token_str:
                    break

                if p_type.type == "number":
                    if (
                        generated and "." in generated
                        and next_token_str.isdigit()
                    ):
                        generated += next_token_str
                        token_ids.append(next_token_id)
                        break
                if p_type.type == "string":
                    if (
                        quote_char is None
                        and next_token_str.replace('Ġ', '') in ("'", '"')
                    ):
                        quote_char = next_token_str.replace('Ġ', '')
                        token_ids.append(next_token_id)
                        logits = self.llm.get_logits_from_input_ids(token_ids)
                        continue
                    if (
                        (quote_char and next_token_str.strip() == quote_char)
                        or 'Ċ' in next_token_str
                    ):
                        break

                generated += next_token_str.replace('Ġ', ' ')
                token_ids.append(next_token_id)
                logits = self.llm.get_logits_from_input_ids(token_ids)

            param_prompt += f"{repr(generated)}\n"
            if p_type.type == "number":
                params[p_name] = float(generated)
            elif generated.isdigit():
                params[p_name] = int(generated)
            else:
                params[p_name] = generated

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
            self.functions = functions
            for fn in self.functions:
                self.function_names.append(fn.name)
            self.output_file = output_file

            for p in prompts:
                func: FunctionSchema | None = self.select_function(p.prompt)
                if func:
                    print(f"\n\nPrompt: {p.prompt}")
                    print(f"Function: {fn.name}")
                    params = self.select_parameter(p.prompt, fn)
                    print(f"Params: {params}")
                    self.generate_output(p, fn, params)
            self.write_output()
        except Exception as e:
            raise ValueError(
                f"\033[0;31mGENERATION ERROR:\033[0m {e}"
            )
