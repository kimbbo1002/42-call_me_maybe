from llm_sdk.llm_sdk import Small_LLM_Model
import json
import os
import numpy
from .models import load_input_files
from .models import Parameter, TestPromptSchema, OutputSchema, FunctionSchema
from typing import Dict, List


class Engine:
    def __init__(self):
        self.llm = Small_LLM_Model()
        self.functions = []
        self.output_file = ""
        self.output: List[dict] = []

        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.vocab = json.load(file)
        self.id_to_token = {}
        for token, id in self.vocab.items():
            self.id_to_token[id] = token

    def select_function(self, prompt: str) -> str:
        fn_defs = ""
        for i, fn in enumerate(self.functions):
            fn_defs += f"{i}: {fn.name}: {fn.description}\n"
        fn_prompt = (
            f"Functions:\n{fn_defs}\n"
            f"User Prompt: {prompt}\n"
            f"Which function index best matches the user prompt?"
            f"Answer with a single digit. The answer is: "
        )
        token_ids_list = self.llm.encode(fn_prompt).tolist()[0]
        logits = self.llm.get_logits_from_input_ids(token_ids_list)

        valid_indices = {str(i) for i in range(len(self.functions))}
        for token_str, token_id_str in self.vocab.items():
            if token_str not in valid_indices:
                logits[int(token_id_str)] = float('-inf')
        for token_id in range(len(logits)):
            if token_id not in self.id_to_token:
                logits[token_id] = float('-inf')
        next_token_id = int(numpy.argmax(logits))
        next_token_str = self.id_to_token[next_token_id]
        return self.functions[int(next_token_str)].name

    def get_parameters(self, prompt: str, fn: FunctionSchema | None) -> dict:
        if not fn:
            return {}
        
        param_defs = ""
        for p_name, p_type in fn.parameters.items():
            param_defs += f"{p_name} (type: {p_type.type})\n"
        param_prompt = (
            f"User Prompt: {prompt}\n"
            f"Called Function: {fn.name}\n"
            f"All Function Parameters:\n{param_defs}\n"
            f"Extract the parameter values from the user prompt.\n"
        )

        params = {}

        for p_name, p_type in fn.parameters.items():
            param_prompt += f"{p_name} = "
            token_ids_list = self.llm.encode(param_prompt).tolist()[0]
            generated = ""
            n_decimals = 0

            while True:
                logits = self.llm.get_logits_from_input_ids(token_ids_list)

                for token_str, token_id_str in self.vocab.items():
                    token_id = int(token_id_str)
                    if p_type.type == "integer":
                        if not token_str.isdigit():
                            logits[token_id] = float('-inf')
                    elif p_type.type == "boolean":
                        if token_str not in ["True", "False"]:
                            logits[token_id] = float('-inf')
                    elif p_type.type == "number":
                        is_valid = all(c.isdigit() or c == '.' for c in token_str.strip())
                        if not is_valid or token_str.strip() == '':
                            logits[token_id] = float('-inf')
                    elif p_type.type == "string":
                        if 'Ċ' in token_str or '=' in token_str:
                            logits[token_id] = float('-inf')
                
                for token_id in range(len(logits)):
                    if token_id not in self.id_to_token:
                        logits[token_id] = float('-inf')
                
                next_token_id = int(numpy.argmax(logits))
                next_token_str = self.id_to_token[next_token_id]
                if next_token_str == '.':
                    n_decimals += 1

                if p_type.type == "number":
                    if not all(c.isdigit() or c == '.' for c in next_token_str.strip()):
                        break
                    if generated and '.' in generated and '.' not in next_token_str:
                        generated += next_token_str
                        token_ids_list.append(next_token_id)
                        break
                    if '.' in generated and '.' in next_token_str:
                        break
                elif p_type.type == "string":
                    if 'Ċ' in next_token_str or next_token_str == 'Ġ':
                        break
                    if '"' in next_token_str and '"' in generated:
                        generated += next_token_str
                        token_ids_list.append(next_token_id)
                        break
                    if "'" in next_token_str and "'" in generated:
                        generated += next_token_str
                        token_ids_list.append(next_token_id)
                        break
                elif p_type.type == "boolean":
                    break
                elif p_type.type == "integer":
                    if not next_token_str.isdigit():
                        break
                if len(generated) > len(prompt) / 2:
                    break
                
                generated += next_token_str
                if n_decimals == 1 and not next_token_str == '.':
                    n_decimals += 2
                token_ids_list.append(next_token_id)
            value = generated.replace("Ġ", " ").replace("Ċ", "\n").strip()
            if (
                value.startswith("'") and value.endswith("'")
                or value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]
            params[p_name] = value
            print(f"{p_name}: {params[p_name]}")
            param_prompt += f"{generated.strip()}\n"
        
        return params

    def get_output(self, prompt: str, selected_fn: FunctionSchema, params: dict) -> dict:
        return {"prompt": prompt.prompt, "name": selected_fn.name, "parameters": params}

    def write_output(self) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w") as file:
            json.dump(self.output, file, indent="\t")

    def start_sim(self, functions: List[FunctionSchema], prompts: List[TestPromptSchema], output_file: str) -> None:
        # load prompts and functions
        self.output_file = output_file
        self.functions = functions

        # execute each prompt and generate output
        for prompt in prompts:
            fn = self.select_function(prompt.prompt)
            print(f"\n\nprompt = {prompt.prompt}")
            print(f"selected function : {fn}\n")
            selected_fn: FunctionSchema | None = None
            for function in functions:
                if fn == function.name:
                    selected_fn = function
                    break
            params = self.get_parameters(prompt.prompt, selected_fn)
            self.output.append(self.get_output(prompt, selected_fn, params))
        
        self.write_output()
