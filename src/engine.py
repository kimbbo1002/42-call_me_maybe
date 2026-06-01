from llm_sdk import Small_LLM_Model
import json
import numpy
from .models import load_input_files
from .models import Parameter, TestPromptSchema, OutputSchema, FunctionSchema


class Engine:
    def __init__(self):
        self.llm = Small_LLM_Model()
        self.functions = []

        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.vocab = dict(json.load(file))

    def select_function(self, prompt: str) -> str:
        supplied_prompt = (
            "User Prompt: " + prompt + f"\nThe function to call is: "
        )
        token_ids = self.llm.encode(supplied_prompt)
        logits = self.llm.get_logits_from_input_ids(token_ids)

        generated = ""
        while generated not in self.functions:
            for token_id_str, token_str in self.vocab.items():
                token_id = int(token_id_str)
                check = generated + token_str

                is_valid = any(fn.name.startswith(check) or check == fn.name for fn in self.functions)
                if not is_valid:
                    logits[token_id] = float('-inf')
            
            next_token_id = numpy.argmax(logits)
            next_token_str = self.vocab[str(next_token_id)]
            generated += next_token_str

            token_ids += [next_token_id]
            logits = self.llm.get_logits_from_input_ids(token_ids)
        return generated
    
    def get_parameters(self, prompt: str, fn: FunctionSchema) -> dict:
        param_prompt = ""
        for p_name, p_type in fn.parameters.items():
            param_prompt += f"{p_name} (type: {p_type.type})\n"

        supplied_prompt = (
            "User Prompt: " + prompt + f"\nCalled Function: {fn.name}\n"
            f"All Function Parameters: {param_prompt}\n"
        )
        params = {}
        
        for p_name, p_type in fn.parameters.items():
            supplied_prompt += f"{p_name} = "
            token_ids = self.llm.encode(supplied_prompt)
            logits = self.llm.get_logits_from_input_ids(token_ids)
            for token_id_str, token_str in self.vocab.items():
                token_id = int(token_id_str)
                if p_type.type == "number":
                    try:
                        int(token_str)
                    except:
                        logits[token_id] = float('-inf')
                elif p_type.type == "string":
                    try:
                        int(token_str)
                        logits[token_id] = float('-inf')
                    except:
                        pass
            param_id = numpy.argmax(logits)
            param_str = self.vocab[str(param_id)]
            params[p_name] = param_str
            supplied_prompt += f"{param_str}\n"
        
        return params
