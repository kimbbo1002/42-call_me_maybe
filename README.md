_This project has been created as part of the 42 curriculum by \<bokim\>._

# Call-Me-Maybe
Introduction to function calling in LLMs

## Description
### Overview
This project is about creating a function calling tool that translates natural language prompts into structured function calls.

For example, given a question such as "What is the sum of 40 and 2?", instead of returning 42, it provides the function name(`fn_add_numbers`) and its arguments(`{"a": 40, "b": 2}`) in a JSON format. 

With **constrained decoding** this project guarantees 100% valid JSON output, and ensures near-perfect reliability even with a small 0.6B parameter model.

## Instructions
### Makefile Commands
Manage project setup, execution, quality control, and cleanup using the provided shortcuts:
| Command | Description |
| :--- | :--- |
| `make install` | Synchronizes the virtual environment and dependencies using `uv`. |
| `make run` | Runs the main application package (`src/`) with configured JSON paths. |
| `make debug` | Launches the execution script inside Python's interactive debugger (`pdb`). |
| `make lint` | Validates standard formatting using `flake8` and runs basic `mypy` typing checks. |
| `make lint-strict` | Executes a strict, zero-tolerance `mypy` configuration over the codebase. |
| `make clean` | Wipes temporary build caches (`.mypy_cache`) and recursive `__pycache__` directories. |
| `make fclean` | Calls `clean`, then completely destroys the `.venv`, lockfile, and generated outputs. |

### Example Usage
#### 1. Quick Start (Default Execution)

To set up your workspace and immediately process the test suite using the default files, run the following sequential commands in your terminal:
```bash
# Step 1: Install dependencies and sync the virtual environment
make install

# Step 2: Run the function extraction pipeline
make run
```
When you execute `make run`, the system maps to the following default configuration values:
- **Function Definitions:** data/input/functions_definition.json

- **Input Prompts:** data/input/function_calling_tests.json

- **Output Path:** data/output/function_calls.json

#### 2. Custom Data Execution
If you want to run the pipeline with different configurations without altering the Makefile, you can override the file path variables directly in your command line:
```bash
make run FUNC="data/custom/my_functions.json" INPUT="data/custom/my_prompts.json" OUTPUT="data/custom/results.json"
```

## Technical Implementations
### Algorithm Explanation
The `Engine` class implements a structured, deterministic **Function Calling (Tool Use)** system on top of a small Language Model (LLM). Instead of relying on the LLM to generate unstructured text and hoping it conforms to a specific schema, this engine forces syntactic and semantic validity via **Logit Masking** during the decoding phase.

The core algorithm operates in two primary sequential phases:
#### **1. Constrained Function Selection** (`select_function`)
- The system constructs a custom prompt containing all available function definitions.
- During autoregressive token generation, a vocabulary mask is dynamically calculated.
- For every vocabulary token, the engine checks if appending its decoded string matches the prefix of any valid function name.
- Logits of invalid tokens are suppressed to $-\infty$, ensuring the LLM only selects a function that explicitly exists in `self.function_names`.

#### **2. Type-Constrained Parameter Extraction** (`select_parameter`)
- Once a function is chosen, the engine iterates through its expected parameters.
- For each parameter, it reads the schema's type constraints (`number`, `string`, etc.).
- While decoding the parameter value, a character-level validation loop suppresses tokens that violate the type constraints (e.g., rejecting non-digit characters when a `number` is expected).
- It also handles simple boundary conditions, such as breaking the loop upon encountering closing quotation marks for strings or trailing newline tokens.

### Design Decisions
- **Inversion of Vocabulary Mapping:** During initialization, the engine reverses the token-to-ID vocabulary dictionary loaded using `get_path_to_vocab_file` from the LLM. This reversed dictionary (`self.id_to_token`) enables quick $O(1)$ lookups during decoding loops, minimizing the perfomrance footprint of the string-decoding step.
- **Byte-level Replacements for Token Space**(`ft_decode`): Instead of using the `decode` function from the LLM, the engine specifically handles BPE tokenizer artifacts like `Ġ` (spaces) and `Ċ` (newlines). This maps raw token bytes directly back to plain-text strings.
- **Fallthrough Typing:** Parameter casting uses a strict cascading hierarchy based on the schema definition. If the parameter is explicitly typed as a `number`, it is directly cast to a `float`. For all other types, the engine attempts to proactively cast the value to an `int`; if a `ValueError` is caught, it falls back to a string, safely stripping out bounding quotation marks (`'`, `"`) and residual trailing whitespace.

### Performance Analysis
#### Efficiency Bottlenecks & Design Trade-offs
- **Token-by-Token Iterative Validation ($O(N \times V)$ Complexity):** During the decoding phase, the engine runs a manual Python loop over the entire vocabulary dictionary (`for _, token_id in self.token_to_id.items()`) for every single generated token. While this nested linear scan introduces a CPU-bound bottleneck for standard tokenizers (vocabulary sizes ($V$) of 32,000 to 50,000+ tokens), **this was an intentional design choice to guarantee absolute control over string boundaries and regex filtering without introducing complex external dependencies or compiling heavy C-extensions during early-stage development.**
  
- **Redundant Forward Passes (Autoregressive Overhead):** Because token generation occurs inside a standard `while` loop without KV-caching, the engine passes the growing sequence of `token_ids` back into `self.llm.get_logits_from_input_ids()` on every iteration. **This stateless approach was intentionally chosen to match the design philosophy of edge-capability models like Qwen-0.6B. At a 0.6B parameter scale, the computational overhead of prompt prefill is trivial compared to the engineering complexity and memory overhead of tracking KV-caches. Prioritizing a stateless architecture ensures ultra-lightweight portability across constrained edge environments while maintaining structural simplicity.**
### Testing Strategy
The evaluation pipeline systematically verifies function routing and parameter extraction accuracy using the data driven directly by the files inside `data/input/`:
- **Schema Validation Testing:** The framework uses `pydantic` schemas (`FunctionSchema` and `TestPromptSchema`) to validate that the function definitions (`functions_definition.json`) and test queries (`function_calling_tests.json`) match the required layout before execution begins.
- **Functional Routing Matrix:** The engine is tested against explicit user prompts targeting distinct tools (such as routing mathematical queries like "What is the sum of 2 and 3?" directly to `fn_add_numbers`, or text manipulation prompts to `fn_reverse_string`).
- **Multi-Parameter Extraction Boundaries:** Test cases containing multiple required arguments (such as extracting parameters `source_string`, `regex`, and `replacement` for `fn_substitute_string_with_regex`) ensure the constrained-decoding logic isolates multiple distinct variables cleanly without leaking context boundaries.
- **End-to-End Pipeline Verification:** The framework checks that the complete workflow—from parsing inputs to dumping results into `data/output/function_calling_results.json`—runs successfully without crashing or throwing formatting errors.

## Project Review
### Challenges Faced
- **Conceptual Learning Curve:** Navigating the foundational mechanics of generative AI, specifically mastering how **tokens** map to vocabulary matrices and how raw **logits** are manipulated, was critical before implementing structured decoding logic.
- **Mitigating Model Hallucination**:The `Small_LLM_Model` frequently hallucinated trailing explanations (e.g., repeating the prompt) rather than strictly outputting raw function arguments. Python's `repr()` function on data strings completely neutralized model rambling by introducing:
  - **Clear Boundaries:** Turning a string like `Hello "world"` into `'"Hello "world""'` gives the small LLM clear, un-ignorable starting and ending markers, anchoring its focus.
  - **Safe Characters:** Converting raw newlines (`\n`) or tabs (`\t`) into literal text characters (`\` and `n`) stops the model from treating them as structural formatting commands, which previously triggered it to write full paragraphs.

### Future Improvements
- **Scaling LLM Capability:** Achieving deterministic, flawless parameters was limited by the reasoning capacity of the **Qwen3-0.6B** base model. Complex tasks, such as generating arguments for `fn_substitute_string_with_regex`, frequently degraded due to the model's small parameter size. Upgrading to a slightly larger or fine-tuned alternative would dramatically increase extraction accuracy.

## Resources
### AI Usage
- **Overcoming Hallucinations with `repr()`:** Leveraged AI insights to implement Python's repr() function on data strings.
- **Documentation Refinement:** Employed AI workflows for technical copywriting, structural layout organization, and paraphrasing the final `README.md`.