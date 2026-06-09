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
### Design Decisions
### Performance Analysis
### Testing Strategy

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