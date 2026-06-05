from .engine import Engine
from .models import load_input_files


def main() -> None:
    functions, prompts, output_file = load_input_files()
    engine = Engine()
    engine.start_sim(functions, prompts, output_file)


if __name__ == "__main__":
    main()
