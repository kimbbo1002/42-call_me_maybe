from .engine import Engine
from .models import load_input_files


def main() -> None:
    try:
        functions, prompts, output_file, model = load_input_files()
        engine = Engine(model)
        engine.start_engine(functions, prompts, output_file)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
