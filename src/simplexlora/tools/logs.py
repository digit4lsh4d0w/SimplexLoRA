from __future__ import annotations

import argparse
from pathlib import Path


def filter_log_file(input_file, output_file):
    """
    Фильтрует строки из файла, удаляя те, которые начинаются с указанных префиксов.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str): Путь к выходному файлу
    """

    # Список префиксов для удаления
    prefixes_to_remove = [
        "{'loss':",
        "{'eval_loss':",
        "{'train_runtime':",
        "  active_adapters",
        "New chosen layers:",
        "self.lora_ranks:",
        "We will use the GPU:",
        "model is",
        "trainable params",
        "[H[JWe",
        "        ",
    ]

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            with open(output_file, "w", encoding="utf-8") as outfile:
                for line in infile:
                    # Проверяем, начинается ли строка с любого из префиксов
                    if not any(line.startswith(prefix) for prefix in prefixes_to_remove):
                        outfile.write(line)

        print("Файл успешно обработан!")
        print(f"Исходный файл: {input_file}")
        print(f"Результирующий файл: {output_file}")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SimplexLoRA training logs")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    filter_log_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
