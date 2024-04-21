"""Golden тесты транслятора и машины.

Конфигурационнфе файлы: "golden/*.yml"
"""

import contextlib
import io
import logging
import os
import pytest
import simulator
import tempfile
import translator


@pytest.mark.golden_test("golden/*.yml")
def test_translator_and_machine(golden, caplog):
    """
    Принцип работы следующий: во внешних файлах специфицируются входные и
    выходные данные для теста. При запуске тестов происходит сравнение и если
    выход изменился -- выводится ошибка.

    Если вы меняете логику работы приложения -- то запускаете тесты с ключом:
    `cd python && poetry run pytest . -v --update-goldens`

    Это обновит файлы конфигурации, и вы можете закоммитить изменения в
    репозиторий, если они корректные.

    Формат файла описания теста -- YAML. Поля определяются доступом из теста к
    аргументу `golden` (`golden[key]` -- входные данные, `golden.out("key")` --
    выходные данные).

    Вход:

    - `in_source` -- исходный код
    - `in_stdin` -- данные на ввод процессора для симуляции
    - `in_logging_level` -- уровень логирования журнала симулятора
    - `in_debug_level` -- уровень детализации отладочного журнала

    Выход:

    - `out_code` -- машинный код и данные, сгенерированный транслятором
    - `out_stdout` -- стандартный вывод транслятора и симулятора
    - `out_log` -- журнал симулятора
    """

    # Создаём временную папку для тестирования приложения.
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Готовим имена файлов для входных и выходных данных.
        source_name = os.path.join(tmpdirname, "source.lsp")
        input_name = os.path.join(tmpdirname, "input.txt")
        binary_name = os.path.join(tmpdirname, "target.bin")
        memory_name = os.path.join(tmpdirname, "target.dat")
        debug_name = os.path.join(tmpdirname, "target.dbg")

        # Записываем входные данные в файлы. Данные берутся из теста.
        with open(source_name, "w", encoding="utf-8") as file:
            file.write(golden["in_source"])
        with open(input_name, "w", encoding="utf-8") as file:
            file.write(golden["in_stdin"])

        logging_level = golden["in_logging_level"]
        debug_level = int(golden["in_debug_level"])

        # Установим уровень отладочного вывода на DEBUG
        if logging_level == "info":
            caplog.set_level(logging.INFO)
        else:
            caplog.set_level(logging.DEBUG)

        # Запускаем транслятор и симулятор и собираем весь
        # стандартный вывод в переменную stdout
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            translator.translate(source_name, binary_name, memory_name, debug_name)
            print("============================================================")
            simulator.simulate(binary_name, memory_name, input_name, debug_name, 0, logging_level, debug_level)

        # Выходные данные также считываем в переменные.
        with open(debug_name, encoding="utf-8") as file:
            debug_output = file.read()

        # Проверяем, что ожидания соответствуют реальности.
        assert debug_output == golden.out["out_code"]
        assert stdout.getvalue() == golden.out["out_stdout"]
        assert caplog.text == golden.out["out_log"]
