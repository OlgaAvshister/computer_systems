# Авшистер Ольга Аркадьевна P3213

`lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob1`

Без усложнения

## Язык программирования

Описание синтаксиса языка в форме БНФ:

```ebnf
digit := '0' | '1' | '2' | ... | '9'
letter := 'a' | 'b' | 'c' | ... 'z'

number := '-'? digit*
string := '"' <any symbol except a double quote> '"'
name := letter*
operation := '+' | '-' | '*' | '/' | '%' | '=' | '!=' | '<' | '>' | '<=' | '>='

expression := 
    number | 
    string |
    name | 
    operation |
    '(' expression* ')'

program := expression*
```

Программы пишутся в виде s-выражений, как в Lisp/Scheme.

Пример программы:

```lisp
(defun prob (count result)
       (if (< count 1000)
           (if (= (% count 3) 0)
               (prob (+ count 1) (+ result count))
               (if (= (% count 5) 0)
                   (prob (+ count 1) (+ result count))
                   (prob (+ count 1) result)))
           result))

(printnumber (prob 1 0))
```

Специальные формы, существующие в языке:

* `do` позволяет выполнить несколько выражений:

        (do expr1 expr2 expr3)

* `if` позволяет сделать ветвление:

        (if cond pos-branch neg-branch)

* `define` объявляет переменную:

        (define name value)

* `defun` объявляет функцию:

        (defun name (arg1 arg2 arg3) body)
    
Язык поддерживает всего два типа данных:

1. Целые числа 
2. Строки 

Числа -- целые 32-битные: от -2147483648 до 2147483647.

Результат всех математических операций будет "обрезан" до младших 32-бит без завершения проргаммы.

Операция деления `/` при делении на ноль завершает программу с ошибкой.

Результат операций сравнения возвращается как число `1` или `0`.

Отдельные символы строки имеет тип целого числа.  

Поддерживаются строковые литералы произвольной длины. В конце строки ставится символ-ноль `\0`, в соответствие с вариантом. 

Область видимости переменных -- локальная в рамках функций и частей `if`.

Все функции должны обладать уникальными именами, не совпадающими с именами переменных в данной области видимости.

Функции имеют доступ только к своим локальным переменным и параметрам. Глобальные переменные поддерживаются, однако доступ к ним имеет только код, не находящийся ни в каких функциях.

Стратегия вычислений -- строгая, аппликативный порядок. То есть, все вычисления происходят в том порядке, в каком они даны в программе. Параметры функций вычисляются до вызова функции, слева направо.

Поддерживается рекурсия.

Язык предоставляет следующие встроенные функции:

* `getchar` получение i-ого символа из строки:

         (getchar "hello" 0)

* `setchar` меняет i-ый символ в строке:

         (setchar str i sym)

* `readchar` читает один символ из потока ввода
* `printchar` печатает один символ в поток вывода
* `printnumber` выводит число на экран

         (printnumber 123)

* `printstring` выводит строку на экран

         (printstring "hello")


## Организация памяти

Система построена по гарвардской архитектуре, то есть память разделена на:

1. память инструкций
2. память данных

Обе памяти работают в линейном, плоском адресном пространстве.

### Регистры

Система обладает аккумуляторной архитектурой. Поэтому большинство операций происходит над аккумулятором. Остальные регистры устанавливаются процессором.

```text
       Registers
+------------------------------+
| acc                          |
| flags                        |
| sp                           |
| da                           |
| pc                           |
+------------------------------+
```

Все регистры -- 32-битные. За исключением регистра `flags`, который хранит результаты сравнения (3 бита).

Никаике переменные не отображаются на регистры, так как по сути нам доступен всего один регистр -- аккумулятор

### Память инструкций

Размер машинного слова -- 48 бит.

Одна инструкция занимает одно машинной слово: 2 байта используется для кода операции, и 4 байта используется для операнда.

Единственный вариант адресации -- прямая адресация. Адрес для памяти данных может быть использован только напрямую в инструкциях перехода и в инструкции `CALL`. Например,

    JMP 100
    
Первым в памяти инструкций располагается главный код программы, далее по порядку располагаются тела всех функций.
   
```
       Instruction memory
+------------------------------+
| 00  : program body           |
|    ...                       |
|  i  : function body          |
|  j  : function body          |
|  k  : function body          |
|    ...                       |
+------------------------------+
```

Память инструкций доступна только для чтения, а именно для выполнения программы из неё и переходов по этой программе.

### Память данных

Размер машинного слова -- 32 бит.

Одно число занимает одно машинное слово.

Один символ также занимает одно машинное слово.

Символы строки хранятся как отдельные машинные слова, включая специальный символ-ноль `\0`.

Варианты адресации:

* прямая адресация, например,

        LD [1000]

* базовая адресация со смещением, например,

        LD [SP + 100]
    
* косвенная адресация 2-ого порядка (разымёнывание указателя):
    
        LD [[SP + 100]]

Структура памяти:

```text
          Data memory
+--------------------------------------+
| 00  : input port reserved address    |
| 01  : output port reserved address   |
| 02  : string literal 1 char 1        |
| 03  : string literal 1 char 2        |
|    ...                               |
|  i  : string literal 2 char 1        |
| i+1 : string literal 2 char 2        |
|    ...                               |
|  j  : pre-alllocated string 1 char 1 |
| j+1 : pre-alllocated string 1 char 2 |
|    ...                               |
|  k  : pre-alllocated string 2 char 1 |
| k+1 : pre-alllocated string 2 char 2 |
|    ...                               |
| n-k : stack variable k               |
|    ...                               |
| n-3 : stack variable 3               |
| n-2 : stack variable 2               |
| n-1 : stack variable 1               |
+--------------------------------------+
```

### Зарезервированные ячейки

Первые два адреса (`00` и `01`) зарезервированы для портов ввода/вывода. (Конкретные адреса настраиваются в коде симулятора)

### Числа

Все числовые константы любого размера (в рамках 32 бит) встраиваются непосредсвенно в саму инструкцию. Например,

    000101000000 LD 1
    
Здесь константа `1` является частью описания инструкции `01000000`.

Другими словами, все числовые литералы используют непосредсвенную адресацию.

### Строки

Все строковые литералы сохраняются в памяти в порядке их обнаружения в исходном коде программы. Транслятор избегает расположения в памяти дублирующихся строк.

Строки сохраняются посимвольно. Один символ - одно машинное слово.

Значением строкового литерала считается адрес первого символа строки данного литерала. Например, при передаче строки в виде параметра функции адрес строки копируется на стек, а не сама строка.

Поэтому, так как адрес строки - это число, то строковые литералы могут использовать непосредсвенную адресацию.

Программист может выделить ничем не заполненную строку фиксированного размера, например, для будущего ввода данных от пользователя. 

    (makestring 100)
    
Такие строки располагаются после обычных строк в порядке их обнаружениях в коде программы.

### Стек

Симулятор выделяет стек фиксированного размера и расширяет им память данных. 

На стеке сохраняются все переменные и параметры, а также адреса возврата функций. 

Строковые переменные хранят адрес первого символа строки.

Так как глобальные переменные трактуются транслятором, как специальный вид локальных переменных, существующих всё время работы программы, то они также существуют на стеке.


## Система команд

### Особенности процессора

Машинное слово -- 32 бит, знаковое. Исключение составляет память команд, размер машинного слова которой 48 бит.

Типы операндов инстукций:

1. `IMMEDIATE`

    Непосредственная константа

2. `ADDRESS`

    Адрес в памяти данных или в памяти инструкций в зависимости от инструкции

3. `STACK_OFFSET`

    Адрес в памяти данных, вычисляемый на базе стекового регистра `SP` с константным смещением

4. `STACK_POINTER`

    Процессор поддерживает специальный режим косвенной адресации относительно стекового регистра `STACK_POINTER`. По сути, это указатель, хранящийся. как переменная на стеке.

`PUSH` всегда кладёт значение аккумулятора на стек.

`POP` не меняет аккумулятор (это сделано для удобства программирования, по сути `POP` используется для очистки стека, а не для взятия значений с него).

Поток управления:

* инкремент PC после каждой инструкции;
* условный (jcc) и безусловный (jmp) переходы.

### Набор инструкций

| Код  | Мнемоника | Тип операнда  | Действие                              |
|------|-----------|---------------|---------------------------------------|
| 0001 | LD        | IMMEDIATE     | operand -> ACC                        |
| 0002 | LD        | ADDRESS       | DMEM[operand] -> ACC                  |
| 0003 | LD        | STACK_OFFSET  | DMEM[SP + operand] -> ACC             |
| 0004 | LD        | STACK_POINTER | DMEM[DMEM[SP + operand]] -> ACC       |
| 0102 | ST        | ADDRESS       | ACC -> DMEM[operand]                  |
| 0103 | ST        | STACK_OFFSET  | ACC -> DMEM[SP + operand]             |
| 0104 | ST        | STACK_POINTER | ACC -> DMEM[DMEM[SP + operand]]       |
| 0201 | ADD       | IMMEDIATE     | ACC + operand -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0202 | ADD       | ADDRESS       | ACC + DMEM[operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0203 | ADD       | STACK_OFFSET  | ACC + DMEM[SP + operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0204 | ADD       | STACK_POINTER | ACC + DMEM[DMEM[SP + operand]] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0301 | SUB       | IMMEDIATE     | ACC - operand -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0302 | SUB       | ADDRESS       | ACC - DMEM[operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0303 | SUB       | STACK_OFFSET  | ACC - DMEM[SP + operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0304 | SUB       | STACK_POINTER | ACC - DMEM[DMEM[SP + operand]] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0401 | MUL       | IMMEDIATE     | ACC * operand -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0402 | MUL       | ADDRESS       | ACC * DMEM[operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0403 | MUL       | STACK_OFFSET  | ACC * DMEM[SP + operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0404 | MUL       | STACK_POINTER | ACC * DMEM[DMEM[SP + operand]] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0501 | DIV       | IMMEDIATE     | ACC / operand -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0502 | DIV       | ADDRESS       | ACC / DMEM[operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0503 | DIV       | STACK_OFFSET  | ACC / DMEM[SP + operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0504 | DIV       | STACK_POINTER | ACC / DMEM[DMEM[SP + operand]] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |    
| 0601 | REM       | IMMEDIATE     | ACC % operand -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0602 | REM       | ADDRESS       | ACC % DMEM[operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0603 | REM       | STACK_OFFSET  | ACC % DMEM[SP + operand] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |
| 0604 | REM       | STACK_POINTER | ACC % DMEM[DMEM[SP + operand]] -> ACC<br>ACC < 0, ACC = 0, ACC > 0 -> FLAGS    |     
| 0702 | JMP       | ADDRESS       | operand -> PC                         |
| 0802 | JE        | ADDRESS       | if ZF = 1 then operand -> PC          |
| 0902 | JG        | ADDRESS       | if PF = 1 then operand -> PC          |
| 0A02 | JGE       | ADDRESS       | if ZF = 1 or PF = 1 then operand -> PC|
| 0B02 | JL        | ADDRESS       | if NF = 1 then operand -> PC          |
| 0C02 | JLE       | ADDRESS       | if ZF = 1 or NF = 1 then operand -> PC|
| 0D02 | JNE       | ADDRESS       | if ZF = 0 then operand -> PC          |
| 0E00 | PUSH      | NONE          | SP = SP - 1; ACC -> DMEM[SP]          |
| 0F00 | POP       | NONE          | SP = SP + 1                           |
| 1002 | CALL      | ADDRESS       | SP = SP - 1;   PC + 1 -> DMEM[SP]; operand -> PC      |
| 1100 | RET       | NONE          | DMEM[SP]| -> PC; SP = SP + 1          |
| 1200 | HLT       | NONE          | (останов машины)                      |

В описании инструкций для упрощения опущены манипуляции регистром `DA`.     

### Кодирование инструкций

Каждая инструкция кодируется с помощью одного машинного слова, размером 48 бит:

| Биты  | Описание          |
|-------|-------------------|
| 0-7   | Код операции      |
| 8-15  | Тип операнда      |
| 16-47 | Значение операнда |

Тип операнда -- это число от 0 до 4:

Типы операндов инструкций:

0. `NONE` (нет операнда)
1. `IMMEDIATE`
2. `ADDRESS`
3. `STACK_OFFSET`
4. `STACK_POINTER`

Значение операнда зависит от типа операнда. Это может быть числовая константа, адрес или смещение относительно значения регистра `SP`.


## Транслятор

................

Этапы трансляции (функция translate):

1. Разбиение стркои на символы ???
...
10. Удаление пробельных символов
...
...............


## Модель процессора

### DataPath

![DataPath](DataPath.png)

### ControlUnit

![ControlUnit](ControlUnit.png)

## Тестирование

1. Golden-tests реализованы в: 
    - [golden/cat.yml](golden/cat.yml)
    - [golden/hello.yml](golden/hello.yml)
    - [golden/hello_user_name.yml](golden/hello_user_name.yml)
    - [golden/prob1.yml](golden/prob1.yml)
    - [golden/example.yml](golden/example.yml)
2. Традиционные интеграционные тесты: [integration_test.py](./integration_test.py) (Depricated).

Запустить тесты: `poetry run pytest . -v`

Обновить конфигурацию golden tests:  `poetry run pytest . -v --update-goldens`

CI при помощи Github Action:

``` yaml
defaults:
  run:
    working-directory: .

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install poetry
          poetry install

      - name: Run tests and collect coverage
        run: |
          poetry run coverage run -m pytest .
          poetry run coverage report -m
        env:
          CI: true

  lint:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install poetry
          poetry install

      - name: Check code formatting with Ruff
        run: poetry run ruff format --check .

      - name: Run Ruff linters
        run: poetry run ruff check .
```

где:

- `poetry` -- управления зависимостями для языка программирования Python.
- `coverage` -- формирование отчёта об уровне покрытия исходного кода.
- `pytest` -- утилита для запуска тестов.
- `ruff` -- утилита для форматирования и проверки стиля кодирования.

Пример использования и журнал работы процессора на примере `cat`:


Пример проверки исходного кода:

