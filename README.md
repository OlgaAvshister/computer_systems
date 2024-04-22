# Авшистер Ольга Аркадьевна P3213

`lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob1`

Без усложнения

## Язык программирования

Описание синтаксиса языка в форме БНФ:

```bnf
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

Стратегия вычислений -- строгая, аппликативный порядок. То есть, все вычисления происходят в том порядке, в каком они даны в программе. Параметры функций вычисляются до вызова функции, слева направо.

Поддерживается рекурсия.

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

Результат операций сравнения возвращается как число `1` или `0`.

Отдельные символы строки имеет тип целого числа.  

Числа -- целые 32-битные: от -2147483648 до 2147483647.

Результат всех математических операций будет "обрезан" до младших 32-бит без завершения проргаммы.

Операция деления `/` при делении на ноль завершает программу с ошибкой.

Поддерживаются строковые литералы произвольной длины. В конце строки ставится символ-ноль `\0`, в соответствие с вариантом. 

Все функции должны обладать уникальными именами, не совпадающими с именами переменных в данной области видимости.

Область видимости переменных -- локальная в рамках функций и частей `if`.

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



todo: как строки располагаетются в памяти ???

## Организация памяти

Система построена по гарвардской архитектуре, то есть память разделена на:

1. память инструкций
2. память данных

Обе памяти работают в линейном, плоском адресном пространстве.

### Регистры

...............................

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

...............................

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
| 00  : string literal 1 char 1        |
| 01  : string literal 1 char 2        |
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

todo: input/output ports

Модель памяти должна включать:

- Какие виды памяти и регистров доступны программисту?
- Где хранятся инструкции, процедуры и прерывания?
- Где хранятся статические и динамические данные?

А также данный раздел должен включать в себя описание того, как происходит работа с 1) литералами, 2) константами, 3) переменными, 4) инструкциями, 5) процедурами, 6) прерываниями во время компиляции и исполнения. К примеру:

- В каких случаях литерал будет использован при помощи непосредственной адресации?
- В каких случаях литерал будет сохранён в статическую память?
- Как будут размещены литералы, сохранённые в статическую память, друг относительно друга?
- Как будет размещаться в память литерал, требующий для хранения несколько машинных слов?
- В каких случаях переменная будет отображена на регистр?
- Как будет разрешаться ситуация, если регистров недостаточно для отображения всех переменных?
- В каких случаях переменная будет отображена на статическую память?
- В каких случаях переменная будет отображена на стек?
- И так далее по каждому из пунктов в зависимости от варианта...


## Система команд

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
    

## Транслятор

## Модель процессора

## Тестирование

