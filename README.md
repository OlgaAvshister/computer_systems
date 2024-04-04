# Авшистер Ольга Аркадьевна P3213

lisp | acc | harv | hw | instr | binary | stream | mem | cstr | prob1

Без усложнения

## Язык программирования

форма БНФ

digit := '0' | '1' | '2' | ... | '9'
letter := 'a' | 'b' | 'c' | ... 'z'

number := '-'? digit*
name := letter*
operation := '+' | '-' | '*' | '/' | '%' | '=' | '!=' | '<' | '>' | '<=' | '>='

expression := 
    number | 
    name | 
    operation |
    '(' expression* ')'

program := expression*

## Организация памяти

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

