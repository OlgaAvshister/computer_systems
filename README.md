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

## Транслятор

## Модель процессора

## Тестирование

