from __future__ import annotations


# common.py

from enum import Enum

INPUT_PORT_ADDR = 0
OUTPUT_PORT_ADDR = 1


class Opcode(Enum):
    LD = 0
    ST = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    REM = 6
    JMP = 7
    JE = 8
    JG = 9
    JGE = 10
    JL = 11
    JLE = 12
    JNE = 13
    PUSH = 14
    POP = 15
    CALL = 16
    RET = 17
    HLT = 18


class OperandType(Enum):
    NONE = 0
    IMMEDIATE = 1
    ADDRESS = 2
    STACK_OFFSET = 3
    STACK_POINTER = 4


# translator.py

from typing import Any


class Token:
    value: str
    line: int | None

    def __init__(self, value, line=None):
        self.value = value
        self.line = line

    def __repr__(self):
        return "Token(" + self.value + ")"


# split string to chars
def split_chars(source: str) -> list[Token]:
    lst = []
    line = 1
    for i in range(len(source)):
        if source[i] != "\n":
            tok = Token(source[i], line)
            lst.append(tok)
        if source[i] == "\n":
            line += 1
    return lst


# combine string literals "..."
def combine_str_lits(lst: list[Token]) -> list[Token]:
    i = 0
    while i < len(lst):
        if lst[i].value == '"':
            while lst[i + 1].value != '"':
                lst[i].value += lst[i + 1].value
                del lst[i + 1]
            lst[i].value += '"'
            del lst[i + 1]
        i += 1
    return lst


# combine digits into numbers
def combine_numbers(lst: list[Token]) -> list[Token]:
    for j in range(len(lst)):
        for i in range(len(lst) - 1):
            if lst[i].value[-1].isdigit():
                if lst[i + 1].value.isdigit():
                    lst[i].value += lst[i + 1].value
                    del lst[i + 1]
                    break
            if lst[i].value == "-":
                if lst[i + 1].value.isdigit():
                    lst[i].value += lst[i + 1].value
                    del lst[i + 1]
                    break
    return lst


# combine letters into names
def combine_names(lst: list[Token]) -> list[Token]:
    for j in range(len(lst)):
        for i in range(len(lst) - 1):
            if lst[i].value.isalpha():
                if lst[i + 1].value.isalpha():
                    lst[i].value += lst[i + 1].value
                    del lst[i + 1]
                    break
    return lst

# combine: '!' with '=', '>' with '=', '<' with '='
def combine_operators(lst: list) -> list:
    for _ in range(len(lst)):  # repeat multiple times
        for i in range(len(lst) - 1):
            if lst[i].value in {"!", "<", ">"}:
                if lst[i + 1].value == "=":
                    lst[i].value += lst[i + 1].value
                    del lst[i + 1]
                    break
    return lst


def not_space(tok: Token) -> bool:
    if tok.value != " ":
        return True
    else:
        return False


def make_tree(toks: list[Token]) -> list[Token | list]:
    lst: list[Token | list] = []
    stack: list[list] = []
    for tok in toks:
        if tok.value == "(":
            stack.append(lst)
            lst = []
        elif tok.value == ")":
            stack[-1].append(lst)
            lst = stack.pop()
        else:
            lst.append(tok)
    return lst


def wrap_program(tree: list[Any]):
    return [Token("defun", 1), Token("_start", 1), [], [Token("do", 1)] + tree]


def rewrite_special_chars(node: list[Any]):
    new_node = []
    for el in node:
        if type(el) is list:
            new_el = rewrite_special_chars(el)
            new_node.append(new_el)
        else:
            if el.value[0] == '"':
                el.value = el.value.replace("\\n", "\n")
                new_node.append(el)
            else:
                new_node.append(el)
    return new_node


def collect_string_literals(node: list[Any]):
    lits = []
    for el in node:
        if type(el) is Token:
            if el.value[0] == '"':
                lits.append(el.value[1:-1])
        else:
            el_lits = collect_string_literals(el)
            for lit in el_lits:
                lits.append(lit)
    return lits


def make_string_map(lits: list[str]) -> dict[str, int]:
    dic = {}
    addr = 2  # first two address are reserved for input/output
    for lit in lits:
        if lit not in dic:
            dic[lit] = addr
            addr += len(lit) + 1  # length + zero character
    return dic


def generate_static_memory(str_map: dict):
    mem = [0, 0]  # reserver two cells for input/output
    for key in str_map:
        for i in range(len(key)):
            mem.append(ord(key[i]))
        mem.append(0)
    return mem


# replace strings with their addresses
def replace_string_literas(node: list[Any], str_map: dict):
    new_node = []
    for el in node:
        if type(el) is list:
            new_el = replace_string_literas(el, str_map)
            new_node.append(new_el)
        else:
            if el.value[0] == '"':
                addr = str_map[el.value[1:-1]]
                new_node.append(addr)
            else:
                new_node.append(el)
    return new_node


def is_number(s: str) -> bool:
    if s == "":
        return False
    if s[0] == "-":
        return is_number(s[1:])
    else:
        for c in s:
            if not ("0" <= c <= "9"):
                return False
        return True


def replace_numbers(node: list[Any]):
    new_node = []
    for el in node:
        if type(el) is list:
            new_el = replace_numbers(el)
            new_node.append(new_el)
        else:
            if type(el) is Token and is_number(el.value):
                num = int(el.value)
                new_node.append(num)
            else:
                new_node.append(el)
    return new_node


def is_token(obj: Any, check_for: str | None = None):
    if check_for is None:
        return type(obj) is Token
    else:
        return type(obj) is Token and obj.value == check_for

    # (makestring n)


def process_makestring_forms(node: Any, mem: list[int]):
    if type(node) is int:
        return node
    elif type(node) is Token:
        return node
    elif type(node) is list and node and is_token(node[0], "makestring"):
        addr = len(mem)
        for i in range(node[1] + 1):
            mem.append(0)
        return addr
    else:
        assert type(node) is list
        res = []
        for el in node:
            el = process_makestring_forms(el, mem)
            res.append(el)
        return res
    return node


# rewrite `getchar` calls to simplify further processing
def rewrite_getchar(node: Any):
    if type(node) is list and node and is_token(node[0], "getchar"):
        head = node[0]
        res = []
        res.append(head)
        res.append([Token("+", head.line), rewrite_getchar(node[1]), rewrite_getchar(node[2])])
        return res
    elif type(node) is list:
        res = []
        for el in node:
            new_el = rewrite_getchar(el)
            res.append(new_el)
        return res
    else:
        return node


# rewrite `setchar` calls to simplify further processing
def rewrite_setchar(node: Any):
    if type(node) is list and node and is_token(node[0], "setchar"):
        head = node[0]
        res = []
        res.append(head)
        res.append([Token("+", head.line), rewrite_setchar(node[1]), rewrite_setchar(node[2])])
        res.append(node[3])
        return res
    elif type(node) is list:
        res = []
        for el in node:
            new_el = rewrite_setchar(el)
            res.append(new_el)
        return res
    else:
        return node


class VarRef:
    name: str
    line: int

    def __init__(self, name: str, line: int):
        self.name = name
        self.line = line

    def __repr__(self):
        return "VarRef(" + str(self.name) + ")"


def is_operation_string(obj: Any):
    return type(obj) is str and obj in {"+", "-", "*", "/", "%", "=", ">", "<", ">=", "<=", "!="}


def is_operation_token(obj: Any):
    return type(obj) is Token and is_operation_string(obj.value)


def is_operation_expr(obj: Any):
    return type(obj) is list and obj and is_operation_token(obj[0])


def is_keyword_string(obj: Any):
    return type(obj) is str and obj in {"do", "if", "define", "defun"}


def is_keyword_token(obj: Any):
    return type(obj) is Token and is_keyword_string(obj.value)


def is_keyword_expr(obj: Any):
    return type(obj) is list and obj and is_keyword_token(obj[0])


def is_builtin_string(obj: Any):
    return type(obj) is str and obj in {"getchar", "setchar", "readchar", "printchar"}


def is_builtin_token(obj: Any):
    return type(obj) is Token and is_builtin_string(obj.value)


def is_builtin_call_expr(obj: Any):
    return type(obj) is list and obj and is_builtin_token(obj[0])


def is_call_expr(obj: Any):
    return type(obj) is list and obj and not is_keyword_token(obj[0])


# move part of a complex expression as
# separate expression giving it name
def extract_variable(el: list, temp_count: int, line: int | None):
    el, temp_count = extract_variables(el, temp_count)
    temp_name = "_temp" + str(temp_count)
    define = [Token("define", el[0].line), Token(temp_name), el]
    temp_count += 1
    return VarRef(temp_name, line), define, temp_count

def extract_call_variables(node: list, temp_count: int):
    res = []
    defines = []
    for el in node:
        if type(el) is list:
            el, define, temp_count = extract_variable(el, temp_count, node[0].line)
            res.append(el)
            defines.append(define)
        else:
            res.append(el)
    if len(defines) > 0:
        # `do` is needed to wrap several expressions
        wrapper: list = [Token("do")]
        for define in defines:
            wrapper.append(define)
        wrapper.append(res)
        return wrapper, temp_count
    else:
        return res, temp_count

# break complex expression into simpler ones
def extract_variables(node: Any, temp_count: int = 0):
    if type(node) is int:
        return node, temp_count
    elif type(node) is Token:
        return node, temp_count
    else:
        assert type(node) is list
        if is_operation_expr(node) or is_builtin_call_expr(node) or is_call_expr(node):
            res, temp_count = extract_call_variables(node, temp_count)
            return res, temp_count
        else:  # special form (for example, `if`)
            res = []
            for el in node:
                el, temp_count = extract_variables(el, temp_count)
                res.append(el)
            return res, temp_count


def collect_function_names(node: list[Any]) -> list[str]:
    names = []
    tick = 0
    for el in node:
        if type(el) is list:
            el_names = collect_function_names(el)
            for name in el_names:
                names.append(name)
        elif type(el) is Token and el.value == "defun":
            tick = 1
        elif type(el) is Token and tick == 1:
            names.append(el.value)
            tick = 0
    return names


class FuncRef:
    name: str
    index: int
    line: int | None
    tail_call: bool
    tail_call_stack_size: int

    def __init__(self, name: str, index: int, line=None):
        self.name = name
        self.index = index
        self.line = line
        self.tail_call = False
        self.tail_call_stack_size = 0

    def __repr__(self):
        return "FuncRef(" + self.name + ")"


def rewrite_function_calls(node: list[Any], func_names: list[str]):
    tick = 0
    new_node = []
    for el in node:
        if type(el) is list:
            new_node.append(rewrite_function_calls(el, func_names))
        elif type(el) is Token and el.value == "defun":
            tick = 1
            new_node.append(el)
        elif type(el) is Token and tick == 1:
            tick = 0
            new_node.append(el)
        elif type(el) is Token and tick == 0 and el.value in func_names:
            index = func_names.index(el.value)
            el = FuncRef(el.value, index, el.line)
            new_node.append(el)
        else:  # not a list
            new_node.append(el)
    return new_node


def count_own_vars(node: Any):
    if is_token(node, "define"):
        return 1
    elif type(node) is list and node and is_token(node[0], "if"):
        # `if` cleans its own stack, so ignore it
        return 0
    elif type(node) is list:  # not an `if`
        count = 0
        for el in node:
            count += count_own_vars(el)
        return count
    else:
        return 0


class Function:
    name: str
    args: list[str]
    body: Any
    var_count: int
    addr: int

    def __init__(self, name: str, args: list[str], body: Any, var_count: int):
        self.name = name
        self.args = args
        self.body = body
        self.var_count = var_count
        self.addr = 0

    def __repr__(self):
        return "Function(" + self.name + ", " + str(self.args) + ")"


def do_collect_functions(node: Any) -> tuple[Any, list[Function]]:
    if type(node) is list and node and is_token(node[0], "defun"):
        assert is_token(node[1]), "Wrong name of function"
        assert len(node) == 4, "Wrong function declaration"
        assert type(node[2]) is list, "Function paramers should be a list"
        args = []
        for tok in node[2]:
            assert type(tok) is Token, "Wrong name of function parameter"
            args.append(tok.value)
        body, funcs = do_collect_functions(node[3])
        var_count = count_own_vars(body)
        func = Function(node[1].value, args, body, var_count)
        funcs.insert(0, func)
        return None, funcs
    elif type(node) is list:
        funcs = []
        res = []
        for el in node:
            el, el_funcs = do_collect_functions(el)
            if el is not None:
                res.append(el)
            funcs.extend(el_funcs)
        return res, funcs
    else:
        return node, []


def collect_functions(tree: list) -> list[Function]:
    # get functions only
    return do_collect_functions(tree)[1]


class ArgRef:
    name: str
    line: int

    def __init__(self, name: str, line: int):
        self.name = name
        self.line = line

    def __repr__(self):
        return "ArgRef(" + str(self.name) + ")"


def rewrite_arguments(node: Any, args: list[str]):
    if type(node) is int:
        return node
    elif type(node) is Token:
        if node.value in args:
            assert type(node.line) is int
            return ArgRef(node.value, node.line)
        else:
            return node
    elif type(node) is list:
        res = []
        for el in node:
            res.append(rewrite_arguments(el, args))
        return res
    else:
        return node


def rewrite_vars(node: Any, names: list[str]):
    if type(node) is list and is_token(node[0], "define"):
        assert is_token(node[1]), "Wrong name of variable"
        names.append(node[1].value)
        res = []
        res.append(node[0])  # define
        res.append(node[1])  # varaible name
        for el in node[2:]:
            res.append(rewrite_vars(el, names))
        return res
    elif is_token(node) and node.value in names:
        return VarRef(node.value, node.line)
    elif type(node) is list:
        res = []
        for el in node:
            res.append(rewrite_vars(el, names))
        return res
    else:
        return node


class If:
    # counters of created variables in different parts of `if`
    cond_vars: int
    true_branch_vars: int
    false_branch_vars: int
    line: int

    def __init__(self, cond_vars: int, true_branch_vars: int, false_branch_vars: int, line: int):
        self.cond_vars = cond_vars
        self.true_branch_vars = true_branch_vars
        self.false_branch_vars = false_branch_vars
        self.line = line

    def __repr__(self):
        return "If(%d %d %d)" % (self.cond_vars, self.true_branch_vars, self.false_branch_vars)


def rewrite_ifs(node: Any):
    if type(node) is list and is_token(node[0], "if"):
        assert len(node) == 4, "Wrong IF structure"
        cond_vars = count_own_vars(node[1])
        true_branch_vars = count_own_vars(node[2])
        false_branch_vars = count_own_vars(node[3])
        head = If(cond_vars, true_branch_vars, false_branch_vars, node[0].line)
        res = [head]
        for el in node[1:]:
            res.append(rewrite_ifs(el))
        return res
    elif type(node) is list:
        res = []
        for el in node:
            res.append(rewrite_ifs(el))
        return res
    else:
        return node


class StackOffset:
    offset: int
    line: int

    def __init__(self, offset: int, line: int):
        self.offset = offset
        self.line = line

    def __repr__(self):
        return "StackOffset(" + str(self.offset) + ")"


def rewrite_var_refs(node: Any, stack: list[str]):
    if type(node) is ArgRef or type(node) is VarRef:
        reversed_stack = stack[::-1]
        return StackOffset(reversed_stack.index(node.name),  node.line)
    elif type(node) is list and is_token(node[0], "define"):
        res = []
        res.append(node[0])  # define
        res.append(node[1])  # variable name
        for el in node[2:]:
            res.append(rewrite_var_refs(el, stack))
        stack.append(node[1].value)
        return res
    elif type(node) is list and type(node[0]) is If:
        res = []
        for el in node:
            old_len = len(stack)
            res.append(rewrite_var_refs(el, stack))
            stack[old_len:] = []
        return res
    elif type(node) is list:
        res = []
        for el in node:
            res.append(rewrite_var_refs(el, stack))
        return res
    else:
        return node


def is_math_op_string(s: str):
    return s in {"+", "-", "*", "/", "%"}


def is_comparison_op_string(s: str):
    return s in {"<", ">", "=", "!=", "<=", ">="}


def is_op_string(s: str):
    return is_math_op_string(s) or is_comparison_op_string(s)


def math_opcode(s: str) -> Opcode:
    if s == "+":
        return Opcode.ADD
    elif s == "-":
        return Opcode.SUB
    elif s == "*":
        return Opcode.MUL
    elif s == "/":
        return Opcode.DIV
    else:
        return Opcode.REM


def comparison_opcode(s: str) -> Opcode:
    if s == ">":
        return Opcode.JG
    elif s == "<":
        return Opcode.JL
    elif s == "=":
        return Opcode.JE
    elif s == "!=":
        return Opcode.JNE
    elif s == ">=":
        return Opcode.JGE
    else:
        return Opcode.JLE


class Instruction:
    opcode: Opcode
    operand_type: OperandType
    operand: int
    line: int | None

    def __init__(self, opcode: Opcode, operand_type: OperandType, operand: int, line=None):
        self.opcode = opcode
        self.operand_type = operand_type
        self.operand = operand
        self.line = line

    def __repr__(self):
        return "Instruction(%s, %s, %d)" % (self.opcode, self.operand_type, self.operand)


def offset_stack_refs(code: list[Instruction], delta: int):
    stack_operand_types = {OperandType.STACK_OFFSET, OperandType.STACK_POINTER}
    for instr in code:
        if instr.operand_type in stack_operand_types:
            instr.operand += delta


def mark_tail_calls(node: Any, func_name: str):
    if type(node) in {int, StackOffset, If}:
        return
    elif type(node) is list:
        if node == []:
            return
        head = node[0]
        if type(head) is FuncRef and head.name == func_name:
            head.tail_call = True
        elif type(head) is If:  # `if` head itself
            mark_tail_calls(node[2], func_name)
            mark_tail_calls(node[3], func_name)
        elif is_token(head, "do") and len(node) > 1:
            mark_tail_calls(node[-1], func_name)
    else:
        raise ValueError("Incorrect program")


def calc_tail_calls_stack_size(node: Any, stack: list[str]):
    if type(node) is list and is_token(node[0], "define"):
        for el in node[2:]:  # skip `define` and variable name
            calc_tail_calls_stack_size(el, stack)
        stack.append(node[1].value)
    elif type(node) is list and type(node[0]) is If:
        for el in node:
            old_len = len(stack)
            calc_tail_calls_stack_size(el, stack)            
            stack[old_len:] = []
    elif type(node) is list and type(node[0]) is FuncRef and node[0].tail_call:
        node[0].tail_call_stack_size = len(stack)
    elif type(node) is list:
        for el in node:
            calc_tail_calls_stack_size(el, stack)


def generate_load(node: Any, line: int | None=None):
    if type(node) is int:
        return [Instruction(Opcode.LD, OperandType.IMMEDIATE, node, line)]
    else:
        return [Instruction(Opcode.LD, OperandType.STACK_OFFSET, node.offset, node.line)]


def generate_define(node: Any, line: int | None=None):
    assert is_token(node[1]), "Wrong name of variable"
    res = generate_instructions(node[2], node[0].line)
    res.append(Instruction(Opcode.PUSH, OperandType.NONE, 0, node[0].line))
    return res


def generate_operation(node: Any, line: int | None=None):
    assert type(node) is list
    assert is_token(node[0])
    assert is_op_string(node[0].value)
    assert len(node) == 3, "Wrong number of arguments"
    if is_math_op_string(node[0].value):
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        opcode = math_opcode(node[0].value)
        if type(node[2]) is int:
            res.append(Instruction(opcode, OperandType.IMMEDIATE, node[2], node[0].line))
        elif type(node[2]) is StackOffset:
            res.append(Instruction(opcode, OperandType.STACK_OFFSET, node[2].offset, node[0].line))
        else:
            raise ValueError("Internal parser error")
        return res
    else:
        # compare and load TRUE (1) or FALSE (0) into accumulator
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        if type(node[2]) is int:
            res.append(Instruction(Opcode.SUB, OperandType.IMMEDIATE, node[2], node[0].line))
        elif type(node[2]) is StackOffset:
            res.append(Instruction(Opcode.SUB, OperandType.STACK_OFFSET, node[2].offset, node[0].line))
        else:
            raise ValueError("Internal parser error")
        opcode = comparison_opcode(node[0].value)
        res.extend(
            [
                Instruction(opcode, OperandType.ADDRESS, 2, node[0].line),
                Instruction(Opcode.LD, OperandType.IMMEDIATE, 0, node[0].line),
                Instruction(Opcode.JMP, OperandType.ADDRESS, 1, node[0].line),
                Instruction(Opcode.LD, OperandType.IMMEDIATE, 1, node[0].line),
            ]
        )
        return res


def generate_do(node: Any, line: int | None=None):
    res = []
    for el in node[1:]:
        res.extend(generate_instructions(el, node[0].line))
    return res


def generate_if(node: Any, line: int | None=None):
    res = []
    # condition
    res.extend(generate_instructions(node[1], node[0].line))
    generate_pops(res, node[0].cond_vars, node[0].line)
    res.append(Instruction(Opcode.SUB, OperandType.IMMEDIATE, 0, node[0].line))  # for flags
    res.append(Instruction(Opcode.JE, OperandType.ADDRESS, 0, node[0].line))
    je_index = len(res) - 1
    # true (main) branch
    res.extend(generate_instructions(node[2], node[0].line))
    generate_pops(res, node[0].true_branch_vars, node[0].line)
    res.append(Instruction(Opcode.JMP, OperandType.ADDRESS, 0, node[0].line))
    res[je_index].operand = len(res) - je_index - 1
    jmp_index = len(res) - 1
    # false (else) branch
    res.extend(generate_instructions(node[3], node[0].line))
    generate_pops(res, node[0].false_branch_vars, node[0].line)
    res[jmp_index].operand = len(res) - jmp_index - 1
    return res


def generate_builtin_call(node: Any, line: int | None=None):
    if type(node) is list and is_token(node[0], "getchar"):
        res = []
        res.append(Instruction(Opcode.LD, OperandType.STACK_POINTER, node[1].offset, node[0].line))
        return res
    elif type(node) is list and is_token(node[0], "setchar"):
        res = []
        res.extend(generate_instructions(node[2], node[0].line))
        res.append(Instruction(Opcode.ST, OperandType.STACK_POINTER, node[1].offset, node[0].line))
        return res
    elif type(node) is list and is_token(node[0], "readchar"):
        res = []
        res.append(Instruction(Opcode.LD, OperandType.ADDRESS, INPUT_PORT_ADDR, node[0].line))
        return res
    elif type(node) is list and is_token(node[0], "printchar"):
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        res.append(Instruction(Opcode.ST, OperandType.ADDRESS, OUTPUT_PORT_ADDR, node[0].line))
        return res
    else:
        raise ValueError("Unknown builtin function")


def generate_call(node: Any, line: int | None=None):        
    func = node[0]
    args_count = len(node[1:])
    if func.tail_call:
        res = []
        for i in range(args_count):
            arg_code = generate_instructions(node[i + 1], node[0].line)
            res.extend(arg_code)
            arg_addr = (args_count - 1) - i + func.tail_call_stack_size + 1
            res.append(Instruction(Opcode.ST, OperandType.STACK_OFFSET, arg_addr, node[0].line))
        # clean up all the variables created before the tail call
        stack_to_clean = func.tail_call_stack_size
        generate_pops(res, stack_to_clean, node[0].line)
        res.append(Instruction(Opcode.JMP, OperandType.ADDRESS, 0, node[0].line))
        return res
    else:
        res = []
        for i in range(args_count):
            arg_code = generate_instructions(node[i + 1], node[0].line)
            offset_stack_refs(arg_code, i)
            res.extend(arg_code)
            res.append(Instruction(Opcode.PUSH, OperandType.NONE, 0, node[0].line))
        res.append(Instruction(Opcode.CALL, OperandType.ADDRESS, node[0].index, node[0].line))
        generate_pops(res, args_count, node[0].line)
        return res


def generate_function(node: Any, line: int | None=None):
    res = []
    res.extend(generate_instructions(node.body))
    generate_pops(res, node.var_count)
    if node.name == "_start":  # main program
        res.append(Instruction(Opcode.HLT, OperandType.NONE, 0))
    else:
        res.append(Instruction(Opcode.RET, OperandType.NONE, 0))
    return res        


def generate_pops(code: list[Instruction], count: int, line: int | None = None):
    code.extend([Instruction(Opcode.POP, OperandType.NONE, 0, line)] * count)


def generate_instructions(node: Any, line: int | None=None) -> list[Instruction]:
    if type(node) in {int, StackOffset}:
        return generate_load(node, line)
    elif type(node) is list and is_token(node[0], "define"):
        return generate_define(node, line)
    elif type(node) is list and is_token(node[0]) and is_op_string(node[0].value):
        return generate_operation(node, line)
    elif type(node) is list and is_token(node[0], "do"):
        return generate_do(node, line)
    elif type(node) is list and type(node[0]) is If:
        return generate_if(node, line)
    elif is_builtin_call_expr(node):
        return generate_builtin_call(node, line)
    elif type(node) is list and type(node[0]) is FuncRef:
        return generate_call(node, line)
    elif type(node) is Function:
        return generate_function(node, line)
    else:
        raise ValueError("Incorrect program")


def calc_func_addrs(funcs: list[Function], funcs_code: list):
    addr = 0
    for i in range(len(funcs)):
        funcs[i].addr = addr
        addr += len(funcs_code[i])


def is_jump_instruction_opcode(opcode: Opcode):
    jumps = {Opcode.JMP, Opcode.JE, Opcode.JG, Opcode.JL, Opcode.JLE, Opcode.JGE, Opcode.JNE}
    return opcode in jumps


def fix_jumps(code: list[Instruction], func_addr: int):
    instr_addr = func_addr
    for i in range(len(code)):
        if is_jump_instruction_opcode(code[i].opcode):
            if code[i].operand == 0:  # tail call jump
                code[i].operand = func_addr
            else:
                code[i].operand += instr_addr + 1
        instr_addr += 1


# make calls use real global absolute addresses
def fix_calls(code: list[Instruction], funcs: list[Function]):
    for i in range(len(code)):
        if code[i].opcode == Opcode.CALL:
            func_index = code[i].operand
            code[i].operand = funcs[func_index].addr


def merge_functions(funcs_code: list):
    res = []
    for code in funcs_code:
        res.extend(code)
    return res


def encode_instruction(instr: Instruction) -> list[int]:
    binary_code = []
    binary_code.append(instr.opcode.value)
    binary_code.append(instr.operand_type.value)
    binary_code.extend(list(instr.operand.to_bytes(4, "little", signed=True)))
    return binary_code


def encode_instructions(program_code: list[Instruction]):
    binary_code = []
    for instr in program_code:
        binary_code.extend(encode_instruction(instr))
    return binary_code


def read_source_code(file_name: str) -> str:
    file = open(file_name)
    content = file.read()
    file.close()
    return content


def read_stdlib(file_name: str) -> str:
    file = open(file_name)
    content = file.read()
    file.close()
    return content


def instruction_to_string(instr: Instruction) -> str:
    if instr.operand_type == OperandType.IMMEDIATE:
        return "%s %d" % (instr.opcode.name, instr.operand)
    elif instr.operand_type == OperandType.ADDRESS:
        if is_jump_instruction_opcode(instr.opcode) or instr.opcode == Opcode.CALL:
            return "%s %d" % (instr.opcode.name, instr.operand)
        else:
            return "%s [%d]" % (instr.opcode.name, instr.operand)
    elif instr.operand_type == OperandType.STACK_OFFSET:
        return "%s [SP + %d]" % (instr.opcode.name, instr.operand)
    elif instr.operand_type == OperandType.STACK_POINTER:
        return "%s [[SP + %d]]" % (instr.opcode.name, instr.operand)
    else:
        return "%s" % instr.opcode.name


def get_source_line(program_source: str, line: int | None) -> str:
    if line is not None:
        lines = program_source.split("\n")
        return lines[line - 1]
    else:
        return ""


def lines_of_code(program_source: str) -> int:
    lines = program_source.split("\n")
    count = 0
    for s in lines:
        for sym in s:
            if sym != " ":
                count += 1
                break
    return count


def make_debug_info(program_source: str, program_code: list[Instruction]):
    res = []
    addr = 0
    for instr in program_code:
        code = encode_instruction(instr)
        hex_code = ""
        for num in code:
            s = hex(num)[2:]
            if len(s) == 1:
                hex_code += "0" + s
            else:
                hex_code += s
        res.append(
            "%04d: %s %s  ; %s"
            % (addr, hex_code, instruction_to_string(instr).ljust(17, " "), get_source_line(program_source, instr.line))
        )
        addr += 1
    return "\n".join(res)


def write_code(code: list[int], file_name: str):
    file = open(file_name, "wb")
    file.write(bytearray(code))
    file.close()


def write_data(mem: list[int], file_name: str):
    file = open(file_name, "wb")
    file.write(bytearray(mem))
    file.close()


def write_debug_info(debug_info: str, file_name: str):
    file = open(file_name, "w")
    file.write(debug_info)
    file.close()


def translate(source_name: str, binary_name: str, memory_name: str, debug_name: str):
    try:
        program_source = read_source_code(source_name)
        loc = lines_of_code(program_source)
        stdlib = read_stdlib("stdlib.lsp")
        program_source += "\n" + stdlib
        lst = split_chars(program_source)
        lst = combine_str_lits(lst)
        lst = combine_numbers(lst)
        lst = combine_names(lst)
        lst = combine_operators(lst)
        lst = list(filter(not_space, lst))
        tree = make_tree(lst)
        tree = wrap_program(tree)
        tree = rewrite_special_chars(tree)
        str_lits = collect_string_literals(tree)
        str_map = make_string_map(str_lits)
        mem = generate_static_memory(str_map)
        tree = replace_string_literas(tree, str_map)
        tree = replace_numbers(tree)
        tree = process_makestring_forms(tree, mem)
        tree = rewrite_getchar(tree)
        tree = rewrite_setchar(tree)
        tree, _ = extract_variables(tree)
        func_names = collect_function_names(tree)
        tree = rewrite_function_calls(tree, func_names)
        funcs = collect_functions(tree)
        for i in range(len(funcs)):
            body = rewrite_arguments(funcs[i].body, funcs[i].args)
            body = rewrite_vars(body, [])
            body = rewrite_ifs(body)
            stack = funcs[i].args[:]
            # reserve one stack element for return address
            stack.append("RETURN_ADDR")
            body = rewrite_var_refs(body, stack)
            mark_tail_calls(body, funcs[i].name)
            calc_tail_calls_stack_size(body, [])
            funcs[i].body = body
        funcs_code = []
        for i in range(len(funcs)):
            code = generate_instructions(funcs[i])
            funcs_code.append(code)
        calc_func_addrs(funcs, funcs_code)
        for i in range(len(funcs_code)):
            fix_jumps(funcs_code[i], funcs[i].addr)
            fix_calls(funcs_code[i], funcs)
        program_code = merge_functions(funcs_code)
        binary_code = encode_instructions(program_code)
        print("LoC:", loc)
        print("Instructions:", len(program_code))
        write_code(binary_code, binary_name)
        write_data(mem, memory_name)
        if debug_name is not None:
            debug_info = make_debug_info(program_source, program_code)
            write_debug_info(debug_info, debug_name)
    except Exception as e:
        print(str(e))


# if __name__ == "__main__":
if False:
    import argparse

    arg_parser = argparse.ArgumentParser(prog="translator", description="Lisp-like language translator")
    arg_parser.add_argument("source_name", help="input file with program source code")
    arg_parser.add_argument("binary_name", help="output file for binary code")
    arg_parser.add_argument("memory_name", help="output file for static memory content")
    arg_parser.add_argument("debug_name", nargs="?", help="output file for debug information")
    args = arg_parser.parse_args()
    translate(args.source_name, args.binary_name, args.memory_name, args.debug_name)

### simulator ###

# from __future__ import annotations
import logging

DATA_WORD_SIZE = 4
CODE_WORD_SIZE = 6
STACK_SIZE = 256

DA_SEL__OPERAND = 0
DA_SEL__SP = 1
DA_SEL__SP_OFFSET = 2
DA_SEL__DATA_OUT = 3

SP_SEL__SP_INC = 0
SP_SEL__SP_DEC = 1

ALU_SEL__OPERAND = 0
ALU_SEL__DATA_OUT = 1

ALU_OP__ADD = 0
ALU_OP__SUB = 1
ALU_OP__MUL = 2
ALU_OP__DIV = 3
ALU_OP__REM = 4

ACC_SEL__OPERAND = 0
ACC_SEL__ALU_RES = 1
ACC_SEL__DATA_OUT = 2

DATA_SEL__ACC = 0
DATA_SEL__NEXT_PC = 1


def wraparound(num: int) -> int:
    # 8 bytes should be enough for all 32-bit operations
    data = num.to_bytes(8, "little", signed=True)
    return int.from_bytes(data[:4], "little", signed=True)


def mux2(a: int | None, b: int | None, sel: int):
    if sel == 0:
        return a
    elif sel == 1:
        return b
    else:
        raise ValueError("Simulation error")


def mux3(a: int | None, b: int | None, c: int | None, sel: int):
    if sel == 0:
        return a
    elif sel == 1:
        return b
    elif sel == 2:
        return c
    else:
        raise ValueError("Simulation error")


def mux4(a: int | None, b: int | None, c: int | None, d: int | None, sel: int):
    if sel == 0:
        return a
    elif sel == 1:
        return b
    elif sel == 2:
        return c
    elif sel == 3:
        return d
    else:
        raise ValueError("Simulation error")


def opcode_to_alu_op(opcode: Opcode) -> int:
    if opcode == Opcode.ADD:
        return ALU_OP__ADD
    elif opcode == Opcode.SUB:
        return ALU_OP__SUB
    elif opcode == Opcode.MUL:
        return ALU_OP__MUL
    elif opcode == Opcode.DIV:
        return ALU_OP__DIV
    elif opcode == Opcode.REM:
        return ALU_OP__REM
    else:
        raise ValueError("Wrong mathematical opcode")


class ALUResult:
    value: int
    nf: bool
    zf: bool
    pf: bool

    def __init__(self, value: int, nf: bool, zf: bool, pf: bool):
        self.value = value
        self.nf = nf
        self.zf = zf
        self.pf = pf

def alu_op(a: int, b: int, op: int):
    if op == ALU_OP__ADD:
        return a + b
    elif op == ALU_OP__SUB:
        return a - b
    elif op == ALU_OP__MUL:
        return a * b
    elif op == ALU_OP__DIV:
        if b != 0:
            return a // b
        else:
            raise ValueError("Divide by zero")
    elif op == ALU_OP__REM:
        if b != 0:
            return a % b
        else:
            raise ValueError("Divide by zero")
    else:
        raise ValueError("Wrong ALU operation")

def alu(a: int, b: int, op: int) -> ALUResult:
    value = alu_op(a, b, op)
    nf = False
    zf = False
    pf = False
    if value < 0:
        nf = True
    elif value == 0:
        zf = True
    else:
        pf = True
    value = wraparound(value)
    return ALUResult(value, nf, zf, pf)


class Signals:
    operand: int | None
    sp_sel: int | None
    latch_sp: bool
    da_sel: int | None
    latch_da: bool
    alu_sel: int | None
    alu_op: int | None
    acc_sel: int | None
    latch_acc: bool
    latch_flags: bool
    data_sel: int | None
    next_pc: int | None
    oe: bool
    wr: bool

    def __init__(
        self,
        operand: int | None = None,
        sp_sel: int | None = None,
        latch_sp: bool = False,
        da_sel: int | None = None,
        latch_da: bool = False,
        alu_sel: int | None = None,
        alu_op: int | None = None,
        acc_sel: int | None = None,
        latch_acc: bool = False,
        latch_flags: bool = False,
        data_sel: int | None = None,
        next_pc: int | None = None,
        oe: bool = False,
        wr: bool = False,
    ):
        self.operand = operand
        self.sp_sel = sp_sel
        self.latch_sp = latch_sp
        self.da_sel = da_sel
        self.latch_da = latch_da
        self.alu_sel = alu_sel
        self.alu_op = alu_op
        self.acc_sel = acc_sel
        self.latch_acc = latch_acc
        self.latch_flags = latch_flags
        self.data_sel = data_sel
        self.next_pc = next_pc
        self.oe = oe
        self.wr = wr


def decode_instruction(binary_code: int) -> Instruction:
    temp = binary_code.to_bytes(6, "little", signed=True)
    opcode = Opcode(temp[0])
    operand_type = OperandType(temp[1])
    operand = int.from_bytes(temp[2:], "little", signed=True)
    return Instruction(opcode, operand_type, operand)


class DataPath:
    acc: int  # accumulator register
    nf: bool  # negative flag (part of flags register)
    zf: bool  # zero flag (part of flags register)
    pf: bool  # positive flag (part of flags register)
    da: int  # data address register
    mem: list[int]
    sp: int  # stack pointer register
    input_stream: list[str]
    output_stream: list[str]

    def __init__(self, mem: list[int], input_stream: list[str]):
        self.acc = 0
        self.nf = False
        self.zf = False
        self.pf = False
        self.da = 0
        self.mem = mem
        self.sp = len(mem)
        self.input_stream = input_stream
        self.output_stream = []


    def read_data(self):
        addr = self.da
        if addr < 0 or addr >= len(self.mem):
            raise ValueError("Memory read out of bounds")
        if addr == INPUT_PORT_ADDR:
            if len(self.input_stream) == 0:
                raise EOFError
            return ord(self.input_stream.pop(0))
        elif addr == OUTPUT_PORT_ADDR:
            raise ValueError("Can not read from output port")
        else:
            return self.mem[addr]


    def write_data(self, value: int):
        addr = self.da
        if addr < 0 or addr >= len(self.mem):
            raise ValueError("Memory write out of bounds")
        if addr == INPUT_PORT_ADDR:
            raise ValueError("Can not write to input port")
        elif addr == OUTPUT_PORT_ADDR:
            self.output_stream.append(chr(value))
        else:
            self.mem[addr] = value


    def latch_acc_flags(self, signals: Signals, operand: int, data_out: int | None):
        if signals.acc_sel == ACC_SEL__ALU_RES or signals.latch_flags:
            assert type(signals.alu_sel) is int
            assert type(signals.alu_op) is int
            alu_right = mux2(operand, data_out, signals.alu_sel)
            alu_res = alu(self.acc, alu_right, signals.alu_op)
        else:
            alu_res = ALUResult(0, False, False, False)
        if signals.latch_acc:
            assert type(signals.acc_sel) is int
            self.acc = mux3(operand, alu_res.value, data_out, signals.acc_sel)
        if signals.latch_flags:
            self.nf = alu_res.nf
            self.zf = alu_res.zf
            self.pf = alu_res.pf        


    def handle_signals(self, signals: Signals):
        data_out: int | None = None
        if signals.operand is not None:
            operand = signals.operand
        else:
            operand = 0
        if signals.oe:
            data_out = self.read_data()
        else:
            data_out = None
        if signals.latch_da:
            assert type(signals.da_sel) is int
            self.da = mux4(operand, self.sp, self.sp + operand, data_out, signals.da_sel)
        if signals.latch_sp:
            assert type(signals.sp_sel) is int
            self.sp = mux2(self.sp + 1, self.sp - 1, signals.sp_sel)
        if signals.latch_acc or signals.latch_flags:
            self.latch_acc_flags(signals, operand, data_out)
        if signals.wr:
            assert type(signals.data_sel) is int
            data_in = mux2(self.acc, signals.next_pc, signals.data_sel)
            self.write_data(data_in)


class ControlUnit:
    dp: DataPath
    pc: int  # program counter
    mem: list[int]


    def __init__(self, dp: DataPath, mem: list[int]):
        self.dp = dp
        self.pc = 0
        self.mem = mem


    def execute_ld(self, instr: Instruction):
        match instr.operand_type:
            case OperandType.IMMEDIATE:
                # operand -> ACC
                self.dp.handle_signals(Signals(operand=instr.operand, acc_sel=ACC_SEL__OPERAND, latch_acc=True))
            case OperandType.ADDRESS:
                # DMEM[operand] -> ACC
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__OPERAND, latch_da=True))
                self.dp.handle_signals(Signals(oe=True, acc_sel=ACC_SEL__DATA_OUT, latch_acc=True))
            case OperandType.STACK_OFFSET:
                # DMEM[SP + operand] -> ACC
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__SP_OFFSET, latch_da=True))
                self.dp.handle_signals(Signals(oe=True, acc_sel=ACC_SEL__DATA_OUT, latch_acc=True))
            case OperandType.STACK_POINTER:
                # DMEM[DMEM[SP + operand]] -> ACC
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__SP_OFFSET, latch_da=True))
                self.dp.handle_signals(Signals(oe=True, da_sel=DA_SEL__DATA_OUT, latch_da=True))
                self.dp.handle_signals(Signals(oe=True, acc_sel=ACC_SEL__DATA_OUT, latch_acc=True))


    def execute_st(self, instr: Instruction):
        match instr.operand_type:
            case OperandType.ADDRESS:
                # ACC -> DMEM[operand]
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__OPERAND, latch_da=True))
                self.dp.handle_signals(Signals(data_sel=DATA_SEL__ACC, wr=True))
            case OperandType.STACK_OFFSET:
                # ACC -> DMEM[SP + operand]
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__SP_OFFSET, latch_da=True))
                self.dp.handle_signals(Signals(data_sel=DATA_SEL__ACC, wr=True))
            case OperandType.STACK_POINTER:
                # ACC -> DMEM[DMEM[SP + operand]]
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__SP_OFFSET, latch_da=True))
                self.dp.handle_signals(Signals(oe=True, da_sel=DA_SEL__DATA_OUT, latch_da=True))
                self.dp.handle_signals(Signals(data_sel=DATA_SEL__ACC, wr=True))


    def execute_math(self, instr: Instruction):
        alu_op = opcode_to_alu_op(instr.opcode)
        match instr.operand_type:
            case OperandType.IMMEDIATE:
                # ACC + operand -> ACC
                # ACC < 0, ACC = 0, ACC > 0 -> FLAGS
                self.dp.handle_signals(
                    Signals(
                        operand=instr.operand,
                        alu_sel=ALU_SEL__OPERAND,
                        alu_op=alu_op,
                        acc_sel=ACC_SEL__ALU_RES,
                        latch_acc=True,
                        latch_flags=True,
                    )
                )
            case OperandType.ADDRESS:
                # ACC + DMEM[operand] -> ACC
                # ACC < 0, ACC = 0, ACC > 0 -> FLAGS
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__OPERAND, latch_da=True))
                self.dp.handle_signals(
                    Signals(
                        oe=True,
                        alu_sel=ALU_SEL__DATA_OUT,
                        alu_op=alu_op,
                        acc_sel=ACC_SEL__ALU_RES,
                        latch_acc=True,
                        latch_flags=True,
                    )
                )
            case OperandType.STACK_OFFSET:
                # ACC + DMEM[SP + operand] -> ACC
                # ACC < 0, ACC = 0, ACC > 0 -> FLAGS
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__SP_OFFSET, latch_da=True))
                self.dp.handle_signals(
                    Signals(
                        oe=True,
                        alu_sel=ALU_SEL__DATA_OUT,
                        alu_op=alu_op,
                        acc_sel=ACC_SEL__ALU_RES,
                        latch_acc=True,
                        latch_flags=True,
                    )
                )
            case OperandType.STACK_POINTER:
                # ACC + DMEM[DMEM[SP + operand]] -> ACC
                # ACC < 0, ACC = 0, ACC > 0 -> FLAGS
                self.dp.handle_signals(Signals(operand=instr.operand, da_sel=DA_SEL__SP_OFFSET, latch_da=True))
                self.dp.handle_signals(Signals(oe=True, da_sel=DA_SEL__DATA_OUT, latch_da=True))
                self.dp.handle_signals(
                    Signals(
                        oe=True,
                        alu_sel=ALU_SEL__DATA_OUT,
                        alu_op=alu_op,
                        acc_sel=ACC_SEL__ALU_RES,
                        latch_acc=True,
                        latch_flags=True,
                    )
                )


    def execute_jump(self, instr: Instruction) -> bool:
        match instr.opcode:
            case Opcode.JMP:
                # operand -> PC
                self.pc = instr.operand
                return True
            case Opcode.JE:
                # if ZF = 1 then operand -> PC
                if self.dp.zf:
                    self.pc = instr.operand
                    return True
            case Opcode.JG:
                # if PF = 1 then operand -> PC
                if self.dp.pf:
                    self.pc = instr.operand
                    return True
            case Opcode.JGE:
                # if ZF = 1 or PF = 1 then operand -> PC
                if self.dp.zf or self.dp.pf:
                    self.pc = instr.operand
                    return True
            case Opcode.JL:
                # if NF = 1 then operand -> PC
                if self.dp.nf:
                    self.pc = instr.operand
                    return True
            case Opcode.JLE:
                # if ZF = 1 or NF = 1 then operand -> PC
                if self.dp.zf or self.dp.nf:
                    self.pc = instr.operand
                    return True
            case Opcode.JNE:
                # if ZF = 0 then operand -> PC
                if not self.dp.zf:
                    self.pc = instr.operand
                    return True
        return False


    def execute_push(self, instr: Instruction):
        # SP - 1 -> SP; ACC -> DMEM[SP]
        self.dp.handle_signals(Signals(sp_sel=SP_SEL__SP_DEC, latch_sp=True))
        self.dp.handle_signals(Signals(da_sel=DA_SEL__SP, latch_da=True))
        self.dp.handle_signals(Signals(data_sel=DATA_SEL__ACC, wr=True))


    def execute_pop(self, instr: Instruction):
        # SP + 1 -> SP
        self.dp.handle_signals(Signals(sp_sel=SP_SEL__SP_INC, latch_sp=True))        


    def execute_call(self, instr: Instruction):
        # SP - 1 -> SP;   PC + 1 -> DMEM[SP];  operand -> PC
        self.dp.handle_signals(Signals(sp_sel=SP_SEL__SP_DEC, latch_sp=True))
        self.dp.handle_signals(Signals(da_sel=DA_SEL__SP, latch_da=True))
        self.dp.handle_signals(Signals(next_pc=self.pc + 1, data_sel=DATA_SEL__NEXT_PC, wr=True))
        self.pc = instr.operand


    def execute_ret(self, instr: Instruction):
        # DMEM[SP] -> PC; SP + 1 -> SP
        self.dp.handle_signals(Signals(da_sel=DA_SEL__SP, latch_da=True))
        self.pc = self.dp.read_data()
        self.dp.handle_signals(Signals(sp_sel=SP_SEL__SP_INC, latch_sp=True))


    def decode_and_execute_instruction(self):
        # fetch
        binary_code = self.mem[self.pc]
        instr = decode_instruction(binary_code)
        jumped = False
        match instr.opcode:
            case Opcode.LD:
                self.execute_ld(instr)
            case Opcode.ST:
                self.execute_st(instr)
            case Opcode.ADD | Opcode.SUB | Opcode.MUL | Opcode.DIV | Opcode.REM:
                self.execute_math(instr)
            case Opcode.JMP | Opcode.JE | Opcode.JG | Opcode.JGE | Opcode.JL | Opcode.JLE | Opcode.JNE:
                jumped = self.execute_jump(instr)
            case Opcode.PUSH:
                self.execute_push(instr)
            case Opcode.POP:
                self.execute_pop(instr)
            case Opcode.CALL:
                self.execute_call(instr)
                jumped = True
            case Opcode.RET:
                self.execute_ret(instr)
                jumped = True
            case Opcode.HLT:
                # (останов машины)
                raise StopIteration
        if not jumped:
            self.pc = self.pc + 1


# convert bytes into machine words
def pack_machine_words(data: list[int], word_size: int) -> list[int]:
    res = []
    temp = []
    for i in range(len(data)):
        temp.append(data[i])
        if len(temp) == word_size:
            value = int.from_bytes(temp, "little", signed=True)
            res.append(value)
            temp = []
    return res


def read_code(file_name: str) -> list[int]:
    file = open(file_name, "rb")
    content = bytearray(file.read())
    binary_code = []
    for byte in content:
        binary_code.append(byte)
    file.close()
    return pack_machine_words(binary_code, CODE_WORD_SIZE)


def read_data(file_name: str) -> list[int]:
    file = open(file_name, "rb")
    content = bytearray(file.read())
    mem = []
    for byte in content:
        mem.append(byte)
    file.close()
    return mem


def read_input(file_name: str) -> list[str]:
    file = open(file_name)
    content = file.read().strip()
    file.close()
    return list(content) + ["\0"]


def read_debug_info(file_name) -> list[str]:
    file = open(file_name)
    content = file.read()
    file.close()
    return content.split("\n")


def parse_debug_info(debug_info: list[str]) -> list[str]:
    res = []
    for line in debug_info:
        res.append(line.split(";")[1][1:])
    return res


def log_state(dp: DataPath, cu: ControlUnit):
    stack = []
    for i in range(len(dp.mem) - dp.sp):
        stack.append(dp.mem[-(i + 1)])
    # logging.debug(
    #     "PC=%08d ACC=%08x FLAGS=[NF=%d,ZF=%d,PF=%d] DA=%08x SP=%08x"
    #     % (cu.pc, dp.acc, dp.nf, dp.zf, dp.pf, dp.da, dp.sp)
    # )
    # logging.debug("stack=%s input_stream=%s output_stream=%s" % (stack, dp.input_stream, dp.output_stream))


def log_instruction(cu: ControlUnit, source_lines: list[str] | None):
    addr: int = cu.pc
    binary_code: int = cu.mem[addr]
    instr: Instruction = decode_instruction(binary_code)
    instr_bytes = binary_code.to_bytes(CODE_WORD_SIZE, "little", signed=True)
    hex_code = ""
    for byte in instr_bytes:
        hex_code += hex(byte)[2:].rjust(2, "0")
    if source_lines is not None:
        logging.debug(
            "%04d: %s %s; %s" % (addr, hex_code, instruction_to_string(instr).ljust(17, " "), source_lines[addr])
        )
    else:
        logging.debug("%04d: %s %s" % (addr, hex_code, instruction_to_string(instr).ljust(17, " ")))


def simulate(
    binary_name: str,
    memory_name: str,
    input_name: str | None,
    debug_name: str | None,
    limit: int = 0,
    logging_level: str = "info",
):
    num_of_instrs = 0
    try:
        match logging_level:
            case "debug":
                logging.getLogger().setLevel(logging.DEBUG)
            case "info":
                logging.getLogger().setLevel(logging.INFO)
            case "warning":
                logging.getLogger().setLevel(logging.WARNING)
            case "error":
                logging.getLogger().setLevel(logging.ERROR)
        code_mem: list[int] = read_code(binary_name)
        data_mem: list[int] = read_data(memory_name)
        if input_name is not None:
            input_stream = read_input(input_name)
        else:
            input_stream = []
        debug_info: list[str] | None
        source_lines: list[str] | None
        if debug_name is not None:
            debug_info = read_debug_info(debug_name)
            source_lines = parse_debug_info(debug_info)
        else:
            debug_info = None
            source_lines = None
        dp = DataPath(data_mem + [0] * STACK_SIZE, input_stream)
        cu = ControlUnit(dp, code_mem)
        log_state(dp, cu)
        while True:
            log_instruction(cu, source_lines)
            cu.decode_and_execute_instruction()
            num_of_instrs += 1
            log_state(dp, cu)
            if limit > 0 and num_of_instrs == limit:
                logging.warning("instructions limit reached")
                break
        logging.info("%d instructions executed", num_of_instrs)
        logging.info("Final output stream: %s", dp.output_stream)
    except StopIteration:
        num_of_instrs += 1  # account for HLT instruction
        logging.info("%d instructions executed", num_of_instrs)
        logging.info("Final output stream: %s", dp.output_stream)
    except EOFError:
        logging.warning("end of input stream reached")
        logging.info("Final output stream: %s", dp.output_stream)
    except Exception:
        logging.exception("Simulation error")


if __name__ == "__main__":
# if False:
    import argparse
    arg_parser = argparse.ArgumentParser(prog="simulator", description="Accumulator architecture CPU simulator")
    arg_parser.add_argument("binary_name", help="input file with binary code")
    arg_parser.add_argument("memory_name", help="input file with static memory content")
    arg_parser.add_argument("input_name", nargs="?", help="input file for simulated input")
    arg_parser.add_argument("debug_name", nargs="?", help="input file with debug information")
    arg_parser.add_argument("--limit", type=int, default=0, nargs="?", help="maximum number of insturctions to execute")
    arg_parser.add_argument("--logging-level", type=str, default="info", nargs="?", help="logging level: error, warning, info (default) or debug")
    args = arg_parser.parse_args()
    simulate(args.binary_name, args.memory_name, args.input_name, args.debug_name, args.limit, args.logging_level)

########################################################
########################################################
########################################################
########################################################
import doctest

doctest.testmod()
# import os; os.system("mypy --no-error-summary " + __file__)
import os

os.system("ruff --ignore=W291,E402,I001,E721,RET505,RET506,RUF005,TRY003,UP031 --no-cache check " + __file__)
# E721 - isinstance vs type
# RET505 - else after return 
# RET506 -  Unnecessary `elif` after `raise` statement
# RUF005 - about concatenation vs *list
# TRY003 - custom exception classes
# UP031 - Use format specifiers instead of percent format
