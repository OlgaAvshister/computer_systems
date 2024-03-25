"""
(define x 123)
(define s "unknown")
(+ (* 2 3) (- 5 7))
(* x (- 10 (/ y 2))) 
(= (% x 3) 0)
(= (% x 5) 0)
# < > <= >= !=
(define m (if (> x 0) x (* x -1)))
(defun abs (x) (if (> x 0) x (* x -1)))
(makestring 100)
(readchar)
(readstring)
(setchar s 5 ch)
(getchar s 0)
(printchar c)
(printnumber x)
(printstring s)

ld immediate       # ld 3
ld address         # ld [3]
ld stack_offset
ld pointer
ld stack_offset_pointer
st address         
st pointer
st stack_offset_pointer
add immediate     
add address
add stack_offset
add pointer
add stack_offset_pointer
sub, mul, div, rem 
cmp immediate
cmp address
jmp code_address
jl code_address
je code_address
jg code_address
jle code_address
jge code_address
jne code_address
call code_address
ret (no argument)
push immediate
push (no argument)
pop (no argument)
in (no argument?)
out (no argument?)
hlt (no argument)
"""
from enum import Enum
prog_line = input()
source = []
while prog_line.strip() != "":
    source.append( prog_line )
    prog_line = input()
s = "\n".join(source)
print(s)
print()

class Token:
    value: str
    line: int | None
    def __init__(self, value, line=None):
        self.value = value
        self.line = line
    def __repr__(self):
        return "Token(" + self.value + ")"
        # return '"' + self.value + '" ' + str(self.line)

# split string to chars
lst = []
line = 1
for i in range(len(s)):
    if s[i] != '\n':
        tok = Token(s[i], line)
        lst.append(tok)
    if s[i] == '\n':
        line += 1
# print(lst)    

# combine string literals "..."
i = 0
while i < len(lst):
    if lst[i].value == '"':
        while lst[i + 1].value != '"':
            lst[i].value += lst[i + 1].value
            del lst[i + 1]
        lst[i].value += '"'
        del lst[i + 1]
    i += 1

# combine digits into numbers and letters into names
for j in range(len(lst)):
    for i in range(len(lst) - 1):
        if lst[i].value[-1].isdigit():
            if lst[i + 1].value.isdigit():
                lst[i].value += lst[i + 1].value
                del lst[i + 1]
                break
        if lst[i].value == '-':
            if lst[i + 1].value.isdigit():
                lst[i].value += lst[i + 1].value
                del lst[i + 1]
                break
        if lst[i].value.isalpha():
            if lst[i + 1].value.isalpha():
                lst[i].value += lst[i + 1].value
                del lst[i + 1]
                break               
                
def not_space(tok: Token):
    if tok.value != ' ':
        return True
    else:
        return False

lst = list(filter(not_space, lst))

# print(lst)

def make_tree(toks: list[any]):
    lst = []
    stack = []
    for tok in toks:
        if tok.value == '(':
            stack.append(lst)
            lst = []
        elif tok.value == ')':
            stack[-1].append(lst)
            lst = stack.pop()            
        else:
            lst.append(tok)
    return lst
    
tree = make_tree(lst)    
# print(tree)

def dump_tree(node: any, level: int = 0):
    indent = '.   ' * level
    if type(node) is list:
        if [subnode for subnode in node if type(subnode) is list]:
            print(indent + '[')
            for el in node:
                dump_tree(el, level + 1)
            print(indent + ']')
        else:
            print(indent + repr(node))
    else:
        print(indent + repr(node))

def wrap_program(tree: list[any]):
    return [
        Token('defun', 1), 
        Token('_start', 1), 
        [],
        [Token('do', 1)] + tree
    ]

tree = wrap_program(tree)
# dump_tree(tree)

def rewrite_special_chars(node: list[any]):   
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

tree = rewrite_special_chars(tree)
# print(tree)

def collect_string_literals(node: list[any]):
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

str_lits = collect_string_literals(tree)
# print(str_lits)

def make_string_map(lits: list[str]):
    dic = {}
    addr = 2  # first two address are reserved for input/output
    for lit in lits:
        dic[lit] = addr
        addr += len(lit) + 1
    return dic
        
str_map = make_string_map(str_lits)
# print(str_map)

def generate_static_memory(str_map: dict):
    mem = [0, 0]
    for key in str_map:
        for i in range(len(key)):
            mem.append(ord(key[i]))
        mem.append(0)
    return mem

mem = generate_static_memory(str_map)
# print(mem)

def replace_string_literas(node: list[any], str_map: dict):   
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
    
# print(str_map)
tree = replace_string_literas(tree, str_map)
# print(tree)

def replace_numbers(node: list[any], str_map: dict):   
    new_node = []
    for el in node:
        if type(el) is list:
            new_el = replace_numbers(el, str_map)
            new_node.append(new_el)
        else:
            if type(el) is Token and el.value.isnumeric():
                num = int(el.value)
                new_node.append(num)
            else:
                new_node.append(el)
    return new_node

tree = replace_numbers(tree, str_map)
# print(tree)

def is_token(obj: any, check_for: str=None):
    if check_for is None:
        return type(obj) is Token 
    else:
        return type(obj) is Token and obj.value == check_for  

def process_makestring_forms(node: any, mem: list[int]):
    if type(node) is int:
        return node
    elif type(node) is Token:
        return node
    elif type(node) is list and node and is_token(node[0], 'makestring'):
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

# print(tree)
tree = process_makestring_forms(tree, mem)
# print(tree)
# print(mem)

#tddo: remove
# def add(x: int, y: int):
#     """
#     >>> x = add(2, 2)
#     >>> y = add(2, 2)
#     >>> add(x, y)
#     8
#     """
#     return x + y

# (if 1 2 3)  ------>  (if 1 (do 2) (do 3))
# def wrap_bodies(tree):
    # return tree
    
# tree = wrap_bodies(tree)
# print(tree)    

def rewrite_getchar(node: any):
    if type(node) is list and node and is_token(node[0], 'getchar'):
        head = node[0]
        res = []
        res.append(head)
        res.append([
            Token('+', head.line), 
            rewrite_getchar(node[1]), 
            rewrite_getchar(node[2])
        ])
        return res
    elif type(node) is list:
        res = []
        for el in node:
            new_el = rewrite_getchar(el)
            res.append(new_el)
        return res
    else:
        return node

tree = rewrite_getchar(tree)
# print(rewrite_getchar(tree))

def rewrite_setchar(node: any):
    if type(node) is list and node and is_token(node[0], 'setchar'):
        head = node[0]
        res = []
        res.append(head)
        res.append([
            Token('+', head.line), 
            rewrite_setchar(node[1]), 
            rewrite_setchar(node[2])
        ])
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
    
tree = rewrite_setchar(tree)    

class VarRef:
    name: str
    line: int
    def __init__(self, name, line):
        self.name = name
        self.line = line
    def __repr__(self):
        return "VarRef(" + str(self.name) + ")"

def is_operation_string(obj: any):
    return type(obj) is str and obj in {'+', '-', '*', '/', '%', '=', '>', '<', '>=', '<=', '!='}

def is_operation_token(obj: any):
    return type(obj) is Token and is_operation_string(obj.value)
    
def is_operation_expr(obj: any):
    return type(obj) is list and obj and is_operation_token(obj[0])   

def is_keyword_string(obj: any):
    return type(obj) is str and obj in {'do', 'if', 'define', 'defun'}

def is_keyword_token(obj: any):
    return type(obj) is Token and is_keyword_string(obj.value)
    
def is_keyword_expr(obj: any):
    return type(obj) is list and obj and is_keyword_token(obj[0])       
    
def is_call_expr(obj: any):
    return type(obj) is list and obj and not is_keyword_token(obj[0])  
    
def is_extractable_expr(obj: any):
    return is_operation_expr(obj) or is_call_expr(obj)
    
def extract_variables(node: any, temp_count: int = 0):
    if type(node) is int:
        return node, temp_count
    elif type(node) is Token:
        return node, temp_count
    else:
        assert type(node) is list
        if is_extractable_expr(node):
            res = []
            defines = []
            for el in node:
                if is_extractable_expr(el):
                    temp_name = '_temp' + str(temp_count)
                    define = [Token('define'), Token(temp_name), el]
                    temp_count += 1
                    res.append(VarRef(temp_name, node[0].line))
                    defines.append(define)
                else:
                    res.append(el)
            if len(defines) > 0:
                # print("defines not empty")
                wrapper = [Token('do')]
                for define in defines:
                    wrapper.append(define)
                wrapper.append(res)
                return wrapper, temp_count
            else:
                return res, temp_count
        else:  # special form
            res = []  
            for el in node:
                el, temp_count = extract_variables(el, temp_count)
                res.append(el) 
            return res, temp_count
    
# print(tree)    
# print()
tree, _ = extract_variables(tree)
# dump_tree(tree)

def collect_function_names(node: list[any]):
    names = []
    tick = 0
    for el in node:
        if type(el) is list:
            el_names = collect_function_names(el)
            for name in el_names:
                names.append(name)
        elif type(el) is Token and el.value == 'defun':
            tick = 1
        elif type(el) is Token and tick == 1:
            names.append(el.value)
            tick = 0
    return names

func_names = collect_function_names(tree)

class FuncRef:
    name: str
    index: int
    line: int | None
    def __init__(self, name: str, index: int, line=None):
        self.name = name
        self.index = index
        self.line = line
    def __repr__(self):
        return "FuncRef(" + self.name + ")"

def rewrite_function_calls(node: list[any], func_names: list[str]):
    tick = 0
    new_node = []
    for el in node:
        if type(el) is list:
            new_node.append(
                rewrite_function_calls(el, func_names)
            )
        elif type(el) is Token and el.value == 'defun':
            tick = 1
            new_node.append(el)
        elif type(el) is Token and tick == 1:
            tick = 0
            new_node.append(el)
        elif type(el) is Token and tick == 0 and el.value in func_names:
            for i in range(len(func_names)):
                if func_names[i] == el.value:
                    index = i
            el = FuncRef(el.value, index, el.line)
            new_node.append(el)
        else:  # not a list
            new_node.append(el)        
    return new_node

# dump_tree(tree)
tree = rewrite_function_calls(tree, func_names)

def count_own_vars(node: any): 
    if is_token(node, 'define'): 
        return 1
    elif type(node) is list and is_token(node[0], 'if'):
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
    body: any
    var_count: int
    def __init__(self, name, args, body, var_count):
        self.name = name
        self.args = args
        self.body = body
        self.var_count = var_count
    def __repr__(self):
        return 'Function(' + self.name + ', ' + str(self.args) + ')'

def do_collect_functions(node: any):
    if type(node) is list and node and is_token(node[0], 'defun'):
        args = []
        for tok in node[2]:
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
    
funcs = collect_functions(tree)
# for func in collect_functions(tree):
#     print(func)
#     print(func.body)
#     print(func.var_count)
#     print()

class ArgRef:
    name: str
    line: int
    def __init__(self, name, line: int):
        self.name = name
        self.line = line
    def __repr__(self):
        return "ArgRef(" + str(self.name) + ")"    
    
def rewrite_arguments(node: any, args: list[str]):
    if type(node) is int:
        return node
    elif type(node) is Token:
        if node.value in args:
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
    
for i in range(len(funcs)):
    # print(i, funcs[i].name, funcs[i].args)
    funcs[i].body = rewrite_arguments(funcs[i].body, funcs[i].args)
    # print(funcs[i].body)

def rewrite_vars(node: any, names: list[str]):
    if type(node) is list and node and is_token(node[0], 'define'):
        names.append(node[1].value)
        res = []
        res.append(node[0])      # define
        res.append(node[1])      # varaible name
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
    
for i in range(len(funcs)):
    # print(i, funcs[i].name, funcs[i].args)
    # print("rewrite_vars")
    # print(funcs[i].body)
    funcs[i].body = rewrite_vars(funcs[i].body, [])
    # print(funcs[i].body)

class If:
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
        return 'If(%d %d %d)' % (
            self.cond_vars, 
            self.true_branch_vars, 
            self.false_branch_vars
        )
    
def rewrite_ifs(node: any):
    if type(node) is list and node and is_token(node[0], 'if'):
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
    
for i in range(len(funcs)):
    funcs[i].body = rewrite_ifs(funcs[i].body)
    # print(funcs[i].body)
    # print()
    
class StackOffset:
    offset: int
    line: int
    def __init__(self, offset: int, line: int):
        self.offset = offset
        self.line = line
    def __repr__(self):
        return 'StackOffset(' + str(self.offset) + ')'
    
def rewrite_var_refs(node: any, stack: list[str]):
    if type(node) is ArgRef or type(node) is VarRef:
        reversed_stack = stack[::-1]
        for i in range(len(reversed_stack)):
            if reversed_stack[i] == node.name:
                return StackOffset(i, node.line)
        assert False
    elif type(node) is list and is_token(node[0], 'define'):
        res = []
        res.append(node[0])     # define
        res.append(node[1])     # variable name
        stack.append(node[1].value)
        # print(stack)
        for el in node[2:]:
            res.append(rewrite_var_refs(el, stack))
        return res
    elif type(node) is list and node and type(node[0]) is If:
        res = []
        for el in node:
            old_len = len(stack)
            res.append(rewrite_var_refs(el, stack))        
            while len(stack) > old_len:
                stack.pop()
        return res
    elif type(node) is list:
        res = []
        for el in node:
            res.append(rewrite_var_refs(el, stack))
        return res
    else:
        return node

for i in range(len(funcs)):
    stack = funcs[i].args
    # stack.append('<RA>')
    # print(funcs[i].body)
    funcs[i].body = rewrite_var_refs(funcs[i].body, stack)    
    # print(funcs[i].body)

def is_comparison_op_string(s: str):
    if s in {'<', '>', '=', '!=', '<=', '>='}:
        return True
    else:
        return False

def is_math_op_string(s: str):
    if s in {'+',  '-',  '*',  '/',  '%'}:
        return True
    else:
        return False    
    
class Opcode(Enum):
    LD = 0
    ST = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    REM = 6
    JMP = 7
    JG = 8
    JE = 9
    JGE = 10
    JL = 11
    JLE = 12
    JNE = 13
    PUSH = 14
    POP = 15
    CALL = 16
    RET = 17
    HLT = 18

def math_opcode(s: str):
    if s == '+':
        return Opcode.ADD
    elif s == '-':
        return Opcode.SUB
    elif s == '*':
        return Opcode.MUL
    elif s == '/':
        return Opcode.DIV
    else:
        return Opcode.REM
    
def comparison_opcode(s: str):
    if s == '>':
        return Opcode.JG
    elif s == '<':
        return Opcode.JL
    elif s == '=':
        return Opcode.JE
    elif s == '!=':
        return Opcode.JNE    
    elif s == '>=':
        return Opcode.JGE
    else:
        return Opcode.JLE
    
# def mark_tail_calls(func_name, node):
    
    
# defun f    (do
#     (f ...)
#     (f ...)
#     if 
#       (...)
#       (do (f 1 2 3) 123)
#       (f 1 2 3)     *
#            )
    
class OperandType(Enum):
    NONE = 0
    IMMEDIATE = 1
    ADDRESS = 2
    STACK_OFFSET = 3
    STACK_POINTER = 4    
    
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
        return 'Instruction(%s, %s, %d)' % (
            self.opcode, 
            self.operand_type,
            self.operand
        )

def offset_stack_refs(node: Any, delta: int):
    if type(node) is int:
        return node
    elif type(node) is StackOffset:
        return StackOffset(node.offset + delta, node.line)
    elif type(node) is list:
        res = []
        for el in node:
            res.append(offset_stack_refs(el, delta))
        return res
    else:
        return node
            
def mark_tail_calls(node, func_name):
    if type(node) is int:
        return
    elif type(node) is FuncRef:
        raise Exception('Incorrect program')
    elif type(node) is If:
        return
    elif type(node) is list:
        if node:
            head = node[0]
            if type(head) is FuncRef and head.name == func_name:
                head.tail_call = True   
            elif type(head) is If:
                mark_tail_calls(node[2], func_name)
                mark_tail_calls(node[3], func_name)
            elif is_token(head, 'do') and len(node) > 1:
                mark_tail_calls(node[-1], func_name)
    else:
        raise Exception('Incorrect program')
        
for i in range(len(funcs)):
    # print(funcs[i].name)
    # dump_tree(funcs[i].body)
    mark_tail_calls(funcs[i].body, funcs[i].name)    
    # dump_tree(funcs[i].body)
    # for el in code:
    #     print(el)
    # print()
    # print()        
    
def calc_tail_calls_stack_size(node, stack):
    if type(node) is list and is_token(node[0], 'define'):
        for el in node[2:]:  # skip `define` and variable name
            calc_tail_calls_stack_size(el, stack)
        stack.append(node[1].value)
    elif type(node) is list and type(node[0]) is If:
        for el in node:
            old_len = len(stack)
            calc_tail_calls_stack_size(el, stack)
            while len(stack) > old_len:
                stack.pop()
    elif type(node) is list and type(node[0]) is FuncRef and node[0].tail_call:
        node[0].tail_call_stack_size = len(stack) 
    elif type(node) is list:
        for el in node:
            calc_tail_calls_stack_size(el, stack)
    else:
        return node    
    
for i in range(len(funcs)):
    print(funcs[i].name)
    # dump_tree(funcs[i].body)
    calc_tail_calls_stack_size(funcs[i].body, [])    
    dump_tree(funcs[i].body)
    # for el in code:
    #     print(el)
    print()
    print()            
    
INPUT_PORT_ADDR = 0
OUTPUT_PORT_ADDR = 1        

def generate_instructions(node: Any, line=None) -> list[Instruction]:
    # print("node", node)
    if type(node) is int:
        return [Instruction(Opcode.LD, OperandType.IMMEDIATE, node, line)]
    elif type(node) is StackOffset:
        return [Instruction(Opcode.LD, OperandType.STACK_OFFSET, node.offset, node.line)]
    elif type(node) is list and is_token(node[0], 'define'):
        res = generate_instructions(node[2], node[0].line)
        res.append(Instruction(Opcode.PUSH, OperandType.NONE, 0, node[0].line))
        return res
    elif type(node) is list and is_token(node[0]) and is_math_op_string(node[0].value):
        #todo: handle non-ints operands
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        opcode = math_opcode(node[0].value)
        if type(node[2]) is int:
            res.append(Instruction(opcode, OperandType.IMMEDIATE, node[2], node[0].line))
        else:
            res.append(Instruction(opcode, OperandType.STACK_OFFSET, node[2].offset, node[0].line))
        return res
    elif type(node) is list and is_token(node[0]) and is_comparison_op_string(node[0].value):
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        if type(node[2]) is int:
            res.append(Instruction(Opcode.SUB, OperandType.IMMEDIATE, node[2], node[0].line))
        else:
            res.append(Instruction(Opcode.SUB, OperandType.STACK_OFFSET, node[2].offset, node[0].line))
        opcode = comparison_opcode(node[0].value)
        res.extend([
            Instruction(opcode, OperandType.ADDRESS, 2, node[0].line),
            Instruction(Opcode.LD, OperandType.IMMEDIATE, 0, node[0].line),
            Instruction(Opcode.JMP, OperandType.ADDRESS, 1, node[0].line),
            Instruction(Opcode.LD, OperandType.IMMEDIATE, 1, node[0].line)
        ])
        return res
    elif type(node) is list and is_token(node[0], 'do'):
        res = []
        for el in node[1:]:
            res.extend(generate_instructions(el, node[0].line))
        return res
    elif type(node) is list and type(node[0]) is If:
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        res.extend(
            [Instruction(Opcode.POP, OperandType.NONE, 0, node[0].line)] * node[0].cond_vars
        )
        res.append(Instruction(Opcode.JNE, OperandType.ADDRESS, 0, node[0].line))
        jne_index = len(res) - 1
        res.extend(generate_instructions(node[2], node[0].line))
        res[jne_index].operand = len(res) - jne_index
        res.extend(
            [Instruction(Opcode.POP, OperandType.NONE, 0, node[0].line)] * node[0].true_branch_vars
        )
        res.append(Instruction(Opcode.JMP, OperandType.ADDRESS, 0, node[0].line))
        jmp_index = len(res) - 1
        res.extend(generate_instructions(node[3], node[0].line))
        res.extend(
            [Instruction(Opcode.POP, OperandType.NONE, 0, node[0].line)] * node[0].false_branch_vars
        )        
        res[jmp_index].operand = len(res) - jmp_index - 1
        return res
    elif type(node) is list and is_token(node[0], 'getchar'):
        res = []
        res.append(Instruction(Opcode.LD, OperandType.STACK_POINTER, node[1].offset, node[0].line))
        return res
    elif type(node) is list and is_token(node[0], 'setchar'):
        res = []
        res.extend(generate_instructions(node[2], node[0].line))
        res.append(Instruction(Opcode.ST, OperandType.STACK_POINTER, node[1].offset, node[0].line))
        return res
    elif type(node) is list and is_token(node[0], 'readchar'):
        res = []
        res.append(Instruction(Opcode.LD, OperandType.ADDRESS, INPUT_PORT_ADDR, node[0].line))
        return res
    elif type(node) is list and is_token(node[0], 'printchar'):
        res = []
        res.extend(generate_instructions(node[1], node[0].line))
        res.append(Instruction(Opcode.ST, OperandType.ADDRESS, OUTPUT_PORT_ADDR, node[0].line))
        return res
    elif type(node) is list and type(node[0]) is FuncRef:
        func = node[0]
        args_count = len(node[1:])
        if func.tail_call:
            res = []
            extra_vars = 0
            for i in range(args_count):
                arg_code = generate_instructions(node[i + 1], node[0].line)
                extra_vars += count_own_vars(node[i + 1])
                res.extend(arg_code)
                extra_offset = func.tail_call_stack_size + extra_vars
                arg_addr = (args_count - 1) - i + extra_offset
                res.append(Instruction(Opcode.ST, OperandType.STACK_OFFSET, arg_addr, node[0].line))
            # func.tail_call_stack_size
            # clean up all the variables created before the tail call            
            size_to_clean = func.tail_call_stack_size + extra_vars
            res.extend(
                [Instruction(Opcode.POP, OperandType.NONE, 0, node[0].line)] * size_to_clean
            )                 
            res.append(Instruction(Opcode.JMP, OperandType.ADDRESS, 0, node[0].line))
            return res            
        else: 
            res = []
            for i in range(args_count):
                arg_code = generate_instructions(node[i + 1], node[0].line)
                arg_code = offset_stack_refs(arg_code, i)
                res.extend(arg_code)
                res.append(Instruction(Opcode.PUSH, OperandType.NONE, 0, node[0].line))
            res.append(Instruction(Opcode.CALL, OperandType.ADDRESS, node[0].index, node[0].line))
            res.extend(
                [Instruction(Opcode.POP, OperandType.NONE, 0, node[0].line)] * args_count
            ) 
            return res
    else:
        raise Exception("Incorrect program")
    
for i in range(len(funcs)):
    # dump_tree(funcs[i].body)
    code = generate_instructions(funcs[i].body)    
    if i == 0:
        code.append(Instruction(Opcode.HLT, OperandType.NONE, 0))
    else:
        code.append(Instruction(Opcode.RET, OperandType.NONE, 0))
    for el in code:
        print(el)
    print()
    print()    
        
def encode_instructions(ar_of_instr):
    code = []
    for instr in ar_of_instr:
        code.append(instr.operand.to_bytes(4, 'little', signed=True))
    return code

def wraparound(num):
    num_to_bytes = num.to_bytes(4, 'little', signed=True)
    return int.from_bytes(num_to_bytes, 'little', signed=True)

def write_code(code, file_name):
    file = open(file_name, "w")
    byte_code = bytearray(code)
    file.write(byte_code)
    file.close()
    
def read_source_code(file_name):
    file = open(file_name, 'r')
    content = file.read()
    file.close()
    
def instruction_to_string(instr):
    if instr.operand_type == IMMEDIATE:
        return str(instr.opcode) + str(instr.operand)
    elif instr.operand_type == ADDRESS:
        return str(instr.opcode) + str([instr.operand])
    elif instr.operand_type == STACK_OFFSET:
        return str(instr.opcode) + str([sp + instr.operand])
    elif instr.operand_type == STACK_POINTER:
        return str(instr.opcode) + str([[sp + instr.operand]])
    else:     
        return str(instr.opcode)
        
### simulator ###

class DataPath:
    acc: int
    da: int  # data address
    mem: list[int]
    sp: int  # stack pointer
    def __init__(self, mem: list[int]):
        self.acc = 0
        self.da = 0
        self.mem = mem
        self.sp = len(mem)
    
class ControlUnit:
    pc: int  # program counter
    mem: list[int]
    def __init__(self, mem: list[int]):
        self.pc = 0
        self.mem = mem

###########################################
import doctest; doctest.testmod()
# import os; os.system("mypy --no-error-summary " + __file__)        
        
