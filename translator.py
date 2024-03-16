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
                
def not_space(tok):
    if tok.value != ' ':
        return True
    else:
        return False

lst = list(filter(not_space, lst))

# print(lst)

def make_tree(toks):
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

def dump_tree(node, level = 0):
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

def wrap_program(tree):
    return [
        Token('defun', 1), 
        Token('_start', 1), 
        [],
        [Token('do', 1)] + tree
    ]

tree = wrap_program(tree)
# dump_tree(tree)

def rewrite_special_chars(node):   
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

def collect_string_literals(node):
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

def replace_string_literas(node, str_map):   
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

def replace_numbers(node, str_map):   
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

def process_makestring_forms(node, mem):
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

class VarRef:
    name: str
    def __init__(self, name):
        self.name = name
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
    
def extract_variables(node, temp_count=0):
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
                    res.append(VarRef(temp_name))
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

def collect_function_names(node):
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
print(func_names)

class FuncRef:
    name: str
    line: int | None
    def __init__(self, name, line=None):
        self.name = name
        self.line = line
    def __repr__(self):
        return "FuncRef(" + self.name + ")"

def rewrite_function_calls(node, func_names):
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
            el = FuncRef(el.value, el.line)
            new_node.append(el)
        else:
            new_node.append(el)        
    return new_node

# dump_tree(tree)
# dump_tree(rewrite_function_calls(tree, func_names))

def rewrite_getchar(node):
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
