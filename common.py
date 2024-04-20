from __future__ import annotations
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


def is_jump_instruction_opcode(opcode: Opcode):
    jumps = {Opcode.JMP, Opcode.JE, Opcode.JG, Opcode.JL, Opcode.JLE, Opcode.JGE, Opcode.JNE}
    return opcode in jumps


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
