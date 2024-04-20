from __future__ import annotations
import argparse
from common import *
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
    lines = content.split("\n")[1:]  # drop "CODE:" header
    # drop static memory debug info (not used in simulator)
    for i in range(len(lines)):
        if lines[i] == "DATA:":
            return lines[:i]
    raise ValueError("Incorrect debug info file format")


def parse_debug_info(debug_info: list[str]) -> list[str]:
    res = []
    for line in debug_info:
        res.append(line.split(";")[1][1:])
    return res


def log_state(dp: DataPath, cu: ControlUnit, debug_level: int):
    stack = []
    for i in range(len(dp.mem) - dp.sp):
        stack.append(dp.mem[-(i + 1)])
    if debug_level == 2:
        logging.debug(
            "PC=%08d ACC=%08x FLAGS=[NF=%d,ZF=%d,PF=%d] DA=%08x SP=%08x"
            % (cu.pc, dp.acc, dp.nf, dp.zf, dp.pf, dp.da, dp.sp)
        )
        logging.debug("stack=%s input_stream=%s output_stream=%s" % (stack, dp.input_stream, dp.output_stream))


def log_instruction(cu: ControlUnit, source_lines: list[str] | None, debug_level: int):
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
    debug_level: int = 2
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
        log_state(dp, cu, debug_level)
        while True:
            log_instruction(cu, source_lines, debug_level)
            cu.decode_and_execute_instruction()
            num_of_instrs += 1
            log_state(dp, cu, debug_level)
            if limit > 0 and num_of_instrs == limit:
                logging.warning("instructions limit reached")
                break
        print("%d instructions executed" % num_of_instrs)
        print("Final output stream: %s" % dp.output_stream)
    except StopIteration:
        num_of_instrs += 1  # account for HLT instruction
        print("%d instructions executed" % num_of_instrs)
        print("Final output stream: %s" % dp.output_stream)
    except EOFError:
        logging.warning("end of input stream reached")
        print("%d instructions executed" % num_of_instrs)
        print("Final output stream: %s" % dp.output_stream)
    except Exception:
        logging.exception("Simulation error")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="simulator", description="Accumulator architecture CPU simulator")
    arg_parser.add_argument("binary_name", help="input file with binary code")
    arg_parser.add_argument("memory_name", help="input file with static memory content")
    arg_parser.add_argument("input_name", nargs="?", help="input file for simulated input")
    arg_parser.add_argument("debug_name", nargs="?", help="input file with debug information")
    arg_parser.add_argument("--limit", type=int, default=0, nargs="?", help="maximum number of insturctions to execute")
    arg_parser.add_argument(
        "--logging-level",
        type=str,
        default="info",
        nargs="?",
        help="logging level: error, warning, info (default) or debug",
    )
    arg_parser.add_argument(
        "--debug-level",
        type=int,
        default=2,
        nargs="?",
        help="debug level: 1 - parial infomation, 2 - full information (defualt)",
    )
    args = arg_parser.parse_args()
    simulate(args.binary_name, args.memory_name, args.input_name, args.debug_name, args.limit, args.logging_level, args.debug_level)
