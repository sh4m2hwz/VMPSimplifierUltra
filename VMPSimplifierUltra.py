from triton import *
from arybo.lib import *
from arybo.lib.exprs_asm import *
from arybo.tools import *
import llvmlite.binding as llvm
import llvmlite.ir as ll

def emulate(ctx):
    rip = ctx.getConcreteRegisterValue(ctx.registers.rip)
    while True:
        opcodes = ctx.getConcreteMemoryAreaValue(rip,16)
        insn = Instruction(rip,opcodes)
        ctx.disassembly(insn)
        opcode = insn.getType()
        ops = insn.getOperands()
        if opcode == OPCODE.X86.TEST or opcode == OPCODE.X86.CMP:
            rip+=insn.getSize()
            continue
        for op in insn.getOperands():
            if op.getType() == OPERAND.MEM:
                ctx.symbolizeMemory(op)
        ctx.processing(insn)
        if opcode == OPCODE.X86.RET or \
        opcode == OPCODE.X86.JMP and ops[0].getType() == OPERAND.REG:
            break
        rip = ctx.getConcreteRegisterValue(ctx.registers.rip)


def load(filename):
    code = open(filename,'rb').read()
    ctx = TritonContext()
    ctx.setArchitecture(ARCH.X86_64)
    ctx.setMode(MODE.AST_OPTIMIZATIONS,True)
    ctx.setMode(MODE.CONSTANT_FOLDING,True)
    ctx.setConcreteMemoryAreaValue(0x1000,code)
    ctx.setConcreteRegisterValue(ctx.registers.rip,0x1000)
    ctx.symbolizeRegister(ctx.registers.rax,'rax')
    ctx.symbolizeRegister(ctx.registers.rcx,'rcx')
    ctx.symbolizeRegister(ctx.registers.rdx,'rdx')
    ctx.symbolizeRegister(ctx.registers.rbx,'rbx')
    ctx.symbolizeRegister(ctx.registers.rsp,'rsp')
    ctx.symbolizeRegister(ctx.registers.rbp,'rbp')
    ctx.symbolizeRegister(ctx.registers.rsi,'rsi')
    ctx.symbolizeRegister(ctx.registers.rdi,'rdi')
    ctx.symbolizeRegister(ctx.registers.rip,'rip')
    ctx.symbolizeRegister(ctx.registers.r8,'r8')
    ctx.symbolizeRegister(ctx.registers.r9,'r9')
    ctx.symbolizeRegister(ctx.registers.r10,'r10')
    ctx.symbolizeRegister(ctx.registers.r11,'r11')
    ctx.symbolizeRegister(ctx.registers.r12,'r12')
    ctx.symbolizeRegister(ctx.registers.r13,'r13')
    ctx.symbolizeRegister(ctx.registers.r14,'r14')
    ctx.symbolizeRegister(ctx.registers.r15,'r15')
    emulate(ctx)
    triton_sexprs = ctx.getSymbolicExpressions()
    arybo_exprs = tritonexprs2arybo(triton_sexprs)
    return ctx, arybo_exprs

def lift_to_llvm(ctx, arybo_exprs):
    ast_vars = []
    for idx, symvar in ctx.getSymbolicVariables().items():
        ast_vars.append(tritonast2arybo(ctx.getAstContext().variable(symvar)).v)
    llvm_function = to_llvm_function(arybo_exprs,ast_vars,"_start")
    return str(llvm_function)

def back_to_native(llvm_ir,output_asm):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm_module = llvm.parse_assembly(llvm_ir)
    passes = llvm.ModulePassManager()
    passes.add_aggressive_dead_code_elimination_pass()
    passes.add_aggressive_instruction_combining_pass()
    passes.add_cfg_simplification_pass()
    passes.add_constant_merge_pass()
    passes.add_dead_arg_elimination_pass()
    passes.add_dead_code_elimination_pass()
    passes.add_dead_store_elimination_pass()
    passes.add_demote_register_to_memory_pass()
    passes.add_dependence_analysis_pass()
    passes.add_global_optimizer_pass()
    passes.add_instruction_combining_pass()
    passes.add_loop_deletion_pass()
    passes.add_loop_extractor_pass()
    passes.add_loop_rotate_pass()
    passes.add_loop_simplification_pass()
    passes.add_loop_strength_reduce_pass()
    passes.add_loop_unroll_and_jam_pass()
    passes.add_loop_unroll_pass()
    passes.add_lower_atomic_pass()
    passes.add_lower_invoke_pass()
    passes.add_lower_switch_pass()
    passes.add_memcpy_optimization_pass()
    passes.add_merge_functions_pass()
    passes.add_merge_returns_pass()
    passes.add_strip_dead_debug_info_pass()
    passes.add_strip_dead_prototypes_pass()
    passes.add_strip_debug_declare_pass()
    passes.add_strip_nondebug_symbols_pass()
    passes.run(llvm_module)
    tm = llvm.Target.from_default_triple().create_target_machine()
    with llvm.create_mcjit_compiler(llvm_module, tm) as ee:
        ee.finalize_object()
        open(output_asm,'w').write(tm.emit_assembly(llvm_module))

def main():
    from sys import argv
    ctx, exprs = load(argv[1])
    llvm_ir = lift_to_llvm(ctx, exprs)
    back_to_native(llvm_ir, argv[2])

if __name__ == "__main__":
    main()
