//! LLVM IR emission from MIR
//!
//! Emits LLVM IR text format that can be compiled with clang or llc.

use crate::error::{LlvmError, Result};
use crate::runtime::{emit_format_strings, emit_runtime_declarations};
use goth_ast::types::{PrimType, Type};
use goth_mir::mir::*;
use std::collections::{HashMap, HashSet};

/// LLVM IR emission context
pub struct LlvmContext {
    /// SSA value counter
    next_ssa: usize,
    /// Local variable to SSA value mapping (for SSA values)
    local_map: HashMap<LocalId, String>,
    /// Type information for locals
    local_types: HashMap<LocalId, Type>,
    /// String literal counter
    next_str: usize,
    /// Accumulated string literals
    string_literals: Vec<(String, String)>, // (name, value)
    /// Locals that use stack allocation (for control flow merging)
    stack_locals: HashMap<LocalId, String>, // LocalId -> alloca pointer name
    /// Function return type
    ret_ty: Type,
}

impl LlvmContext {
    pub fn new(ret_ty: Type) -> Self {
        LlvmContext {
            next_ssa: 0,
            local_map: HashMap::new(),
            local_types: HashMap::new(),
            next_str: 0,
            string_literals: Vec::new(),
            stack_locals: HashMap::new(),
            ret_ty,
        }
    }

    /// Check if a local uses stack allocation
    fn is_stack_local(&self, local: &LocalId) -> bool {
        self.stack_locals.contains_key(local)
    }

    /// Get the stack pointer for a local
    fn get_stack_ptr(&self, local: &LocalId) -> Option<&String> {
        self.stack_locals.get(local)
    }

    /// Register a stack-allocated local
    fn register_stack_local(&mut self, local: LocalId, ptr: String) {
        self.stack_locals.insert(local, ptr);
    }

    /// Generate fresh SSA value name
    fn fresh_ssa(&mut self) -> String {
        let name = format!("%{}", self.next_ssa);
        self.next_ssa += 1;
        name
    }

    /// Get SSA value for a local
    fn get_ssa(&self, local: &LocalId) -> Result<String> {
        self.local_map
            .get(local)
            .cloned()
            .ok_or_else(|| LlvmError::UndefinedLocal(format!("{:?}", local)))
    }

    /// Register local with SSA value
    fn register_local(&mut self, local: LocalId, ssa: String, ty: Type) {
        self.local_map.insert(local, ssa);
        self.local_types.insert(local, ty);
    }

    /// Add a string literal, return its global name
    fn add_string(&mut self, value: &str) -> String {
        let name = format!("@.str.{}", self.next_str);
        self.next_str += 1;
        self.string_literals.push((name.clone(), value.to_string()));
        name
    }
}

/// Check if a type is integer-like
fn is_int_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::I64 | PrimType::I32 | PrimType::I16 | PrimType::I8) => true,
        Type::Prim(PrimType::U64 | PrimType::U32 | PrimType::U16 | PrimType::U8) => true,
        Type::Prim(PrimType::Int | PrimType::Nat) => true,
        Type::Var(name) => matches!(name.as_ref(), "I" | "Int" | "ℤ" | "N" | "Nat" | "ℕ"),
        _ => false,
    }
}

/// Check if a type is float-like
fn is_float_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::F64 | PrimType::F32) => true,
        Type::Var(name) => matches!(name.as_ref(), "F" | "Float"),
        _ => false,
    }
}

/// Check if a type is boolean
fn is_bool_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::Bool) => true,
        Type::Var(name) => matches!(name.as_ref(), "B" | "Bool"),
        _ => false,
    }
}

/// Check if a type is string
fn is_string_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::String) => true,
        Type::Var(name) => matches!(name.as_ref(), "String" | "Str"),
        // [n]Char is also a string type
        Type::Tensor(_, elem_ty) => matches!(elem_ty.as_ref(), Type::Prim(PrimType::Char)),
        _ => false,
    }
}

/// Check if a type is unit/void (empty tuple)
fn is_unit_type(ty: &Type) -> bool {
    match ty {
        Type::Tuple(fields) if fields.is_empty() => true,
        _ => false,
    }
}

/// Escape a string for LLVM IR constant format
fn escape_string_for_llvm(s: &str) -> String {
    let mut result = String::new();
    for byte in s.bytes() {
        match byte {
            b'\\' => result.push_str("\\5C"),
            b'"' => result.push_str("\\22"),
            b'\n' => result.push_str("\\0A"),
            b'\r' => result.push_str("\\0D"),
            b'\t' => result.push_str("\\09"),
            0x20..=0x7E => result.push(byte as char),
            _ => result.push_str(&format!("\\{:02X}", byte)),
        }
    }
    // Add null terminator
    result.push_str("\\00");
    result
}

/// Analyze a function to find locals that need stack allocation.
/// A local needs stack allocation if:
/// 1. It is assigned in multiple blocks, OR
/// 2. It is assigned in one block and used in a different block
fn find_multi_block_locals(func: &Function) -> HashMap<LocalId, Type> {
    let mut assignments: HashMap<LocalId, HashSet<usize>> = HashMap::new();
    let mut uses: HashMap<LocalId, HashSet<usize>> = HashMap::new();
    let mut local_types: HashMap<LocalId, Type> = HashMap::new();

    // Helper to extract locals from an operand
    fn collect_operand_locals(op: &Operand, block_id: usize, uses: &mut HashMap<LocalId, HashSet<usize>>) {
        if let Operand::Local(local) = op {
            uses.entry(*local).or_default().insert(block_id);
        }
    }

    // Helper to extract locals from a Rhs
    fn collect_rhs_locals(rhs: &Rhs, block_id: usize, uses: &mut HashMap<LocalId, HashSet<usize>>) {
        match rhs {
            Rhs::Use(op) => collect_operand_locals(op, block_id, uses),
            Rhs::BinOp(_, left, right) => {
                collect_operand_locals(left, block_id, uses);
                collect_operand_locals(right, block_id, uses);
            }
            Rhs::UnaryOp(_, op) => collect_operand_locals(op, block_id, uses),
            Rhs::Call { args, .. } => {
                for arg in args {
                    collect_operand_locals(arg, block_id, uses);
                }
            }
            Rhs::ClosureCall { closure, args } => {
                collect_operand_locals(closure, block_id, uses);
                for arg in args {
                    collect_operand_locals(arg, block_id, uses);
                }
            }
            Rhs::Tuple(ops) => {
                for op in ops {
                    collect_operand_locals(op, block_id, uses);
                }
            }
            Rhs::TupleField(op, _) => collect_operand_locals(op, block_id, uses),
            Rhs::Array(ops) => {
                for op in ops {
                    collect_operand_locals(op, block_id, uses);
                }
            }
            Rhs::ArrayFill { size, value } => {
                collect_operand_locals(size, block_id, uses);
                collect_operand_locals(value, block_id, uses);
            }
            Rhs::Prim { args, .. } => {
                for arg in args {
                    collect_operand_locals(arg, block_id, uses);
                }
            }
            Rhs::MakeVariant { payload, .. } => {
                if let Some(op) = payload {
                    collect_operand_locals(op, block_id, uses);
                }
            }
            Rhs::MakeClosure { captures, .. } => {
                for op in captures {
                    collect_operand_locals(op, block_id, uses);
                }
            }
            Rhs::Const(_) => {}
            // Tensor operations
            Rhs::TensorMap { tensor, func } => {
                collect_operand_locals(tensor, block_id, uses);
                collect_operand_locals(func, block_id, uses);
            }
            Rhs::TensorReduce { tensor, .. } => {
                collect_operand_locals(tensor, block_id, uses);
            }
            Rhs::TensorFilter { tensor, pred } => {
                collect_operand_locals(tensor, block_id, uses);
                collect_operand_locals(pred, block_id, uses);
            }
            Rhs::TensorZip { left, right } => {
                collect_operand_locals(left, block_id, uses);
                collect_operand_locals(right, block_id, uses);
            }
            Rhs::Slice { array, start, end } => {
                collect_operand_locals(array, block_id, uses);
                if let Some(s) = start {
                    collect_operand_locals(s, block_id, uses);
                }
                if let Some(e) = end {
                    collect_operand_locals(e, block_id, uses);
                }
            }
            Rhs::ContractCheck { predicate, .. } => {
                collect_operand_locals(predicate, block_id, uses);
            }
            Rhs::Uncertain { value, uncertainty } => {
                collect_operand_locals(value, block_id, uses);
                collect_operand_locals(uncertainty, block_id, uses);
            }
            Rhs::Index(arr, idx) => {
                collect_operand_locals(arr, block_id, uses);
                collect_operand_locals(idx, block_id, uses);
            }
            Rhs::Iota(n) => {
                collect_operand_locals(n, block_id, uses);
            }
            Rhs::Range(start, end) => {
                collect_operand_locals(start, block_id, uses);
                collect_operand_locals(end, block_id, uses);
            }
            Rhs::GetTag(op) | Rhs::GetPayload(op) => {
                collect_operand_locals(op, block_id, uses);
            }
        }
    }

    // Helper to extract locals from a terminator
    fn collect_term_locals(term: &Terminator, block_id: usize, uses: &mut HashMap<LocalId, HashSet<usize>>) {
        match term {
            Terminator::Return(op) => collect_operand_locals(op, block_id, uses),
            Terminator::If { cond, .. } => collect_operand_locals(cond, block_id, uses),
            Terminator::Switch { scrutinee, .. } => collect_operand_locals(scrutinee, block_id, uses),
            Terminator::Goto(_) | Terminator::Unreachable => {}
        }
    }

    // Track assignments and uses in entry block (block 0)
    for stmt in &func.body.stmts {
        assignments.entry(stmt.dest).or_default().insert(0);
        local_types.insert(stmt.dest, stmt.ty.clone());
        collect_rhs_locals(&stmt.rhs, 0, &mut uses);
    }
    collect_term_locals(&func.body.term, 0, &mut uses);

    // Track assignments and uses in other blocks
    for (block_id, block) in &func.blocks {
        let bid = block_id.0 as usize;
        for stmt in &block.stmts {
            assignments.entry(stmt.dest).or_default().insert(bid);
            local_types.insert(stmt.dest, stmt.ty.clone());
            collect_rhs_locals(&stmt.rhs, bid, &mut uses);
        }
        collect_term_locals(&block.term, bid, &mut uses);
    }

    // A local needs stack allocation if:
    // 1. It's assigned in multiple blocks, OR
    // 2. It's used in a block different from where it's assigned
    let mut result = HashMap::new();
    for (local, assigned_blocks) in &assignments {
        let needs_alloca = if assigned_blocks.len() > 1 {
            // Assigned in multiple blocks
            true
        } else if let Some(used_blocks) = uses.get(local) {
            // Check if used in a different block than where assigned
            let assigned_block = *assigned_blocks.iter().next().unwrap();
            used_blocks.iter().any(|&b| b != assigned_block)
        } else {
            false
        };

        if needs_alloca {
            if let Some(ty) = local_types.get(local) {
                result.insert(*local, ty.clone());
            }
        }
    }

    result
}

/// Emit LLVM type
pub fn emit_type(ty: &Type) -> Result<String> {
    match ty {
        // Fixed-width integers
        Type::Prim(PrimType::I64) => Ok("i64".to_string()),
        Type::Prim(PrimType::I32) => Ok("i32".to_string()),
        Type::Prim(PrimType::I16) => Ok("i16".to_string()),
        Type::Prim(PrimType::I8) => Ok("i8".to_string()),
        Type::Prim(PrimType::U64) => Ok("i64".to_string()),
        Type::Prim(PrimType::U32) => Ok("i32".to_string()),
        Type::Prim(PrimType::U16) => Ok("i16".to_string()),
        Type::Prim(PrimType::U8) => Ok("i8".to_string()),

        // Floating point
        Type::Prim(PrimType::F64) => Ok("double".to_string()),
        Type::Prim(PrimType::F32) => Ok("float".to_string()),

        // Arbitrary precision (map to i64 for now)
        Type::Prim(PrimType::Int) => Ok("i64".to_string()),
        Type::Prim(PrimType::Nat) => Ok("i64".to_string()),

        // Other primitives
        Type::Prim(PrimType::Bool) => Ok("i1".to_string()),
        Type::Prim(PrimType::Char) => Ok("i32".to_string()), // Unicode scalar is 32-bit
        Type::Prim(PrimType::Byte) => Ok("i8".to_string()),
        Type::Prim(PrimType::String) => Ok("i8*".to_string()), // Pointer to null-terminated UTF-8

        Type::Tuple(fields) if fields.is_empty() => Ok("void".to_string()),

        Type::Tuple(fields) => {
            let field_types: Result<Vec<_>> = fields.iter().map(|f| emit_type(&f.ty)).collect();
            Ok(format!("{{ {} }}", field_types?.join(", ")))
        }

        Type::Tensor(_, elem) => {
            // Tensors are represented as pointers to heap-allocated arrays
            // Structure: { i64 len, elem* data }
            let _elem_ty = emit_type(elem)?;
            Ok("i8*".to_string()) // Opaque pointer for now
        }

        Type::Fn(arg, ret) => {
            let arg_ty = emit_type(arg)?;
            let ret_ty = emit_type(ret)?;
            Ok(format!("{} ({})*", ret_ty, arg_ty))
        }

        // Type variables - map to concrete LLVM types
        Type::Var(name) => match name.as_ref() {
            "I" | "Int" | "ℤ" => Ok("i64".to_string()),
            "F" | "Float" => Ok("double".to_string()),
            "B" | "Bool" => Ok("i1".to_string()),
            "N" | "Nat" | "ℕ" => Ok("i64".to_string()),
            _ => Ok("i64".to_string()),
        },

        // Optional type (nullable pointer)
        Type::Option(inner) => {
            let _inner_ty = emit_type(inner)?;
            Ok("i8*".to_string()) // Opaque pointer for nullable
        }

        // Effectful type (strip effects for codegen)
        Type::Effectful(inner, _) => emit_type(inner),

        // Interval type (strip constraints for codegen)
        Type::Interval(inner, _) => emit_type(inner),

        // Forall (use inner type)
        Type::Forall(_, inner) => emit_type(inner),

        // Exists (use inner type)
        Type::Exists(_, inner) => emit_type(inner),

        // Type application (use the base type)
        Type::App(base, _) => emit_type(base),

        // Variant (sum type) - represented as tagged union
        Type::Variant(_) => Ok("i8*".to_string()), // Pointer to tag + payload

        // Refinement type (strip predicate)
        Type::Refinement { base, .. } => emit_type(base),

        // Uncertain type
        Type::Uncertain(val_ty, _) => emit_type(val_ty),

        // Hole (should be resolved by type inference)
        Type::Hole => Ok("i64".to_string()),

        _ => Err(LlvmError::UnsupportedType(format!("{:?}", ty))),
    }
}

/// Emit LLVM constant
fn emit_constant(ctx: &mut LlvmContext, constant: &Constant, ty: &Type) -> Result<(String, String)> {
    let ssa = ctx.fresh_ssa();
    let llvm_ty = emit_type(ty)?;

    let code = match constant {
        Constant::Int(n) => {
            // No instruction needed for constants in LLVM - they're used inline
            // But we need to track the value
            return Ok((n.to_string(), String::new()));
        }
        Constant::Float(f) => {
            // LLVM requires floats to have a decimal point or be in hex format
            // Use scientific notation but ensure there's always a decimal point
            let s = format!("{:e}", f);
            // If the mantissa doesn't contain a decimal point, add ".0"
            let formatted = if !s.contains('.') {
                // Find 'e' and insert ".0" before it
                if let Some(e_pos) = s.find('e') {
                    format!("{}.0{}", &s[..e_pos], &s[e_pos..])
                } else {
                    format!("{}.0", s)
                }
            } else {
                s
            };
            return Ok((formatted, String::new()));
        }
        Constant::Bool(b) => {
            let val = if *b { "1" } else { "0" };
            return Ok((val.to_string(), String::new()));
        }
        Constant::String(s) => {
            // Add string to the global string literals
            // Return a GEP to get i8* from the array constant
            let name = ctx.add_string(s);
            let len = s.len() + 1; // Include null terminator
            let ssa = ctx.fresh_ssa();
            let code = format!(
                "  {} = getelementptr [{} x i8], [{} x i8]* {}, i64 0, i64 0\n",
                ssa, len, len, name
            );
            return Ok((ssa, code));
        }
        Constant::Unit => {
            // Unit/void type - return a marker that should be skipped in function calls
            // Using "undef" as a placeholder that won't be used
            return Ok(("undef".to_string(), String::new()));
        }
    };

    Ok((ssa, code))
}

/// Emit operand - returns the LLVM value representation
fn emit_operand(
    ctx: &mut LlvmContext,
    op: &Operand,
    ty: &Type,
    output: &mut String,
) -> Result<String> {
    match op {
        Operand::Const(c) => {
            let (val, code) = emit_constant(ctx, c, ty)?;
            if !code.is_empty() {
                output.push_str(&code);
            }
            Ok(val)
        }
        Operand::Local(local) => {
            // If this is a stack-allocated local, emit a load
            if let Some(ptr) = ctx.get_stack_ptr(local).cloned() {
                // Use the local's actual type, not the expected type
                let local_ty = ctx.local_types.get(local).cloned().unwrap_or_else(|| ty.clone());
                let llvm_ty = emit_type(&local_ty)?;
                if llvm_ty == "void" {
                    return Ok("undef".to_string());
                }
                let ssa = ctx.fresh_ssa();
                output.push_str(&format!("  {} = load {}, {}* {}\n", ssa, llvm_ty, llvm_ty, ptr));
                Ok(ssa)
            } else {
                ctx.get_ssa(local)
            }
        }
    }
}

/// Emit binary operation
fn emit_binop(
    ctx: &mut LlvmContext,
    op: &goth_ast::op::BinOp,
    left: &str,
    right: &str,
    ty: &Type,
    operand_ty: &Type,
) -> Result<(String, String)> {
    let ssa = ctx.fresh_ssa();
    let llvm_ty = emit_type(ty)?;
    let operand_llvm_ty = emit_type(operand_ty)?;

    let (op_name, result_ty) = if is_int_type(ty) {
        let op_name = match op {
            goth_ast::op::BinOp::Add => "add",
            goth_ast::op::BinOp::Sub => "sub",
            goth_ast::op::BinOp::Mul => "mul",
            goth_ast::op::BinOp::Div => "sdiv",
            goth_ast::op::BinOp::Mod => "srem",
            goth_ast::op::BinOp::Lt => "icmp slt",
            goth_ast::op::BinOp::Gt => "icmp sgt",
            goth_ast::op::BinOp::Leq => "icmp sle",
            goth_ast::op::BinOp::Geq => "icmp sge",
            goth_ast::op::BinOp::Eq => "icmp eq",
            goth_ast::op::BinOp::Neq => "icmp ne",
            _ => return Err(LlvmError::UnsupportedOp(format!("{:?}", op))),
        };
        let result_ty = match op {
            goth_ast::op::BinOp::Lt
            | goth_ast::op::BinOp::Gt
            | goth_ast::op::BinOp::Leq
            | goth_ast::op::BinOp::Geq
            | goth_ast::op::BinOp::Eq
            | goth_ast::op::BinOp::Neq => "i1",
            _ => &llvm_ty,
        };
        (op_name, result_ty.to_string())
    } else if is_float_type(ty) {
        let op_name = match op {
            goth_ast::op::BinOp::Add => "fadd",
            goth_ast::op::BinOp::Sub => "fsub",
            goth_ast::op::BinOp::Mul => "fmul",
            goth_ast::op::BinOp::Div => "fdiv",
            goth_ast::op::BinOp::Lt => "fcmp olt",
            goth_ast::op::BinOp::Gt => "fcmp ogt",
            goth_ast::op::BinOp::Leq => "fcmp ole",
            goth_ast::op::BinOp::Geq => "fcmp oge",
            goth_ast::op::BinOp::Eq => "fcmp oeq",
            goth_ast::op::BinOp::Neq => "fcmp one",
            _ => return Err(LlvmError::UnsupportedOp(format!("{:?}", op))),
        };
        let result_ty = match op {
            goth_ast::op::BinOp::Lt
            | goth_ast::op::BinOp::Gt
            | goth_ast::op::BinOp::Leq
            | goth_ast::op::BinOp::Geq
            | goth_ast::op::BinOp::Eq
            | goth_ast::op::BinOp::Neq => "i1",
            _ => &llvm_ty,
        };
        (op_name, result_ty.to_string())
    } else if is_bool_type(ty) {
        // Bool result type - either logical ops on bools, or comparison ops on integers/floats
        let is_float_operand = is_float_type(operand_ty);
        match op {
            // Logical operations on Bool operands
            goth_ast::op::BinOp::And => {
                let code = format!("  {} = and i1 {}, {}\n", ssa, left, right);
                return Ok((ssa, code));
            }
            goth_ast::op::BinOp::Or => {
                let code = format!("  {} = or i1 {}, {}\n", ssa, left, right);
                return Ok((ssa, code));
            }
            // Comparison operations - use fcmp for floats, icmp for integers
            goth_ast::op::BinOp::Lt => {
                let code = if is_float_operand {
                    format!("  {} = fcmp olt {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                } else {
                    format!("  {} = icmp slt {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                };
                return Ok((ssa, code));
            }
            goth_ast::op::BinOp::Gt => {
                let code = if is_float_operand {
                    format!("  {} = fcmp ogt {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                } else {
                    format!("  {} = icmp sgt {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                };
                return Ok((ssa, code));
            }
            goth_ast::op::BinOp::Leq => {
                let code = if is_float_operand {
                    format!("  {} = fcmp ole {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                } else {
                    format!("  {} = icmp sle {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                };
                return Ok((ssa, code));
            }
            goth_ast::op::BinOp::Geq => {
                let code = if is_float_operand {
                    format!("  {} = fcmp oge {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                } else {
                    format!("  {} = icmp sge {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                };
                return Ok((ssa, code));
            }
            goth_ast::op::BinOp::Eq => {
                let code = if is_float_operand {
                    format!("  {} = fcmp oeq {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                } else {
                    format!("  {} = icmp eq {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                };
                return Ok((ssa, code));
            }
            goth_ast::op::BinOp::Neq => {
                let code = if is_float_operand {
                    format!("  {} = fcmp one {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                } else {
                    format!("  {} = icmp ne {} {}, {}\n", ssa, operand_llvm_ty, left, right)
                };
                return Ok((ssa, code));
            }
            _ => return Err(LlvmError::UnsupportedOp(format!("Bool {:?}", op))),
        }
    } else if is_string_type(ty) {
        // String comparison - call strcmp and compare result
        match op {
            goth_ast::op::BinOp::Eq | goth_ast::op::BinOp::Neq => {
                // Call strcmp and compare with 0
                let cmp_ssa = ctx.fresh_ssa();
                let op_name = if *op == goth_ast::op::BinOp::Eq { "icmp eq" } else { "icmp ne" };
                let code = format!(
                    "  {} = call i32 @strcmp(i8* {}, i8* {})\n  {} = {} i32 {}, 0\n",
                    cmp_ssa, left, right, ssa, op_name, cmp_ssa
                );
                return Ok((ssa, code));
            }
            _ => return Err(LlvmError::UnsupportedOp(format!("String {:?}", op))),
        }
    } else {
        return Err(LlvmError::UnsupportedType(format!("{:?}", ty)));
    };

    let code = format!("  {} = {} {} {}, {}\n", ssa, op_name, llvm_ty, left, right);

    Ok((ssa, code))
}

/// Emit statement
fn emit_stmt(ctx: &mut LlvmContext, stmt: &Stmt, output: &mut String) -> Result<()> {
    let (ssa, code) = match &stmt.rhs {
        Rhs::Use(op) => {
            let op_val = emit_operand(ctx, op, &stmt.ty, output)?;
            // Just alias the value
            (op_val, String::new())
        }

        Rhs::Const(c) => {
            let (val, code) = emit_constant(ctx, c, &stmt.ty)?;
            (val, code)
        }

        Rhs::BinOp(op, left, right) => {
            // Handle tensor operations specially
            match op {
                goth_ast::op::BinOp::Map => {
                    // Map: array ↦ closure
                    // For now, call a runtime function with the closure
                    let arr_ty = Type::Prim(PrimType::I64);
                    let arr_val = emit_operand(ctx, left, &arr_ty, output)?;
                    let closure_val = emit_operand(ctx, right, &arr_ty, output)?;

                    // Get array length
                    let len_ssa = ctx.fresh_ssa();
                    output.push_str(&format!(
                        "  {} = call i64 @goth_len(i8* {})\n",
                        len_ssa, arr_val
                    ));

                    // Call map runtime function
                    let ssa = ctx.fresh_ssa();
                    let code = format!(
                        "  {} = call i8* @goth_map_i64(i8* {}, i8* {}, i64 {})\n",
                        ssa, arr_val, closure_val, len_ssa
                    );
                    (ssa, code)
                }
                goth_ast::op::BinOp::Filter => {
                    // Filter: array ▸ closure
                    let arr_ty = Type::Prim(PrimType::I64);
                    let arr_val = emit_operand(ctx, left, &arr_ty, output)?;
                    let closure_val = emit_operand(ctx, right, &arr_ty, output)?;

                    // Get array length
                    let len_ssa = ctx.fresh_ssa();
                    output.push_str(&format!(
                        "  {} = call i64 @goth_len(i8* {})\n",
                        len_ssa, arr_val
                    ));

                    // Call filter runtime function
                    let ssa = ctx.fresh_ssa();
                    let code = format!(
                        "  {} = call i8* @goth_filter_i64(i8* {}, i8* {}, i64 {})\n",
                        ssa, arr_val, closure_val, len_ssa
                    );
                    (ssa, code)
                }
                _ => {
                    // For comparison/logical ops, operand types differ from result type
                    // Look up actual operand types from context for each operand
                    let get_operand_type = |op: &Operand| -> Type {
                        match op {
                            Operand::Local(local) => {
                                ctx.local_types.get(local).cloned().unwrap_or_else(|| stmt.ty.clone())
                            }
                            Operand::Const(c) => match c {
                                Constant::Int(_) => Type::Prim(PrimType::I64),
                                Constant::Float(_) => Type::Prim(PrimType::F64),
                                Constant::Bool(_) => Type::Prim(PrimType::Bool),
                                Constant::String(_) => Type::Prim(PrimType::String),
                                Constant::Unit => Type::Tuple(vec![]),
                            },
                        }
                    };
                    let left_ty = get_operand_type(left);
                    let right_ty = get_operand_type(right);

                    let mut left_val = emit_operand(ctx, left, &left_ty, output)?;
                    let mut right_val = emit_operand(ctx, right, &right_ty, output)?;

                    // Handle type conversions for arithmetic operations
                    let is_arithmetic = matches!(op,
                        goth_ast::op::BinOp::Add | goth_ast::op::BinOp::Sub |
                        goth_ast::op::BinOp::Mul | goth_ast::op::BinOp::Div |
                        goth_ast::op::BinOp::Mod
                    );
                    let result_is_float = is_float_type(&stmt.ty);
                    let result_is_int = is_int_type(&stmt.ty);

                    if is_arithmetic && result_is_float {
                        // Convert integer operands to float if result is float
                        if is_int_type(&left_ty) {
                            let conv_ssa = ctx.fresh_ssa();
                            output.push_str(&format!(
                                "  {} = sitofp i64 {} to double\n",
                                conv_ssa, left_val
                            ));
                            left_val = conv_ssa;
                        }
                        if is_int_type(&right_ty) {
                            let conv_ssa = ctx.fresh_ssa();
                            output.push_str(&format!(
                                "  {} = sitofp i64 {} to double\n",
                                conv_ssa, right_val
                            ));
                            right_val = conv_ssa;
                        }
                    } else if is_arithmetic && result_is_int {
                        // Convert float operands to int if result is int
                        if is_float_type(&left_ty) {
                            let conv_ssa = ctx.fresh_ssa();
                            output.push_str(&format!(
                                "  {} = fptosi double {} to i64\n",
                                conv_ssa, left_val
                            ));
                            left_val = conv_ssa;
                        }
                        if is_float_type(&right_ty) {
                            let conv_ssa = ctx.fresh_ssa();
                            output.push_str(&format!(
                                "  {} = fptosi double {} to i64\n",
                                conv_ssa, right_val
                            ));
                            right_val = conv_ssa;
                        }
                    }

                    // Use the operand type for comparison operations, result type for arithmetic
                    let operand_ty_for_binop = if is_arithmetic {
                        stmt.ty.clone()
                    } else {
                        left_ty.clone()
                    };

                    let (ssa, code) = emit_binop(ctx, op, &left_val, &right_val, &stmt.ty, &operand_ty_for_binop)?;

                    // If this is a comparison and dest type is i64, extend the result
                    let is_comparison = matches!(op,
                        goth_ast::op::BinOp::Lt | goth_ast::op::BinOp::Gt |
                        goth_ast::op::BinOp::Leq | goth_ast::op::BinOp::Geq |
                        goth_ast::op::BinOp::Eq | goth_ast::op::BinOp::Neq
                    );
                    let dest_is_i64 = matches!(&stmt.ty, Type::Prim(PrimType::I64));

                    if is_comparison && dest_is_i64 {
                        output.push_str(&code);
                        let extended_ssa = ctx.fresh_ssa();
                        let extend_code = format!("  {} = zext i1 {} to i64\n", extended_ssa, ssa);
                        (extended_ssa, extend_code)
                    } else {
                        (ssa, code)
                    }
                }
            }
        }

        Rhs::UnaryOp(op, operand) => {
            let op_val = emit_operand(ctx, operand, &stmt.ty, output)?;
            let llvm_ty = emit_type(&stmt.ty)?;

            // Handle Sum/Prod specially - they need intermediate values
            match op {
                goth_ast::op::UnaryOp::Sum => {
                    // Sum reduction: get length first, then call appropriate runtime function
                    let len_ssa = ctx.fresh_ssa();
                    output.push_str(&format!(
                        "  {} = call i64 @goth_len(i8* {})\n",
                        len_ssa, op_val
                    ));
                    let ssa = ctx.fresh_ssa();
                    // Check if result type is float or int
                    let is_float = match &stmt.ty {
                        Type::Prim(PrimType::F64) | Type::Prim(PrimType::F32) => true,
                        _ => false,
                    };
                    let (func_name, ret_ty) = if is_float {
                        ("goth_sum_f64", "double")
                    } else {
                        ("goth_sum_i64", "i64")
                    };
                    let code = format!(
                        "  {} = call {} @{}(i8* {}, i64 {})\n",
                        ssa, ret_ty, func_name, op_val, len_ssa
                    );
                    (ssa, code)
                }
                goth_ast::op::UnaryOp::Prod => {
                    // Product reduction: get length first, then call appropriate runtime function
                    let len_ssa = ctx.fresh_ssa();
                    output.push_str(&format!(
                        "  {} = call i64 @goth_len(i8* {})\n",
                        len_ssa, op_val
                    ));
                    let ssa = ctx.fresh_ssa();
                    // Check if result type is float or int
                    let is_float = match &stmt.ty {
                        Type::Prim(PrimType::F64) | Type::Prim(PrimType::F32) => true,
                        _ => false,
                    };
                    let (func_name, ret_ty) = if is_float {
                        ("goth_prod_f64", "double")
                    } else {
                        ("goth_prod_i64", "i64")
                    };
                    let code = format!(
                        "  {} = call {} @{}(i8* {}, i64 {})\n",
                        ssa, ret_ty, func_name, op_val, len_ssa
                    );
                    (ssa, code)
                }
                goth_ast::op::UnaryOp::Sign => {
                    // sign(x) = copysign(1.0, x) for non-zero, 0 for zero
                    // Emits: fcmp, copysign, select - must allocate SSAs in order
                    let cmp_ssa = ctx.fresh_ssa();
                    let copysign_ssa = ctx.fresh_ssa();
                    let select_ssa = ctx.fresh_ssa();
                    let code = format!(
                        "  {} = fcmp one double {}, 0.0\n  {} = call double @copysign(double 1.0, double {})\n  {} = select i1 {}, double {}, double 0.0\n",
                        cmp_ssa, op_val, copysign_ssa, op_val, select_ssa, cmp_ssa, copysign_ssa
                    );
                    (select_ssa, code)
                }
                _ => {
                    let ssa = ctx.fresh_ssa();
                    let code = match op {
                        goth_ast::op::UnaryOp::Neg => {
                            if is_int_type(&stmt.ty) {
                                format!("  {} = sub {} 0, {}\n", ssa, llvm_ty, op_val)
                            } else {
                                format!("  {} = fneg {} {}\n", ssa, llvm_ty, op_val)
                            }
                        }
                        goth_ast::op::UnaryOp::Not => {
                            format!("  {} = xor i1 {}, 1\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Sqrt => {
                            format!("  {} = call double @sqrt(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Floor => {
                            format!("  {} = call double @floor(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Ceil => {
                            format!("  {} = call double @ceil(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Gamma => {
                            format!("  {} = call double @tgamma(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Ln => {
                            format!("  {} = call double @log(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Exp => {
                            format!("  {} = call double @exp(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Sin => {
                            format!("  {} = call double @sin(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Cos => {
                            format!("  {} = call double @cos(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Abs => {
                            format!("  {} = call double @fabs(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Tan => {
                            format!("  {} = call double @tan(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Asin => {
                            format!("  {} = call double @asin(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Acos => {
                            format!("  {} = call double @acos(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Atan => {
                            format!("  {} = call double @atan(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Sinh => {
                            format!("  {} = call double @sinh(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Cosh => {
                            format!("  {} = call double @cosh(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Tanh => {
                            format!("  {} = call double @tanh(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Log10 => {
                            format!("  {} = call double @log10(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Log2 => {
                            format!("  {} = call double @log2(double {})\n", ssa, op_val)
                        }
                        goth_ast::op::UnaryOp::Round => {
                            format!("  {} = call double @round(double {})\n", ssa, op_val)
                        }
                        _ => return Err(LlvmError::UnsupportedOp(format!("{:?}", op))),
                    };
                    (ssa, code)
                }
            }
        }

        Rhs::Call { func, args, arg_tys } => {
            let ret_ty = emit_type(&stmt.ty)?;
            let is_void_ret = ret_ty == "void";

            let mut arg_strs = Vec::new();
            for (arg, arg_ty) in args.iter().zip(arg_tys.iter()) {
                // Skip unit-typed arguments - they don't exist in LLVM
                if is_unit_type(arg_ty) {
                    continue;
                }
                let llvm_ty = emit_type(arg_ty)?;
                let arg_val = emit_operand(ctx, arg, arg_ty, output)?;
                arg_strs.push(format!("{} {}", llvm_ty, arg_val));
            }

            if is_void_ret {
                let code = format!(
                    "  call void @{}({})\n",
                    func,
                    arg_strs.join(", ")
                );
                // Return a placeholder for void - won't be used
                ("undef".to_string(), code)
            } else {
                let ssa = ctx.fresh_ssa();
                let code = format!(
                    "  {} = call {} @{}({})\n",
                    ssa,
                    ret_ty,
                    func,
                    arg_strs.join(", ")
                );
                (ssa, code)
            }
        }

        Rhs::Iota(n) => {
            let ssa = ctx.fresh_ssa();
            let n_ty = Type::Prim(PrimType::I64);
            let n_val = emit_operand(ctx, n, &n_ty, output)?;

            let code = format!("  {} = call i8* @goth_iota(i64 {})\n", ssa, n_val);

            (ssa, code)
        }

        Rhs::Range(start, end) => {
            let ssa = ctx.fresh_ssa();
            let int_ty = Type::Prim(PrimType::I64);
            let start_val = emit_operand(ctx, start, &int_ty, output)?;
            let end_val = emit_operand(ctx, end, &int_ty, output)?;

            let code = format!(
                "  {} = call i8* @goth_range(i64 {}, i64 {})\n",
                ssa, start_val, end_val
            );

            (ssa, code)
        }

        Rhs::TensorReduce { tensor, op } => {
            let tensor_ty = Type::Prim(PrimType::I64); // Placeholder
            let tensor_val = emit_operand(ctx, tensor, &tensor_ty, output)?;

            // Get length first
            let len_ssa = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = call i64 @goth_len(i8* {})\n",
                len_ssa, tensor_val
            ));

            // Check if result type is float or int
            let is_float = match &stmt.ty {
                Type::Prim(PrimType::F64) | Type::Prim(PrimType::F32) => true,
                _ => false,
            };

            let ssa = ctx.fresh_ssa();
            let (func_name, ret_ty) = match (op, is_float) {
                (ReduceOp::Sum, true) => ("goth_sum_f64", "double"),
                (ReduceOp::Sum, false) => ("goth_sum_i64", "i64"),
                (ReduceOp::Prod, true) => ("goth_prod_f64", "double"),
                (ReduceOp::Prod, false) => ("goth_prod_i64", "i64"),
                (ReduceOp::Min, _) => ("goth_min_i64", "i64"),
                (ReduceOp::Max, _) => ("goth_max_i64", "i64"),
            };

            let code = format!(
                "  {} = call {} @{}(i8* {}, i64 {})\n",
                ssa, ret_ty, func_name, tensor_val, len_ssa
            );

            (ssa, code)
        }

        Rhs::Index(arr, idx) => {
            // Look up array type from context
            let arr_ty = match arr {
                Operand::Local(local) => ctx.local_types.get(local).cloned(),
                _ => None,
            }.unwrap_or(Type::Prim(PrimType::I64));
            let arr_val = emit_operand(ctx, arr, &arr_ty, output)?;
            let idx_val = emit_operand(ctx, idx, &Type::Prim(PrimType::I64), output)?;

            // Use the statement's result type to determine the right index function
            let elem_ty = emit_type(&stmt.ty)?;

            // For string indexing, return string
            if elem_ty == "i8*" || is_string_type(&stmt.ty) {
                let ssa = ctx.fresh_ssa();
                let code = format!(
                    "  {} = call i8* @goth_index_str(i8* {}, i64 {})\n",
                    ssa, arr_val, idx_val
                );
                (ssa, code)
            } else if elem_ty == "i32" {
                // Char type - index i64 array and truncate
                let tmp_ssa = ctx.fresh_ssa();
                output.push_str(&format!(
                    "  {} = call i64 @goth_index_i64(i8* {}, i64 {})\n",
                    tmp_ssa, arr_val, idx_val
                ));
                let ssa = ctx.fresh_ssa();
                let code = format!("  {} = trunc i64 {} to i32\n", ssa, tmp_ssa);
                (ssa, code)
            } else {
                // Default i64 indexing
                let ssa = ctx.fresh_ssa();
                let code = format!(
                    "  {} = call i64 @goth_index_i64(i8* {}, i64 {})\n",
                    ssa, arr_val, idx_val
                );
                (ssa, code)
            }
        }

        Rhs::Prim { name, args } => {
            let ret_ty = emit_type(&stmt.ty)?;
            let is_void_ret = ret_ty == "void";

            // Handle specific primitives
            match name.as_str() {
                "print" => {
                    if let Some(arg) = args.first() {
                        // Detect argument type from the operand or local type
                        let is_string = match arg {
                            Operand::Const(Constant::String(_)) => true,
                            Operand::Local(local) => {
                                ctx.local_types.get(local)
                                    .map(|ty| is_string_type(ty))
                                    .unwrap_or(false)
                            }
                            _ => false,
                        };
                        let is_float = match arg {
                            Operand::Const(Constant::Float(_)) => true,
                            Operand::Local(local) => {
                                ctx.local_types.get(local)
                                    .map(|ty| is_float_type(ty))
                                    .unwrap_or(false)
                            }
                            _ => false,
                        };
                        let is_bool = match arg {
                            Operand::Const(Constant::Bool(_)) => true,
                            Operand::Local(local) => {
                                ctx.local_types.get(local)
                                    .map(|ty| is_bool_type(ty))
                                    .unwrap_or(false)
                            }
                            _ => false,
                        };
                        if is_string {
                            let arg_ty = Type::Prim(PrimType::String);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_string(i8* {})\n", arg_val);
                            output.push_str(&code);
                        } else if is_float {
                            let arg_ty = Type::Prim(PrimType::F64);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_f64(double {})\n", arg_val);
                            output.push_str(&code);
                        } else if is_bool {
                            let arg_ty = Type::Prim(PrimType::Bool);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_bool(i1 {})\n", arg_val);
                            output.push_str(&code);
                        } else {
                            let arg_ty = Type::Prim(PrimType::I64);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_i64(i64 {})\n", arg_val);
                            output.push_str(&code);
                        }
                        output.push_str("  call void @goth_print_newline()\n");
                    }
                    // Return unit placeholder
                    ("undef".to_string(), String::new())
                }
                "write" => {
                    // write: print without newline
                    if let Some(arg) = args.first() {
                        let is_string = match arg {
                            Operand::Const(Constant::String(_)) => true,
                            Operand::Local(local) => {
                                ctx.local_types.get(local)
                                    .map(|ty| is_string_type(ty))
                                    .unwrap_or(false)
                            }
                            _ => false,
                        };
                        let is_float = match arg {
                            Operand::Const(Constant::Float(_)) => true,
                            Operand::Local(local) => {
                                ctx.local_types.get(local)
                                    .map(|ty| is_float_type(ty))
                                    .unwrap_or(false)
                            }
                            _ => false,
                        };
                        if is_string {
                            let arg_ty = Type::Prim(PrimType::String);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_string(i8* {})\n", arg_val);
                            output.push_str(&code);
                        } else if is_float {
                            let arg_ty = Type::Prim(PrimType::F64);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_f64(double {})\n", arg_val);
                            output.push_str(&code);
                        } else {
                            let arg_ty = Type::Prim(PrimType::I64);
                            let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                            let code = format!("  call void @goth_print_i64(i64 {})\n", arg_val);
                            output.push_str(&code);
                        }
                    }
                    ("undef".to_string(), String::new())
                }
                "flush" => {
                    output.push_str("  call void @fflush(i8* null)\n");
                    ("undef".to_string(), String::new())
                }
                // TUI primitives - these take unit and return unit
                "cursorHide" | "cursorShow" | "screenClear" | "rawModeEnter" | "rawModeExit" => {
                    // These are zero-argument void functions
                    let code = format!("  call void @goth_{}()\n", name);
                    output.push_str(&code);
                    ("undef".to_string(), String::new())
                }
                "cursorMove" => {
                    // cursorMove takes (row, col)
                    if args.len() >= 2 {
                        let row_ty = Type::Prim(PrimType::I64);
                        let col_ty = Type::Prim(PrimType::I64);
                        let row_val = emit_operand(ctx, &args[0], &row_ty, output)?;
                        let col_val = emit_operand(ctx, &args[1], &col_ty, output)?;
                        let code = format!("  call void @goth_cursorMove(i64 {}, i64 {})\n", row_val, col_val);
                        output.push_str(&code);
                    }
                    ("undef".to_string(), String::new())
                }
                "sleep" => {
                    if let Some(arg) = args.first() {
                        let arg_ty = Type::Prim(PrimType::I64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        let code = format!("  call void @goth_sleep(i64 {})\n", arg_val);
                        output.push_str(&code);
                    }
                    ("undef".to_string(), String::new())
                }
                "toString" => {
                    // Convert value to string
                    // Note: MIR may incorrectly type this as Char, but it returns String
                    if let Some(arg) = args.first() {
                        // Determine argument type
                        let arg_ty = match arg {
                            Operand::Local(local) => {
                                ctx.local_types.get(local).cloned().unwrap_or(Type::Prim(PrimType::I64))
                            }
                            Operand::Const(c) => match c {
                                Constant::Int(_) => Type::Prim(PrimType::I64),
                                Constant::Float(_) => Type::Prim(PrimType::F64),
                                Constant::Bool(_) => Type::Prim(PrimType::Bool),
                                _ => Type::Prim(PrimType::I64),
                            },
                        };
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        let llvm_arg_ty = emit_type(&arg_ty)?;

                        // For Char (i32), extend to i64 first
                        let (final_arg, final_ty) = if llvm_arg_ty == "i32" {
                            let ext_ssa = ctx.fresh_ssa();
                            output.push_str(&format!("  {} = sext i32 {} to i64\n", ext_ssa, arg_val));
                            (ext_ssa, "i64".to_string())
                        } else {
                            (arg_val, llvm_arg_ty)
                        };

                        // Create the SSA for the result AFTER any extension
                        let ssa = ctx.fresh_ssa();
                        output.push_str(&format!("  {} = call i8* @goth_toString_{}({} {})\n",
                            ssa,
                            if is_float_type(&arg_ty) { "f64" } else { "i64" },
                            final_ty,
                            final_arg
                        ));
                        // Override: toString ALWAYS returns String, regardless of MIR type
                        ctx.register_local(stmt.dest, ssa.clone(), Type::Prim(PrimType::String));
                        return Ok(());
                    } else {
                        ("undef".to_string(), String::new())
                    }
                }
                "len" => {
                    if let Some(arg) = args.first() {
                        let arr_ty = Type::Prim(PrimType::I64);
                        let arr_val = emit_operand(ctx, arg, &arr_ty, output)?;
                        let ssa = ctx.fresh_ssa();
                        let code = format!("  {} = call i64 @goth_len(i8* {})\n", ssa, arr_val);
                        (ssa, code)
                    } else {
                        ("0".to_string(), String::new())
                    }
                }
                "reverse" => {
                    if let Some(arg) = args.first() {
                        let arr_ty = Type::Prim(PrimType::I64);
                        let arr_val = emit_operand(ctx, arg, &arr_ty, output)?;
                        let ssa = ctx.fresh_ssa();
                        let len_ssa = ctx.fresh_ssa();
                        output.push_str(&format!(
                            "  {} = call i64 @goth_len(i8* {})\n",
                            len_ssa, arr_val
                        ));
                        let code = format!(
                            "  {} = call i8* @goth_reverse(i8* {}, i64 {})\n",
                            ssa, arr_val, len_ssa
                        );
                        (ssa, code)
                    } else {
                        ("undef".to_string(), String::new())
                    }
                }
                "toFloat" => {
                    // Convert integer to float
                    if let Some(arg) = args.first() {
                        let ssa = ctx.fresh_ssa();
                        let arg_ty = Type::Prim(PrimType::I64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        let code = format!("  {} = sitofp i64 {} to double\n", ssa, arg_val);
                        (ssa, code)
                    } else {
                        ("0.0".to_string(), String::new())
                    }
                }
                "toInt" => {
                    // Convert to integer (if needed)
                    if let Some(arg) = args.first() {
                        // Get actual argument type
                        let actual_arg_ty = match arg {
                            Operand::Local(local) => {
                                ctx.local_types.get(local).cloned().unwrap_or(Type::Prim(PrimType::F64))
                            }
                            Operand::Const(c) => match c {
                                Constant::Int(_) => Type::Prim(PrimType::I64),
                                Constant::Float(_) => Type::Prim(PrimType::F64),
                                _ => Type::Prim(PrimType::F64),
                            },
                        };

                        if is_int_type(&actual_arg_ty) {
                            // Already an integer, just pass through
                            let arg_val = emit_operand(ctx, arg, &actual_arg_ty, output)?;
                            (arg_val, String::new())
                        } else {
                            // Convert float to integer
                            let ssa = ctx.fresh_ssa();
                            let arg_val = emit_operand(ctx, arg, &Type::Prim(PrimType::F64), output)?;
                            let code = format!("  {} = fptosi double {} to i64\n", ssa, arg_val);
                            (ssa, code)
                        }
                    } else {
                        ("0".to_string(), String::new())
                    }
                }
                "floor" => {
                    if let Some(arg) = args.first() {
                        let ssa = ctx.fresh_ssa();
                        let arg_ty = Type::Prim(PrimType::F64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        let code = format!("  {} = call double @floor(double {})\n", ssa, arg_val);
                        (ssa, code)
                    } else {
                        ("0.0".to_string(), String::new())
                    }
                }
                "ceil" => {
                    if let Some(arg) = args.first() {
                        let ssa = ctx.fresh_ssa();
                        let arg_ty = Type::Prim(PrimType::F64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        let code = format!("  {} = call double @ceil(double {})\n", ssa, arg_val);
                        (ssa, code)
                    } else {
                        ("0.0".to_string(), String::new())
                    }
                }
                "sqrt" => {
                    if let Some(arg) = args.first() {
                        let ssa = ctx.fresh_ssa();
                        let arg_ty = Type::Prim(PrimType::F64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        let code = format!("  {} = call double @sqrt(double {})\n", ssa, arg_val);
                        (ssa, code)
                    } else {
                        ("0.0".to_string(), String::new())
                    }
                }
                _ => {
                    // Generic primitive - emit as function call
                    // Skip unit-typed arguments
                    let mut arg_strs = Vec::new();
                    for arg in args {
                        // Skip unit constants
                        if let Operand::Const(Constant::Unit) = arg {
                            continue;
                        }
                        // Look up the actual type of the argument
                        let arg_ty = match arg {
                            Operand::Local(local) => {
                                ctx.local_types.get(local).cloned().unwrap_or(Type::Prim(PrimType::I64))
                            }
                            Operand::Const(c) => match c {
                                Constant::Int(_) => Type::Prim(PrimType::I64),
                                Constant::Float(_) => Type::Prim(PrimType::F64),
                                Constant::Bool(_) => Type::Prim(PrimType::Bool),
                                Constant::String(_) => Type::Prim(PrimType::String),
                                Constant::Unit => Type::Tuple(vec![]),
                            },
                        };
                        let llvm_ty = emit_type(&arg_ty)?;
                        if llvm_ty == "void" {
                            continue;
                        }
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        arg_strs.push(format!("{} {}", llvm_ty, arg_val));
                    }
                    if is_void_ret {
                        let code = format!(
                            "  call void @goth_{}({})\n",
                            name,
                            arg_strs.join(", ")
                        );
                        ("undef".to_string(), code)
                    } else {
                        let ssa = ctx.fresh_ssa();
                        let code = format!(
                            "  {} = call {} @goth_{}({})\n",
                            ssa,
                            ret_ty,
                            name,
                            arg_strs.join(", ")
                        );
                        (ssa, code)
                    }
                }
            }
        }

        Rhs::Tuple(ops) => {
            // Empty tuple is unit type
            if ops.is_empty() {
                ("undef".to_string(), String::new())
            } else if let Some(first) = ops.first() {
                let val = emit_operand(ctx, first, &stmt.ty, output)?;
                (val, String::new())
            } else {
                ("undef".to_string(), String::new())
            }
        }

        Rhs::TupleField(tup, idx) => {
            // For now, simplified - just pass through
            let tup_val = emit_operand(ctx, tup, &stmt.ty, output)?;
            (tup_val, String::new())
        }

        Rhs::Array(elems) => {
            // Allocate array and fill
            let ssa = ctx.fresh_ssa();
            let len = elems.len();

            // Allocate
            let alloc_ssa = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = call i8* @malloc(i64 {})\n",
                alloc_ssa,
                (len + 1) * 8
            )); // +1 for length

            // Store length
            let len_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = bitcast i8* {} to i64*\n",
                len_ptr, alloc_ssa
            ));
            output.push_str(&format!(
                "  store i64 {}, i64* {}\n",
                len, len_ptr
            ));

            // Store elements
            for (i, elem) in elems.iter().enumerate() {
                let elem_ty = Type::Prim(PrimType::I64);
                let elem_val = emit_operand(ctx, elem, &elem_ty, output)?;
                let elem_ptr = ctx.fresh_ssa();
                let offset_ptr = ctx.fresh_ssa();
                output.push_str(&format!(
                    "  {} = getelementptr i64, i64* {}, i64 {}\n",
                    offset_ptr,
                    len_ptr,
                    i + 1
                ));
                output.push_str(&format!(
                    "  store i64 {}, i64* {}\n",
                    elem_val, offset_ptr
                ));
            }

            (alloc_ssa, String::new())
        }

        Rhs::MakeClosure { func, captures } => {
            // For now, just return function pointer
            let ssa = format!("@{}", func);
            (ssa, String::new())
        }

        Rhs::ClosureCall { closure, args } => {
            let ret_ty = emit_type(&stmt.ty)?;
            let is_void_ret = ret_ty == "void";
            let closure_val = emit_operand(ctx, closure, &stmt.ty, output)?;

            let mut arg_strs = Vec::new();
            for arg in args {
                // Check if operand is a unit constant and skip it
                if let Operand::Const(Constant::Unit) = arg {
                    continue;
                }
                // Determine actual argument type from operand
                let arg_ty = match arg {
                    Operand::Local(local) => {
                        ctx.local_types.get(local).cloned().unwrap_or(Type::Prim(PrimType::I64))
                    }
                    Operand::Const(c) => match c {
                        Constant::Int(_) => Type::Prim(PrimType::I64),
                        Constant::Float(_) => Type::Prim(PrimType::F64),
                        Constant::Bool(_) => Type::Prim(PrimType::Bool),
                        Constant::String(_) => Type::Prim(PrimType::String),
                        Constant::Unit => Type::Tuple(vec![]),
                    },
                };
                // Skip unit-typed arguments
                if is_unit_type(&arg_ty) {
                    continue;
                }
                let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                let llvm_ty = emit_type(&arg_ty)?;
                arg_strs.push(format!("{} {}", llvm_ty, arg_val));
            }

            // Indirect call
            if is_void_ret {
                let code = format!(
                    "  call void {}({})\n",
                    closure_val,
                    arg_strs.join(", ")
                );
                ("undef".to_string(), code)
            } else {
                let ssa = ctx.fresh_ssa();
                let code = format!(
                    "  {} = call {} {}({})\n",
                    ssa,
                    ret_ty,
                    closure_val,
                    arg_strs.join(", ")
                );
                (ssa, code)
            }
        }

        Rhs::MakeVariant { tag, constructor: _, payload } => {
            // Variant representation: { i32 tag, i64 payload }
            // Allocate 16 bytes (8 for tag padded, 8 for payload)
            let alloc_ssa = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = call i8* @malloc(i64 16)\n",
                alloc_ssa
            ));

            // Cast to i64* to store tag
            let tag_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = bitcast i8* {} to i64*\n",
                tag_ptr, alloc_ssa
            ));

            // Store tag
            output.push_str(&format!(
                "  store i64 {}, i64* {}\n",
                tag, tag_ptr
            ));

            // Store payload if present
            if let Some(p) = payload {
                let payload_ty = Type::Prim(PrimType::I64);
                let payload_val = emit_operand(ctx, p, &payload_ty, output)?;

                let payload_ptr = ctx.fresh_ssa();
                output.push_str(&format!(
                    "  {} = getelementptr i64, i64* {}, i64 1\n",
                    payload_ptr, tag_ptr
                ));
                output.push_str(&format!(
                    "  store i64 {}, i64* {}\n",
                    payload_val, payload_ptr
                ));
            }

            (alloc_ssa, String::new())
        }

        Rhs::GetTag(variant) => {
            let variant_ty = Type::Prim(PrimType::I64); // Pointer type
            let variant_val = emit_operand(ctx, variant, &variant_ty, output)?;

            // Cast to i64* and load tag
            let tag_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = bitcast i8* {} to i64*\n",
                tag_ptr, variant_val
            ));

            let ssa = ctx.fresh_ssa();
            let code = format!(
                "  {} = load i64, i64* {}\n",
                ssa, tag_ptr
            );

            (ssa, code)
        }

        Rhs::GetPayload(variant) => {
            let variant_ty = Type::Prim(PrimType::I64); // Pointer type
            let variant_val = emit_operand(ctx, variant, &variant_ty, output)?;

            // Cast to i64* and get payload (offset 1)
            let base_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = bitcast i8* {} to i64*\n",
                base_ptr, variant_val
            ));

            let payload_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = getelementptr i64, i64* {}, i64 1\n",
                payload_ptr, base_ptr
            ));

            let ssa = ctx.fresh_ssa();
            let code = format!(
                "  {} = load i64, i64* {}\n",
                ssa, payload_ptr
            );

            (ssa, code)
        }

        Rhs::ArrayFill { size, value } => {
            // Create array filled with n copies of a value: [n]⊢v
            let size_ty = Type::Prim(PrimType::I64);
            let size_val = emit_operand(ctx, size, &size_ty, output)?;
            let value_ty = Type::Prim(PrimType::I64);
            let value_val = emit_operand(ctx, value, &value_ty, output)?;

            // Calculate allocation size: (size + 1) * 8 bytes
            let size_plus_one = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = add i64 {}, 1\n",
                size_plus_one, size_val
            ));
            let alloc_size = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = mul i64 {}, 8\n",
                alloc_size, size_plus_one
            ));

            // Allocate memory
            let alloc_ssa = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = call i8* @malloc(i64 {})\n",
                alloc_ssa, alloc_size
            ));

            // Store length at position 0
            let len_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = bitcast i8* {} to i64*\n",
                len_ptr, alloc_ssa
            ));
            output.push_str(&format!(
                "  store i64 {}, i64* {}\n",
                size_val, len_ptr
            ));

            // Fill loop: for i = 0 to size-1, store value at position i+1
            // We'll unroll for small constant sizes or use a simple loop
            // For now, emit a loop
            let loop_header = format!("arrayfill.header.{}", ctx.next_ssa);
            let loop_body = format!("arrayfill.body.{}", ctx.next_ssa);
            let loop_exit = format!("arrayfill.exit.{}", ctx.next_ssa);
            ctx.next_ssa += 1;

            // Initialize loop counter
            let counter_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = alloca i64\n",
                counter_ptr
            ));
            output.push_str(&format!(
                "  store i64 0, i64* {}\n",
                counter_ptr
            ));
            output.push_str(&format!(
                "  br label %{}\n",
                loop_header
            ));

            // Loop header: check if counter < size
            output.push_str(&format!("{}:\n", loop_header));
            let counter_val = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = load i64, i64* {}\n",
                counter_val, counter_ptr
            ));
            let cmp = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = icmp slt i64 {}, {}\n",
                cmp, counter_val, size_val
            ));
            output.push_str(&format!(
                "  br i1 {}, label %{}, label %{}\n",
                cmp, loop_body, loop_exit
            ));

            // Loop body: store value, increment counter
            output.push_str(&format!("{}:\n", loop_body));
            let offset = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = add i64 {}, 1\n",
                offset, counter_val
            ));
            let elem_ptr = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = getelementptr i64, i64* {}, i64 {}\n",
                elem_ptr, len_ptr, offset
            ));
            output.push_str(&format!(
                "  store i64 {}, i64* {}\n",
                value_val, elem_ptr
            ));
            let next_counter = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = add i64 {}, 1\n",
                next_counter, counter_val
            ));
            output.push_str(&format!(
                "  store i64 {}, i64* {}\n",
                next_counter, counter_ptr
            ));
            output.push_str(&format!(
                "  br label %{}\n",
                loop_header
            ));

            // Loop exit
            output.push_str(&format!("{}:\n", loop_exit));

            (alloc_ssa, String::new())
        }

        _ => {
            return Err(LlvmError::UnsupportedOp(format!(
                "Statement: {:?}",
                stmt.rhs
            )));
        }
    };

    output.push_str(&code);

    let llvm_ty = emit_type(&stmt.ty)?;

    // For unit/void types, register with "undef" so later references don't fail
    if llvm_ty == "void" {
        ctx.register_local(stmt.dest, "undef".to_string(), stmt.ty.clone());
        return Ok(());
    }

    // If this is a stack-allocated local, emit a store instead of registering SSA
    if let Some(ptr) = ctx.get_stack_ptr(&stmt.dest).cloned() {
        output.push_str(&format!("  store {} {}, {}* {}\n", llvm_ty, ssa, llvm_ty, ptr));
    } else {
        ctx.register_local(stmt.dest, ssa, stmt.ty.clone());
    }

    Ok(())
}

/// Emit block with label
fn emit_block(
    ctx: &mut LlvmContext,
    block: &Block,
    label: Option<&str>,
    output: &mut String,
) -> Result<()> {
    if let Some(lbl) = label {
        output.push_str(&format!("{}:\n", lbl));
    }

    // Emit statements
    for stmt in &block.stmts {
        emit_stmt(ctx, stmt, output)?;
    }

    // Emit terminator
    match &block.term {
        Terminator::Return(op) => {
            // Use the function's return type from context
            let ret_ty = ctx.ret_ty.clone();
            let llvm_ty = emit_type(&ret_ty)?;
            if llvm_ty == "void" {
                // Void functions don't return a value
                output.push_str("  ret void\n");
            } else {
                let ret_val = emit_operand(ctx, op, &ret_ty, output)?;
                output.push_str(&format!("  ret {} {}\n", llvm_ty, ret_val));
            }
        }

        Terminator::Goto(block_id) => {
            output.push_str(&format!("  br label %bb{}\n", block_id.0));
        }

        Terminator::If {
            cond,
            then_block,
            else_block,
        } => {
            let cond_ty = Type::Prim(PrimType::Bool);
            let cond_val = emit_operand(ctx, cond, &cond_ty, output)?;
            // Condition is already i1 (Bool type from comparison/logical ops)
            output.push_str(&format!(
                "  br i1 {}, label %bb{}, label %bb{}\n",
                cond_val, then_block.0, else_block.0
            ));
        }

        Terminator::Switch {
            scrutinee,
            cases,
            default,
        } => {
            let scrut_ty = Type::Prim(PrimType::I64);
            let scrut_val = emit_operand(ctx, scrutinee, &scrut_ty, output)?;

            output.push_str(&format!(
                "  switch i64 {}, label %bb{} [\n",
                scrut_val, default.0
            ));
            for (val, block_id) in cases {
                let val_str = match val {
                    Constant::Int(n) => n.to_string(),
                    _ => "0".to_string(),
                };
                output.push_str(&format!("    i64 {}, label %bb{}\n", val_str, block_id.0));
            }
            output.push_str("  ]\n");
        }

        Terminator::Unreachable => {
            output.push_str("  unreachable\n");
        }
    }

    Ok(())
}

/// Emit function - returns (function_code, string_literals)
fn emit_function(func: &Function, is_main: bool) -> Result<(String, Vec<(String, String)>)> {
    let mut ctx = LlvmContext::new(func.ret_ty.clone());
    let mut output = String::new();

    // Function signature
    let ret_type = emit_type(&func.ret_ty)?;

    let param_strs: Vec<String> = func
        .params
        .iter()
        .enumerate()
        .filter_map(|(i, ty)| {
            let llvm_ty = emit_type(ty).unwrap_or_else(|_| "i64".to_string());
            // Skip void parameters (unit type) in the signature, but still register them
            if llvm_ty == "void" {
                // Register with "undef" so later references don't fail
                ctx.register_local(LocalId::new(i as u32), "undef".to_string(), ty.clone());
                return None;
            }
            let param_ssa = format!("%arg{}", i);
            // Register parameter
            ctx.register_local(LocalId::new(i as u32), param_ssa.clone(), ty.clone());
            Some(format!("{} {}", llvm_ty, param_ssa))
        })
        .collect();

    // For main, we need special handling
    let func_name = if is_main && func.name == "main" {
        "goth_main"
    } else {
        &func.name
    };

    output.push_str(&format!(
        "define {} @{}({}) {{\n",
        ret_type,
        func_name,
        param_strs.join(", ")
    ));

    // Find locals that need stack allocation (assigned in multiple blocks)
    let multi_block_locals = find_multi_block_locals(func);

    // Emit entry block label first
    output.push_str("entry:\n");

    // Emit alloca instructions for multi-block locals at entry
    for (local, ty) in &multi_block_locals {
        let llvm_ty = emit_type(ty)?;
        if llvm_ty == "void" {
            // For void types, pre-register with "undef" so uses before definition work
            ctx.register_local(*local, "undef".to_string(), ty.clone());
            continue;
        }
        let ptr_name = ctx.fresh_ssa();
        output.push_str(&format!("  {} = alloca {}\n", ptr_name, llvm_ty));
        ctx.register_stack_local(*local, ptr_name);
        // Also register the type so we can look it up later
        ctx.local_types.insert(*local, ty.clone());
    }

    // Emit entry block body (without re-emitting the label)
    if func.blocks.is_empty() {
        emit_block(&mut ctx, &func.body, None, &mut output)?;
    } else {
        emit_block(&mut ctx, &func.body, None, &mut output)?;

        // Emit additional blocks
        for (block_id, block) in &func.blocks {
            let label = format!("bb{}", block_id.0);
            emit_block(&mut ctx, block, Some(&label), &mut output)?;
        }
    }

    output.push_str("}\n");

    Ok((output, ctx.string_literals))
}

/// Emit the C main function that calls goth_main
fn emit_c_main(main_func: &Function) -> Result<String> {
    let mut output = String::new();

    let ret_ty = emit_type(&main_func.ret_ty)?;
    let is_void_ret = ret_ty == "void";
    let is_float_ret = matches!(&main_func.ret_ty, Type::Prim(PrimType::F64))
        || matches!(&main_func.ret_ty, Type::Var(n) if n.as_ref() == "F" || n.as_ref() == "Float");

    output.push_str("\n; C main entry point\n");
    output.push_str("define i32 @main(i32 %argc, i8** %argv) {\n");
    output.push_str("entry:\n");

    // Filter out void/unit parameters
    let real_params: Vec<_> = main_func.params.iter()
        .filter(|ty| emit_type(ty).map(|t| t != "void").unwrap_or(true))
        .collect();

    // Helper to emit the call, handling void return type
    let emit_call = |output: &mut String, args: &str| {
        if is_void_ret {
            output.push_str(&format!("  call void @goth_main({})\n", args));
        } else {
            output.push_str(&format!("  %result = call {} @goth_main({})\n", ret_ty, args));
        }
    };

    // If goth main takes arguments, we need to parse them from argv
    if real_params.is_empty() {
        // No arguments - just call
        emit_call(&mut output, "");
    } else if real_params.len() == 1 {
        let param_ty = real_params[0];
        let llvm_ty = emit_type(param_ty)?;
        let is_float_param = matches!(param_ty, Type::Prim(PrimType::F64))
            || matches!(param_ty, Type::Var(n) if n.as_ref() == "F" || n.as_ref() == "Float");

        // One argument - parse first command line arg
        output.push_str("  ; Check if we have command line args\n");
        output.push_str("  %has_args = icmp sgt i32 %argc, 1\n");
        output.push_str("  br i1 %has_args, label %parse_arg, label %use_default\n");
        output.push_str("\n");
        output.push_str("parse_arg:\n");
        output.push_str("  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1\n");
        output.push_str("  %argv1 = load i8*, i8** %argv1_ptr\n");

        if is_float_param {
            output.push_str("  %parsed = call double @atof(i8* %argv1)\n");
        } else {
            output.push_str("  %parsed = call i64 @atol(i8* %argv1)\n");
        }

        output.push_str("  br label %call_main\n");
        output.push_str("\n");
        output.push_str("use_default:\n");
        output.push_str("  br label %call_main\n");
        output.push_str("\n");
        output.push_str("call_main:\n");

        if is_float_param {
            output.push_str("  %arg0 = phi double [ %parsed, %parse_arg ], [ 0.0, %use_default ]\n");
        } else {
            output.push_str(&format!("  %arg0 = phi {} [ %parsed, %parse_arg ], [ 0, %use_default ]\n", llvm_ty));
        }

        emit_call(&mut output, &format!("{} %arg0", llvm_ty));
    } else {
        // Multiple arguments - parse each from argv
        let mut arg_calls = Vec::new();

        for (i, param_ty) in real_params.iter().enumerate() {
            let llvm_ty = emit_type(param_ty)?;
            let is_float = matches!(param_ty, Type::Prim(PrimType::F64))
                || matches!(param_ty, Type::Var(n) if n.as_ref() == "F" || n.as_ref() == "Float");

            let arg_idx = i + 1; // argv[0] is program name
            output.push_str(&format!(
                "  %argv{}_ptr = getelementptr i8*, i8** %argv, i64 {}\n",
                arg_idx, arg_idx
            ));
            output.push_str(&format!(
                "  %argv{} = load i8*, i8** %argv{}_ptr\n",
                arg_idx, arg_idx
            ));

            if is_float {
                output.push_str(&format!(
                    "  %arg{} = call double @atof(i8* %argv{})\n",
                    i, arg_idx
                ));
            } else {
                output.push_str(&format!(
                    "  %arg{} = call i64 @atol(i8* %argv{})\n",
                    i, arg_idx
                ));
            }

            arg_calls.push(format!("{} %arg{}", llvm_ty, i));
        }

        emit_call(&mut output, &arg_calls.join(", "));
    }

    output.push_str("\n");

    // Only print result if not void
    if !is_void_ret {
        output.push_str("  ; Print result\n");

        if is_float_ret {
            output.push_str("  call void @goth_print_f64(double %result)\n");
        } else {
            output.push_str(&format!("  call void @goth_print_i64({} %result)\n", ret_ty));
        }

        output.push_str("  call void @goth_print_newline()\n");
    }

    output.push_str("\n");
    output.push_str("  ; Return 0\n");
    output.push_str("  ret i32 0\n");
    output.push_str("}\n");

    Ok(output)
}

/// Emit complete program
pub fn emit_program(program: &Program) -> Result<String> {
    let mut output = String::new();

    // Module header
    output.push_str("; ModuleID = 'goth_program'\n");
    output.push_str("source_filename = \"goth_program.goth\"\n");
    output.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n");
    output.push_str("\n");

    // Runtime declarations
    output.push_str(&emit_runtime_declarations());

    // Format strings
    output.push_str(&emit_format_strings());

    // Emit all functions and collect string literals
    // We need to track string literal renaming for each function
    let main_func = program.functions.iter().find(|f| f.name == "main");
    let mut all_string_literals: Vec<(String, String)> = Vec::new();
    let mut function_outputs: Vec<String> = Vec::new();
    let mut global_str_idx = 0usize;

    for func in &program.functions {
        let is_main = func.name == "main";
        let (mut func_code, string_literals) = emit_function(func, is_main)?;

        // Rename string literals to be globally unique using two-phase approach
        // to avoid partial substring matches

        // Sort by old name index descending to avoid partial matches within each phase
        let mut sorted_literals: Vec<_> = string_literals.into_iter().collect();
        sorted_literals.sort_by(|(a, _), (b, _)| {
            let get_idx = |s: &str| -> usize {
                s.strip_prefix("@.str.").and_then(|n| n.parse().ok()).unwrap_or(0)
            };
            get_idx(b).cmp(&get_idx(a))  // Descending order
        });

        // Phase 1: Replace old names with unique temporary placeholders
        let mut temp_to_final: Vec<(String, String)> = Vec::new();
        for (i, (old_name, value)) in sorted_literals.into_iter().enumerate() {
            let temp_name = format!("@__goth_tmp_str_{}_{}__", global_str_idx + i, func.name);
            let final_name = format!("@.str.{}", global_str_idx + i);
            func_code = func_code.replace(&old_name, &temp_name);
            temp_to_final.push((temp_name, final_name.clone()));
            all_string_literals.push((final_name, value));
        }
        global_str_idx += temp_to_final.len();

        // Phase 2: Replace temp placeholders with final names
        for (temp_name, final_name) in temp_to_final {
            func_code = func_code.replace(&temp_name, &final_name);
        }

        function_outputs.push(func_code);
    }

    // Emit collected string literals before functions
    if !all_string_literals.is_empty() {
        output.push_str("; String literals\n");
        for (name, value) in &all_string_literals {
            let escaped = escape_string_for_llvm(value);
            let len = value.len() + 1; // Include null terminator
            output.push_str(&format!(
                "{} = private unnamed_addr constant [{} x i8] c\"{}\", align 1\n",
                name, len, escaped
            ));
        }
        output.push_str("\n");
    }

    // Output functions
    for func_code in function_outputs {
        output.push_str(&func_code);
        output.push_str("\n");
    }

    // Emit C main wrapper if we have a goth main
    if let Some(main_fn) = main_func {
        output.push_str(&emit_c_main(main_fn)?);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_type_i64() {
        assert_eq!(emit_type(&Type::Prim(PrimType::I64)).unwrap(), "i64");
    }

    #[test]
    fn test_emit_type_f64() {
        assert_eq!(emit_type(&Type::Prim(PrimType::F64)).unwrap(), "double");
    }

    #[test]
    fn test_emit_type_bool() {
        assert_eq!(emit_type(&Type::Prim(PrimType::Bool)).unwrap(), "i1");
    }

    #[test]
    fn test_emit_type_var() {
        assert_eq!(
            emit_type(&Type::Var("I".into())).unwrap(),
            "i64"
        );
    }
}
