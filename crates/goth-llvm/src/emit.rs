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
        Type::Prim(PrimType::I64) => true,
        Type::Var(name) => matches!(name.as_ref(), "I" | "Int" | "ℤ" | "N" | "Nat" | "ℕ"),
        _ => false,
    }
}

/// Check if a type is float-like
fn is_float_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::F64) => true,
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

/// Analyze a function to find locals that are assigned in multiple blocks.
/// These need stack allocation (alloca/store/load) instead of direct SSA.
fn find_multi_block_locals(func: &Function) -> HashMap<LocalId, Type> {
    let mut assignments: HashMap<LocalId, HashSet<usize>> = HashMap::new();
    let mut local_types: HashMap<LocalId, Type> = HashMap::new();

    // Track assignments in entry block (block 0)
    for stmt in &func.body.stmts {
        assignments.entry(stmt.dest).or_default().insert(0);
        local_types.insert(stmt.dest, stmt.ty.clone());
    }

    // Track assignments in other blocks
    for (block_id, block) in &func.blocks {
        for stmt in &block.stmts {
            assignments.entry(stmt.dest).or_default().insert(block_id.0 as usize);
            local_types.insert(stmt.dest, stmt.ty.clone());
        }
    }

    // Return locals assigned in more than one block
    assignments
        .into_iter()
        .filter(|(_, blocks)| blocks.len() > 1)
        .filter_map(|(local, _)| local_types.get(&local).map(|ty| (local, ty.clone())))
        .collect()
}

/// Emit LLVM type
pub fn emit_type(ty: &Type) -> Result<String> {
    match ty {
        Type::Prim(PrimType::I64) => Ok("i64".to_string()),
        Type::Prim(PrimType::F64) => Ok("double".to_string()),
        Type::Prim(PrimType::Bool) => Ok("i1".to_string()),
        Type::Prim(PrimType::Char) => Ok("i8".to_string()),

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
            // LLVM requires hex format for floats
            return Ok((format!("{:e}", f), String::new()));
        }
        Constant::Bool(b) => {
            let val = if *b { "1" } else { "0" };
            return Ok((val.to_string(), String::new()));
        }
        Constant::Unit => {
            return Ok(("void".to_string(), String::new()));
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
                let llvm_ty = emit_type(ty)?;
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
) -> Result<(String, String)> {
    let ssa = ctx.fresh_ssa();
    let llvm_ty = emit_type(ty)?;

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
        let op_name = match op {
            goth_ast::op::BinOp::And => "and",
            goth_ast::op::BinOp::Or => "or",
            goth_ast::op::BinOp::Eq => "icmp eq",
            goth_ast::op::BinOp::Neq => "icmp ne",
            _ => return Err(LlvmError::UnsupportedOp(format!("{:?}", op))),
        };
        (op_name, "i1".to_string())
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
            let left_val = emit_operand(ctx, left, &stmt.ty, output)?;
            let right_val = emit_operand(ctx, right, &stmt.ty, output)?;
            emit_binop(ctx, op, &left_val, &right_val, &stmt.ty)?
        }

        Rhs::UnaryOp(op, operand) => {
            let op_val = emit_operand(ctx, operand, &stmt.ty, output)?;
            let llvm_ty = emit_type(&stmt.ty)?;

            // Handle Sum/Prod specially - they need intermediate values
            match op {
                goth_ast::op::UnaryOp::Sum => {
                    // Sum reduction: get length first, then call goth_sum_i64
                    let len_ssa = ctx.fresh_ssa();
                    output.push_str(&format!(
                        "  {} = call i64 @goth_len(i8* {})\n",
                        len_ssa, op_val
                    ));
                    let ssa = ctx.fresh_ssa();
                    let code = format!(
                        "  {} = call i64 @goth_sum_i64(i8* {}, i64 {})\n",
                        ssa, op_val, len_ssa
                    );
                    (ssa, code)
                }
                goth_ast::op::UnaryOp::Prod => {
                    // Product reduction: get length first, then call goth_prod_i64
                    let len_ssa = ctx.fresh_ssa();
                    output.push_str(&format!(
                        "  {} = call i64 @goth_len(i8* {})\n",
                        len_ssa, op_val
                    ));
                    let ssa = ctx.fresh_ssa();
                    let code = format!(
                        "  {} = call i64 @goth_prod_i64(i8* {}, i64 {})\n",
                        ssa, op_val, len_ssa
                    );
                    (ssa, code)
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
                        _ => return Err(LlvmError::UnsupportedOp(format!("{:?}", op))),
                    };
                    (ssa, code)
                }
            }
        }

        Rhs::Call { func, args } => {
            let ssa = ctx.fresh_ssa();
            let ret_ty = emit_type(&stmt.ty)?;

            let mut arg_strs = Vec::new();
            for arg in args {
                // Need to infer arg type - for now assume i64
                let arg_ty = Type::Prim(PrimType::I64);
                let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                arg_strs.push(format!("i64 {}", arg_val));
            }

            let code = format!(
                "  {} = call {} @{}({})\n",
                ssa,
                ret_ty,
                func,
                arg_strs.join(", ")
            );

            (ssa, code)
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
            let ssa = ctx.fresh_ssa();
            let tensor_ty = Type::Prim(PrimType::I64); // Placeholder
            let tensor_val = emit_operand(ctx, tensor, &tensor_ty, output)?;

            // Get length first
            let len_ssa = ctx.fresh_ssa();
            output.push_str(&format!(
                "  {} = call i64 @goth_len(i8* {})\n",
                len_ssa, tensor_val
            ));

            let func_name = match op {
                ReduceOp::Sum => "goth_sum_i64",
                ReduceOp::Prod => "goth_prod_i64",
                ReduceOp::Min => "goth_min_i64",
                ReduceOp::Max => "goth_max_i64",
            };

            let code = format!(
                "  {} = call i64 @{}(i8* {}, i64 {})\n",
                ssa, func_name, tensor_val, len_ssa
            );

            (ssa, code)
        }

        Rhs::Index(arr, idx) => {
            let ssa = ctx.fresh_ssa();
            let arr_ty = Type::Prim(PrimType::I64);
            let arr_val = emit_operand(ctx, arr, &arr_ty, output)?;
            let idx_val = emit_operand(ctx, idx, &Type::Prim(PrimType::I64), output)?;

            let code = format!(
                "  {} = call i64 @goth_index_i64(i8* {}, i64 {})\n",
                ssa, arr_val, idx_val
            );

            (ssa, code)
        }

        Rhs::Prim { name, args } => {
            let ssa = ctx.fresh_ssa();
            let ret_ty = emit_type(&stmt.ty)?;

            // Handle specific primitives
            match name.as_str() {
                "print" => {
                    if let Some(arg) = args.first() {
                        let arg_ty = Type::Prim(PrimType::I64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        // For now, assume i64
                        let code = format!("  call void @goth_print_i64(i64 {})\n", arg_val);
                        output.push_str(&code);
                        let code = "  call void @goth_print_newline()\n".to_string();
                        return Ok(());
                    }
                    (ssa, String::new())
                }
                "len" => {
                    if let Some(arg) = args.first() {
                        let arr_ty = Type::Prim(PrimType::I64);
                        let arr_val = emit_operand(ctx, arg, &arr_ty, output)?;
                        let code = format!("  {} = call i64 @goth_len(i8* {})\n", ssa, arr_val);
                        (ssa, code)
                    } else {
                        (ssa, String::new())
                    }
                }
                "reverse" => {
                    if let Some(arg) = args.first() {
                        let arr_ty = Type::Prim(PrimType::I64);
                        let arr_val = emit_operand(ctx, arg, &arr_ty, output)?;
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
                        (ssa, String::new())
                    }
                }
                _ => {
                    // Generic primitive - emit as function call
                    let mut arg_strs = Vec::new();
                    for arg in args {
                        let arg_ty = Type::Prim(PrimType::I64);
                        let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                        arg_strs.push(format!("i64 {}", arg_val));
                    }
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

        Rhs::Tuple(ops) => {
            // For now, just use the first element or return 0
            if let Some(first) = ops.first() {
                let val = emit_operand(ctx, first, &stmt.ty, output)?;
                (val, String::new())
            } else {
                ("0".to_string(), String::new())
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
            let ssa = ctx.fresh_ssa();
            let ret_ty = emit_type(&stmt.ty)?;
            let closure_val = emit_operand(ctx, closure, &stmt.ty, output)?;

            let mut arg_strs = Vec::new();
            for arg in args {
                let arg_ty = Type::Prim(PrimType::I64);
                let arg_val = emit_operand(ctx, arg, &arg_ty, output)?;
                arg_strs.push(format!("i64 {}", arg_val));
            }

            // Indirect call
            let code = format!(
                "  {} = call {} {}({})\n",
                ssa,
                ret_ty,
                closure_val,
                arg_strs.join(", ")
            );

            (ssa, code)
        }

        _ => {
            return Err(LlvmError::UnsupportedOp(format!(
                "Statement: {:?}",
                stmt.rhs
            )));
        }
    };

    output.push_str(&code);

    // If this is a stack-allocated local, emit a store instead of registering SSA
    if let Some(ptr) = ctx.get_stack_ptr(&stmt.dest).cloned() {
        let llvm_ty = emit_type(&stmt.ty)?;
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
            let ret_val = emit_operand(ctx, op, &ret_ty, output)?;
            let llvm_ty = emit_type(&ret_ty)?;
            output.push_str(&format!("  ret {} {}\n", llvm_ty, ret_val));
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

/// Emit function
fn emit_function(func: &Function, is_main: bool) -> Result<String> {
    let mut ctx = LlvmContext::new(func.ret_ty.clone());
    let mut output = String::new();

    // Function signature
    let ret_type = emit_type(&func.ret_ty)?;

    let param_strs: Vec<String> = func
        .params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let llvm_ty = emit_type(ty).unwrap_or_else(|_| "i64".to_string());
            let param_ssa = format!("%arg{}", i);
            // Register parameter
            ctx.register_local(LocalId::new(i as u32), param_ssa.clone(), ty.clone());
            format!("{} {}", llvm_ty, param_ssa)
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
        let ptr_name = ctx.fresh_ssa();
        output.push_str(&format!("  {} = alloca {}\n", ptr_name, llvm_ty));
        ctx.register_stack_local(*local, ptr_name);
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

    Ok(output)
}

/// Emit the C main function that calls goth_main
fn emit_c_main(main_func: &Function) -> Result<String> {
    let mut output = String::new();

    let ret_ty = emit_type(&main_func.ret_ty)?;
    let is_float_ret = matches!(&main_func.ret_ty, Type::Prim(PrimType::F64))
        || matches!(&main_func.ret_ty, Type::Var(n) if n.as_ref() == "F" || n.as_ref() == "Float");

    output.push_str("\n; C main entry point\n");
    output.push_str("define i32 @main(i32 %argc, i8** %argv) {\n");
    output.push_str("entry:\n");

    // If goth main takes arguments, we need to parse them from argv
    if main_func.params.is_empty() {
        // No arguments - just call
        output.push_str(&format!("  %result = call {} @goth_main()\n", ret_ty));
    } else if main_func.params.len() == 1 {
        let param_ty = &main_func.params[0];
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

        output.push_str(&format!(
            "  %result = call {} @goth_main({} %arg0)\n",
            ret_ty, llvm_ty
        ));
    } else {
        // Multiple arguments - parse each from argv
        let mut arg_calls = Vec::new();

        for (i, param_ty) in main_func.params.iter().enumerate() {
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

        output.push_str(&format!(
            "  %result = call {} @goth_main({})\n",
            ret_ty,
            arg_calls.join(", ")
        ));
    }

    output.push_str("\n");
    output.push_str("  ; Print result\n");

    if is_float_ret {
        output.push_str("  call void @goth_print_f64(double %result)\n");
    } else {
        output.push_str(&format!("  call void @goth_print_i64({} %result)\n", ret_ty));
    }

    output.push_str("  call void @goth_print_newline()\n");
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

    // Emit all functions
    let main_func = program.functions.iter().find(|f| f.name == "main");

    for func in &program.functions {
        let is_main = func.name == "main";
        output.push_str(&emit_function(func, is_main)?);
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
