//! MLIR code emission from MIR
//!
//! Emits MLIR text format using standard dialects:
//! - `func` dialect for functions
//! - `arith` dialect for arithmetic operations
//! - `cf` dialect for unstructured control flow
//! - `tensor` dialect for array operations
//! - `goth` dialect for Goth-specific operations
//!
//! This module provides two implementations:
//! 1. Legacy implementation (MlirContext) - used by existing code
//! 2. New implementation (MlirBuilder) - uses the modular dialect approach
//!
//! The new implementation is available via `emit_program_v2` and will
//! eventually replace the legacy implementation.

use crate::error::{MlirError, Result};
use crate::context::TextMlirContext;
use crate::builder::MlirBuilder;
use crate::types::type_to_mlir_string;
use goth_ast::types::{Type, PrimType};
use goth_mir::mir::*;
use std::collections::HashMap;

/// MLIR emission context
pub struct MlirContext {
    /// Current indentation level
    indent: usize,
    /// SSA value counter
    next_ssa: usize,
    /// Local variable to SSA value mapping
    local_map: HashMap<LocalId, String>,
    /// Type information for locals
    local_types: HashMap<LocalId, Type>,
}

impl MlirContext {
    pub fn new() -> Self {
        MlirContext {
            indent: 0,
            next_ssa: 0,
            local_map: HashMap::new(),
            local_types: HashMap::new(),
        }
    }
    
    /// Generate fresh SSA value name
    fn fresh_ssa(&mut self) -> String {
        let name = format!("%{}", self.next_ssa);
        self.next_ssa += 1;
        name
    }
    
    /// Get SSA value for a local
    fn get_ssa(&self, local: &LocalId) -> Result<String> {
        self.local_map.get(local)
            .cloned()
            .ok_or_else(|| MlirError::CodeGen(format!("Undefined local: {:?}", local)))
    }
    
    /// Register local with SSA value
    fn register_local(&mut self, local: LocalId, ssa: String, ty: Type) {
        self.local_map.insert(local, ssa);
        self.local_types.insert(local, ty);
    }
    
    /// Increase indentation
    fn push_indent(&mut self) {
        self.indent += 1;
    }
    
    /// Decrease indentation
    fn pop_indent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }
    
    /// Get current indentation string
    fn indent_str(&self) -> String {
        "  ".repeat(self.indent)
    }
}

/// Emit MLIR type
pub fn emit_type(ty: &Type) -> Result<String> {
    match ty {
        Type::Prim(PrimType::I64) => Ok("i64".to_string()),
        Type::Prim(PrimType::F64) => Ok("f64".to_string()),
        Type::Prim(PrimType::Bool) => Ok("i1".to_string()),
        Type::Prim(PrimType::String) => Ok("!llvm.ptr<i8>".to_string()),

        Type::Tuple(fields) if fields.is_empty() => Ok("()".to_string()),
        
        Type::Tuple(fields) => {
            let field_types: Result<Vec<_>> = fields.iter()
                .map(|f| emit_type(&f.ty))
                .collect();
            Ok(format!("tuple<{}>", field_types?.join(", ")))
        }
        
        Type::Tensor(shape, elem) => {
            use goth_ast::shape::Dim;
            let elem_ty = emit_type(elem)?;
            
            // Build shape string
            let shape_str: Vec<String> = shape.0.iter().map(|dim| {
                match dim {
                    Dim::Const(n) => n.to_string(),
                    _ => "?".to_string(),
                }
            }).collect();
            
            if shape_str.is_empty() {
                Ok(format!("tensor<{}>", elem_ty))
            } else {
                Ok(format!("tensor<{}x{}>", shape_str.join("x"), elem_ty))
            }
        }
        
        Type::Fn(arg, ret) => {
            let arg_ty = emit_type(arg)?;
            let ret_ty = emit_type(ret)?;
            Ok(format!("({}) -> {}", arg_ty, ret_ty))
        }

        // Type variables - map to concrete MLIR types
        Type::Var(name) => {
            match name.as_ref() {
                "I" | "Int" | "ℤ" => Ok("i64".to_string()),
                "F" | "Float" => Ok("f64".to_string()),
                "B" | "Bool" => Ok("i1".to_string()),
                "N" | "Nat" | "ℕ" => Ok("i64".to_string()),  // Naturals as i64
                _ => Ok("i64".to_string()),  // Default to i64
            }
        }

        _ => Err(MlirError::UnsupportedType(format!("{:?}", ty))),
    }
}

/// Emit MLIR constant
fn emit_constant(ctx: &mut MlirContext, constant: &Constant, ty: &Type) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let mlir_ty = emit_type(ty)?;
    
    let value = match constant {
        Constant::Int(n) => {
            format!("{}{} = arith.constant {} : {}", 
                ctx.indent_str(), ssa, n, mlir_ty)
        }
        Constant::Float(f) => {
            format!("{}{} = arith.constant {} : {}", 
                ctx.indent_str(), ssa, f, mlir_ty)
        }
        Constant::Bool(b) => {
            let val = if *b { "true" } else { "false" };
            format!("{}{} = arith.constant {} : {}", 
                ctx.indent_str(), ssa, val, mlir_ty)
        }
        Constant::Unit => {
            format!("{}{} = arith.constant () : ()",
                ctx.indent_str(), ssa)
        }
        Constant::String(s) => {
            // Use memref for string literals
            format!("{}{} = arith.constant \"{}\" : !llvm.ptr<i8>",
                ctx.indent_str(), ssa, s.escape_default())
        }
    };
    
    Ok(format!("{}\n", value))
}

/// Emit operand (returns SSA value)
fn emit_operand(ctx: &mut MlirContext, op: &Operand, output: &mut String) -> Result<String> {
    match op {
        Operand::Const(c) => {
            // Need type information - assume i64 for now
            // TODO: Thread type information through
            let ty = match c {
                Constant::Int(_) => Type::Prim(PrimType::I64),
                Constant::Float(_) => Type::Prim(PrimType::F64),
                Constant::Bool(_) => Type::Prim(PrimType::Bool),
                Constant::String(_) => Type::Prim(PrimType::String),
                Constant::Unit => Type::Tuple(vec![]),
            };
            
            let const_code = emit_constant(ctx, c, &ty)?;
            output.push_str(&const_code);
            
            // Extract the SSA value we just created
            let ssa = ctx.fresh_ssa();
            let prev_ssa = format!("%{}", ctx.next_ssa - 2);
            Ok(prev_ssa)
        }
        
        Operand::Local(local) => {
            ctx.get_ssa(local)
        }
    }
}

/// Check if a type is integer-like (i64, I, Int, etc.)
fn is_int_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::I64) => true,
        Type::Var(name) => matches!(name.as_ref(), "I" | "Int" | "ℤ" | "N" | "Nat" | "ℕ"),
        _ => false,
    }
}

/// Check if a type is float-like (f64, F, Float, etc.)
fn is_float_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::F64) => true,
        Type::Var(name) => matches!(name.as_ref(), "F" | "Float"),
        _ => false,
    }
}

/// Check if type is boolean
fn is_bool_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::Bool) => true,
        Type::Var(name) => matches!(name.as_ref(), "B" | "Bool"),
        _ => false,
    }
}

/// Emit binary operation
fn emit_binop(ctx: &mut MlirContext, op: &goth_ast::op::BinOp,
              left: String, right: String, ty: &Type) -> Result<String> {
    let ssa = ctx.fresh_ssa();

    // For Bool result type, handle logical ops and comparisons specially
    if is_bool_type(ty) {
        match op {
            // Logical operations on Bool operands
            goth_ast::op::BinOp::And => {
                return Ok(format!("{}{} = arith.andi {}, {} : i1\n",
                    ctx.indent_str(), ssa, left, right));
            }
            goth_ast::op::BinOp::Or => {
                return Ok(format!("{}{} = arith.ori {}, {} : i1\n",
                    ctx.indent_str(), ssa, left, right));
            }
            // Comparison operations - default to integer comparison
            // (The operands are integers, result is Bool)
            goth_ast::op::BinOp::Lt => {
                return Ok(format!("{}{} = arith.cmpi slt, {}, {} : i64\n",
                    ctx.indent_str(), ssa, left, right));
            }
            goth_ast::op::BinOp::Gt => {
                return Ok(format!("{}{} = arith.cmpi sgt, {}, {} : i64\n",
                    ctx.indent_str(), ssa, left, right));
            }
            goth_ast::op::BinOp::Leq => {
                return Ok(format!("{}{} = arith.cmpi sle, {}, {} : i64\n",
                    ctx.indent_str(), ssa, left, right));
            }
            goth_ast::op::BinOp::Geq => {
                return Ok(format!("{}{} = arith.cmpi sge, {}, {} : i64\n",
                    ctx.indent_str(), ssa, left, right));
            }
            goth_ast::op::BinOp::Eq => {
                return Ok(format!("{}{} = arith.cmpi eq, {}, {} : i64\n",
                    ctx.indent_str(), ssa, left, right));
            }
            goth_ast::op::BinOp::Neq => {
                return Ok(format!("{}{} = arith.cmpi ne, {}, {} : i64\n",
                    ctx.indent_str(), ssa, left, right));
            }
            _ => return Err(MlirError::UnsupportedOp(format!("Bool {:?}", op))),
        }
    }

    let mlir_ty = emit_type(ty)?;

    let op_name = if is_int_type(ty) {
        match op {
            goth_ast::op::BinOp::Add => "arith.addi",
            goth_ast::op::BinOp::Sub => "arith.subi",
            goth_ast::op::BinOp::Mul => "arith.muli",
            goth_ast::op::BinOp::Div => "arith.divsi",
            goth_ast::op::BinOp::Mod => "arith.remsi",
            goth_ast::op::BinOp::Lt => "arith.cmpi slt,",
            goth_ast::op::BinOp::Gt => "arith.cmpi sgt,",
            goth_ast::op::BinOp::Leq => "arith.cmpi sle,",
            goth_ast::op::BinOp::Geq => "arith.cmpi sge,",
            goth_ast::op::BinOp::Eq => "arith.cmpi eq,",
            goth_ast::op::BinOp::Neq => "arith.cmpi ne,",
            _ => return Err(MlirError::UnsupportedOp(format!("{:?}", op))),
        }
    } else if is_float_type(ty) {
        match op {
            goth_ast::op::BinOp::Add => "arith.addf",
            goth_ast::op::BinOp::Sub => "arith.subf",
            goth_ast::op::BinOp::Mul => "arith.mulf",
            goth_ast::op::BinOp::Div => "arith.divf",
            goth_ast::op::BinOp::Lt => "arith.cmpf olt,",
            goth_ast::op::BinOp::Gt => "arith.cmpf ogt,",
            goth_ast::op::BinOp::Leq => "arith.cmpf ole,",
            goth_ast::op::BinOp::Geq => "arith.cmpf oge,",
            goth_ast::op::BinOp::Eq => "arith.cmpf oeq,",
            goth_ast::op::BinOp::Neq => "arith.cmpf one,",
            _ => return Err(MlirError::UnsupportedOp(format!("{:?}", op))),
        }
    } else {
        return Err(MlirError::UnsupportedType(format!("{:?}", ty)));
    };

    Ok(format!("{}{} = {} {}, {} : {}\n",
        ctx.indent_str(), ssa, op_name, left, right, mlir_ty))
}

/// Emit unary operation
fn emit_unary(ctx: &mut MlirContext, op: &goth_ast::op::UnaryOp,
              operand: String, ty: &Type) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let mlir_ty = emit_type(ty)?;

    // Handle reduction operations specially (they reduce tensor to scalar)
    match op {
        goth_ast::op::UnaryOp::Sum => {
            return Ok(format!("{}{} = goth.reduce_sum {} : {}\n",
                ctx.indent_str(), ssa, operand, mlir_ty));
        }
        goth_ast::op::UnaryOp::Prod => {
            return Ok(format!("{}{} = goth.reduce_prod {} : {}\n",
                ctx.indent_str(), ssa, operand, mlir_ty));
        }
        _ => {}
    }

    let op_name = match op {
        goth_ast::op::UnaryOp::Neg => {
            if is_int_type(ty) {
                "arith.subi"
            } else if is_float_type(ty) {
                "arith.negf"
            } else {
                return Err(MlirError::UnsupportedType(format!("{:?}", ty)));
            }
        },
        goth_ast::op::UnaryOp::Sqrt => "math.sqrt",
        goth_ast::op::UnaryOp::Floor => "math.floor",
        goth_ast::op::UnaryOp::Ceil => "math.ceil",
        goth_ast::op::UnaryOp::Not => "arith.xori",  // XOR with 1 for boolean NOT
        _ => return Err(MlirError::UnsupportedOp(format!("{:?}", op))),
    };

    if matches!(op, goth_ast::op::UnaryOp::Neg) && is_int_type(ty) {
        // Integer negation: 0 - x
        let zero = ctx.fresh_ssa();
        let mut code = format!("{}{} = arith.constant 0 : {}\n",
            ctx.indent_str(), zero, mlir_ty);
        code.push_str(&format!("{}{} = {} {}, {} : {}\n",
            ctx.indent_str(), ssa, op_name, zero, operand, mlir_ty));
        Ok(code)
    } else if matches!(op, goth_ast::op::UnaryOp::Not) {
        // Boolean NOT: XOR with 1
        let one = ctx.fresh_ssa();
        let mut code = format!("{}{} = arith.constant true : {}\n",
            ctx.indent_str(), one, mlir_ty);
        code.push_str(&format!("{}{} = {} {}, {} : {}\n",
            ctx.indent_str(), ssa, op_name, operand, one, mlir_ty));
        Ok(code)
    } else {
        Ok(format!("{}{} = {} {} : {}\n",
            ctx.indent_str(), ssa, op_name, operand, mlir_ty))
    }
}

/// Emit statement
fn emit_stmt(ctx: &mut MlirContext, stmt: &Stmt, output: &mut String) -> Result<()> {
    let (ssa, code) = match &stmt.rhs {
        Rhs::Use(op) => {
            let ssa = ctx.fresh_ssa();
            let op_ssa = emit_operand(ctx, op, output)?;
            // Just an assignment
            let code = format!("{}{} = {} : {}\n", 
                ctx.indent_str(), ssa, op_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }
        
        Rhs::BinOp(op, left, right) => {
            let left_ssa = emit_operand(ctx, left, output)?;
            let right_ssa = emit_operand(ctx, right, output)?;
            let code = emit_binop(ctx, op, left_ssa, right_ssa, &stmt.ty)?;
            // Extract ssa from binop
            let ssa = format!("%{}", ctx.next_ssa - 1);
            (ssa, code)
        }
        
        Rhs::UnaryOp(op, operand) => {
            let op_ssa = emit_operand(ctx, operand, output)?;
            let code = emit_unary(ctx, op, op_ssa, &stmt.ty)?;
            // Extract ssa from unary
            let ssa = format!("%{}", ctx.next_ssa - 1);
            (ssa, code)
        }
        
        Rhs::Tuple(ops) => {
            let ssa = ctx.fresh_ssa();
            let mut op_ssas = Vec::new();
            for op in ops {
                op_ssas.push(emit_operand(ctx, op, output)?);
            }
            
            // Create tuple - using unrealized_conversion_cast for now
            // TODO: Use proper tuple construction
            let code = format!("{}{} = builtin.unrealized_conversion_cast {}: ({}) -> tuple<{}>\n",
                ctx.indent_str(), ssa, 
                op_ssas.join(", "),
                op_ssas.iter().map(|_| "?").collect::<Vec<_>>().join(", "),
                emit_type(&stmt.ty)?);
            (ssa, code)
        }
        
        Rhs::TupleField(tup, idx) => {
            let ssa = ctx.fresh_ssa();
            let tup_ssa = emit_operand(ctx, tup, output)?;
            // Extract tuple element
            let code = format!("{}{} = builtin.unrealized_conversion_cast {}[{}] : tuple<...> -> {}\n",
                ctx.indent_str(), ssa, tup_ssa, idx, emit_type(&stmt.ty)?);
            (ssa, code)
        }
        
        Rhs::Array(elems) => {
            let ssa = ctx.fresh_ssa();
            let mut elem_ssas = Vec::new();
            for elem in elems {
                elem_ssas.push(emit_operand(ctx, elem, output)?);
            }

            // Create tensor
            let code = format!("{}{} = tensor.from_elements {} : {}\n",
                ctx.indent_str(), ssa,
                elem_ssas.join(", "),
                emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::ArrayFill { size, value } => {
            let ssa = ctx.fresh_ssa();
            let size_ssa = emit_operand(ctx, size, output)?;
            let value_ssa = emit_operand(ctx, value, output)?;

            // Use linalg.fill or tensor.generate for filled arrays
            // For now, use a comment placeholder
            let code = format!("{}// {}: ArrayFill(size={}, value={})\n{}{} = tensor.splat {} : {}\n",
                ctx.indent_str(), ssa, size_ssa, value_ssa,
                ctx.indent_str(), ssa, value_ssa,
                emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::MakeClosure { func, captures } => {
            let ssa = ctx.fresh_ssa();
            // Closure creation - pack function pointer + environment
            let mut cap_ssas = Vec::new();
            for cap in captures {
                cap_ssas.push(emit_operand(ctx, cap, output)?);
            }
            
            // For now, use comment to indicate closure creation
            let code = format!("{}// {}: MakeClosure({}, [{}])\n",
                ctx.indent_str(), ssa, func, cap_ssas.join(", "));
            (ssa, code)
        }
        
        Rhs::ClosureCall { closure, args } => {
            let ssa = ctx.fresh_ssa();
            let closure_ssa = emit_operand(ctx, closure, output)?;
            let mut arg_ssas = Vec::new();
            for arg in args {
                arg_ssas.push(emit_operand(ctx, arg, output)?);
            }

            // Indirect call through closure
            let code = format!("{}{} = func.call_indirect {}({}) : {} -> {}\n",
                ctx.indent_str(), ssa, closure_ssa,
                arg_ssas.join(", "),
                "...", // Type signature
                emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::Iota(n) => {
            let ssa = ctx.fresh_ssa();
            let n_ssa = emit_operand(ctx, n, output)?;
            // Use linalg.index or custom dialect for iota
            let code = format!("{}{} = goth.iota {} : {}\n",
                ctx.indent_str(), ssa, n_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::Range(start, end) => {
            let ssa = ctx.fresh_ssa();
            let start_ssa = emit_operand(ctx, start, output)?;
            let end_ssa = emit_operand(ctx, end, output)?;
            let code = format!("{}{} = goth.range {}, {} : {}\n",
                ctx.indent_str(), ssa, start_ssa, end_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::Prim { name, args } => {
            let ssa = ctx.fresh_ssa();
            let mut arg_ssas = Vec::new();
            for arg in args {
                arg_ssas.push(emit_operand(ctx, arg, output)?);
            }
            let code = format!("{}{} = goth.{} {} : {}\n",
                ctx.indent_str(), ssa, name, arg_ssas.join(", "), emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::TensorReduce { tensor, op } => {
            let ssa = ctx.fresh_ssa();
            let tensor_ssa = emit_operand(ctx, tensor, output)?;
            let op_name = match op {
                goth_mir::mir::ReduceOp::Sum => "sum",
                goth_mir::mir::ReduceOp::Prod => "prod",
                goth_mir::mir::ReduceOp::Min => "min",
                goth_mir::mir::ReduceOp::Max => "max",
            };
            let code = format!("{}{} = goth.reduce_{} {} : {}\n",
                ctx.indent_str(), ssa, op_name, tensor_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::TensorMap { tensor, func } => {
            let ssa = ctx.fresh_ssa();
            let tensor_ssa = emit_operand(ctx, tensor, output)?;
            let func_ssa = emit_operand(ctx, func, output)?;
            let code = format!("{}{} = goth.map {}, {} : {}\n",
                ctx.indent_str(), ssa, tensor_ssa, func_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::TensorFilter { tensor, pred } => {
            let ssa = ctx.fresh_ssa();
            let tensor_ssa = emit_operand(ctx, tensor, output)?;
            let pred_ssa = emit_operand(ctx, pred, output)?;
            let code = format!("{}{} = goth.filter {}, {} : {}\n",
                ctx.indent_str(), ssa, tensor_ssa, pred_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }

        Rhs::Index(arr, idx) => {
            let ssa = ctx.fresh_ssa();
            let arr_ssa = emit_operand(ctx, arr, output)?;
            let idx_ssa = emit_operand(ctx, idx, output)?;
            let code = format!("{}{} = tensor.extract {}[{}] : {}\n",
                ctx.indent_str(), ssa, arr_ssa, idx_ssa, emit_type(&stmt.ty)?);
            (ssa, code)
        }

        _ => {
            return Err(MlirError::UnsupportedOp(
                format!("Statement: {:?}", stmt.rhs)
            ));
        }
    };
    
    output.push_str(&code);
    ctx.register_local(stmt.dest, ssa, stmt.ty.clone());
    
    Ok(())
}

/// Emit block with optional label
fn emit_block_with_label(ctx: &mut MlirContext, block: &Block, label: Option<&str>, output: &mut String) -> Result<()> {
    // Emit label if provided
    if let Some(lbl) = label {
        output.push_str(&format!("{}^{}:\n", ctx.indent_str(), lbl));
    }

    // Emit statements
    for stmt in &block.stmts {
        emit_stmt(ctx, stmt, output)?;
    }

    // Emit terminator
    match &block.term {
        Terminator::Return(op) => {
            let ret_ssa = emit_operand(ctx, op, output)?;
            output.push_str(&format!("{}func.return {} : {}\n",
                ctx.indent_str(), ret_ssa, "?"));
        }

        Terminator::Goto(block_id) => {
            output.push_str(&format!("{}cf.br ^bb{}\n",
                ctx.indent_str(), block_id.0));
        }

        Terminator::If { cond, then_block, else_block } => {
            let cond_ssa = emit_operand(ctx, cond, output)?;
            output.push_str(&format!("{}cf.cond_br {}, ^bb{}, ^bb{}\n",
                ctx.indent_str(), cond_ssa, then_block.0, else_block.0));
        }

        Terminator::Switch { scrutinee, cases, default } => {
            let scrut_ssa = emit_operand(ctx, scrutinee, output)?;
            let case_str: Vec<String> = cases.iter()
                .map(|(c, b)| format!("{}: ^bb{}", c, b.0))
                .collect();
            output.push_str(&format!("{}cf.switch {}, [{}], ^bb{}\n",
                ctx.indent_str(), scrut_ssa, case_str.join(", "), default.0));
        }

        Terminator::Unreachable => {
            output.push_str(&format!("{}// Unreachable\n", ctx.indent_str()));
        }
    }

    Ok(())
}

/// Emit block (backwards compatible)
fn emit_block(ctx: &mut MlirContext, block: &Block, output: &mut String) -> Result<()> {
    emit_block_with_label(ctx, block, None, output)
}

/// Emit function
pub fn emit_function(func: &Function) -> Result<String> {
    let mut ctx = MlirContext::new();
    let mut output = String::new();

    // Function signature
    let param_types: Result<Vec<_>> = func.params.iter()
        .map(|ty| emit_type(ty))
        .collect();
    let param_types = param_types?;

    let ret_type = emit_type(&func.ret_ty)?;

    // Emit function header
    output.push_str(&format!("func.func @{}(", func.name));

    for (i, ty) in param_types.iter().enumerate() {
        if i > 0 {
            output.push_str(", ");
        }
        let param_ssa = ctx.fresh_ssa();
        output.push_str(&format!("{}: {}", param_ssa, ty));

        // Register parameter as local
        let local_id = LocalId::new(i as u32);
        ctx.register_local(local_id, param_ssa.clone(), func.params[i].clone());
    }

    output.push_str(&format!(") -> {} {{\n", ret_type));

    ctx.push_indent();

    // Emit entry block (with label if there are more blocks)
    if func.blocks.is_empty() {
        emit_block(&mut ctx, &func.body, &mut output)?;
    } else {
        emit_block_with_label(&mut ctx, &func.body, Some("entry"), &mut output)?;

        // Emit additional blocks
        for (block_id, block) in &func.blocks {
            let label = format!("bb{}", block_id.0);
            emit_block_with_label(&mut ctx, block, Some(&label), &mut output)?;
        }
    }

    ctx.pop_indent();
    output.push_str("}\n");

    Ok(output)
}

/// Emit program
pub fn emit_program(program: &Program) -> Result<String> {
    let mut output = String::new();

    // Module header
    output.push_str("module {\n");

    // Emit all functions
    for func in &program.functions {
        output.push_str(&emit_function(func)?);
        output.push_str("\n");
    }

    output.push_str("}\n");

    Ok(output)
}

/// Emit program using the new builder infrastructure
///
/// This is the new implementation that uses the modular dialect approach.
/// It will eventually replace `emit_program` once fully tested.
pub fn emit_program_v2(program: &Program) -> Result<String> {
    let mut ctx = TextMlirContext::new();
    let mut builder = MlirBuilder::new(&mut ctx);
    builder.emit_program(program)?;
    Ok(ctx.into_output())
}

/// Emit a single function using the new builder infrastructure
pub fn emit_function_v2(func: &Function) -> Result<String> {
    let mut ctx = TextMlirContext::new();
    let mut builder = MlirBuilder::new(&mut ctx);
    builder.emit_function(func)?;
    Ok(ctx.into_output())
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_ast::op::BinOp;
    
    #[test]
    fn test_emit_prim_types() {
        assert_eq!(emit_type(&Type::Prim(PrimType::I64)).unwrap(), "i64");
        assert_eq!(emit_type(&Type::Prim(PrimType::F64)).unwrap(), "f64");
        assert_eq!(emit_type(&Type::Prim(PrimType::Bool)).unwrap(), "i1");
    }
    
    #[test]
    fn test_emit_function_type() {
        let ty = Type::func(
            Type::Prim(PrimType::I64),
            Type::Prim(PrimType::I64),
        );
        let mlir = emit_type(&ty).unwrap();
        assert!(mlir.contains("i64") && mlir.contains("->"));
    }
    
    #[test]
    fn test_emit_simple_function() {
        // fn add_one(x: i64) -> i64 { return x + 1 }
        let func = Function {
            name: "add_one".to_string(),
            params: vec![Type::Prim(PrimType::I64)],
            ret_ty: Type::Prim(PrimType::I64),
            body: Block {
                stmts: vec![
                    Stmt {
                        dest: LocalId::new(1),
                        ty: Type::Prim(PrimType::I64),
                        rhs: Rhs::BinOp(
                            BinOp::Add,
                            Operand::Local(LocalId::new(0)), // parameter
                            Operand::Const(Constant::Int(1)),
                        ),
                    },
                ],
                term: Terminator::Return(Operand::Local(LocalId::new(1))),
            },
            blocks: vec![],
                is_closure: false,
        };
        
        let mlir = emit_function(&func).unwrap();
        
        assert!(mlir.contains("func.func @add_one"));
        assert!(mlir.contains("i64"));
        assert!(mlir.contains("arith.addi"));
        assert!(mlir.contains("func.return"));
    }
    
    #[test]
    fn test_emit_constant() {
        let func = Function {
            name: "constant".to_string(),
            params: vec![],
            ret_ty: Type::Prim(PrimType::I64),
            body: Block::with_return(Operand::Const(Constant::Int(42))),
            blocks: vec![],
                is_closure: false,
        };
        
        let mlir = emit_function(&func).unwrap();
        assert!(mlir.contains("arith.constant 42"));
        assert!(mlir.contains("i64"));
    }
    
    #[test]
    fn test_emit_binop_int() {
        // fn test() -> i64 { return 1 + 2 }
        let func = Function {
            name: "test".to_string(),
            params: vec![],
            ret_ty: Type::Prim(PrimType::I64),
            body: Block {
                stmts: vec![
                    Stmt {
                        dest: LocalId::new(0),
                        ty: Type::Prim(PrimType::I64),
                        rhs: Rhs::BinOp(
                            BinOp::Add,
                            Operand::Const(Constant::Int(1)),
                            Operand::Const(Constant::Int(2)),
                        ),
                    },
                ],
                term: Terminator::Return(Operand::Local(LocalId::new(0))),
            },
            blocks: vec![],
                is_closure: false,
        };
        
        let mlir = emit_function(&func).unwrap();
        assert!(mlir.contains("arith.addi"));
    }
    
    #[test]
    fn test_emit_binop_float() {
        // fn test() -> f64 { return 1.0 * 2.0 }
        let func = Function {
            name: "test".to_string(),
            params: vec![],
            ret_ty: Type::Prim(PrimType::F64),
            body: Block {
                stmts: vec![
                    Stmt {
                        dest: LocalId::new(0),
                        ty: Type::Prim(PrimType::F64),
                        rhs: Rhs::BinOp(
                            BinOp::Mul,
                            Operand::Const(Constant::Float(1.0)),
                            Operand::Const(Constant::Float(2.0)),
                        ),
                    },
                ],
                term: Terminator::Return(Operand::Local(LocalId::new(0))),
            },
            blocks: vec![],
                is_closure: false,
        };
        
        let mlir = emit_function(&func).unwrap();
        assert!(mlir.contains("arith.mulf"));
        assert!(mlir.contains("f64"));
    }
    
    #[test]
    fn test_emit_program() {
        let program = Program {
            functions: vec![
                Function {
                    name: "main".to_string(),
                    params: vec![],
                    ret_ty: Type::Prim(PrimType::I64),
                    body: Block::with_return(Operand::Const(Constant::Int(42))),
                    blocks: vec![],
                is_closure: false,
                },
            ],
            entry: "main".to_string(),
        };
        
        let mlir = emit_program(&program).unwrap();
        
        assert!(mlir.contains("module {"));
        assert!(mlir.contains("func.func @main"));
        assert!(mlir.contains("}"));
    }
    
    #[test]
    fn test_emit_multiple_statements() {
        // fn test(x: i64) -> i64 {
        //   let y = x + 1
        //   let z = y * 2
        //   return z
        // }
        let func = Function {
            name: "test".to_string(),
            params: vec![Type::Prim(PrimType::I64)],
            ret_ty: Type::Prim(PrimType::I64),
            body: Block {
                stmts: vec![
                    Stmt {
                        dest: LocalId::new(1),
                        ty: Type::Prim(PrimType::I64),
                        rhs: Rhs::BinOp(
                            BinOp::Add,
                            Operand::Local(LocalId::new(0)),
                            Operand::Const(Constant::Int(1)),
                        ),
                    },
                    Stmt {
                        dest: LocalId::new(2),
                        ty: Type::Prim(PrimType::I64),
                        rhs: Rhs::BinOp(
                            BinOp::Mul,
                            Operand::Local(LocalId::new(1)),
                            Operand::Const(Constant::Int(2)),
                        ),
                    },
                ],
                term: Terminator::Return(Operand::Local(LocalId::new(2))),
            },
            blocks: vec![],
                is_closure: false,
        };
        
        let mlir = emit_function(&func).unwrap();
        
        // Should have two operations
        assert_eq!(mlir.matches("arith.addi").count(), 1);
        assert_eq!(mlir.matches("arith.muli").count(), 1);
    }
    
    #[test]
    fn test_emit_integration_with_mir() {
        use goth_mir::lower_expr;
        use goth_ast::expr::Expr;
        use goth_ast::literal::Literal;
        
        // Simple expression: 1 + 2
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        
        // Lower to MIR
        let mir_program = lower_expr(&expr).unwrap();
        
        // Emit MLIR
        let mlir = emit_program(&mir_program).unwrap();
        
        assert!(mlir.contains("module {"));
        assert!(mlir.contains("func.func @main"));
        assert!(mlir.contains("arith.addi"));
    }
    
    #[test]
    fn test_emit_lambda() {
        use goth_mir::lower_expr;
        use goth_ast::expr::Expr;
        use goth_ast::literal::Literal;
        
        // Lambda: λ→ ₀ + 1
        let expr = Expr::Lam(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),
            Box::new(Expr::Lit(Literal::Int(1))),
        )));
        
        let mir_program = lower_expr(&expr).unwrap();
        let mlir = emit_program(&mir_program).unwrap();
        
        // Should generate lambda function + main
        assert!(mlir.contains("func.func @lambda_0"));
        assert!(mlir.contains("func.func @main"));
    }
    
    #[test]
    fn test_emit_pretty_print() {
        let program = Program {
            functions: vec![
                Function {
                    name: "add".to_string(),
                    params: vec![
                        Type::Prim(PrimType::I64),
                        Type::Prim(PrimType::I64),
                    ],
                    ret_ty: Type::Prim(PrimType::I64),
                    body: Block {
                        stmts: vec![
                            Stmt {
                                dest: LocalId::new(2),
                                ty: Type::Prim(PrimType::I64),
                                rhs: Rhs::BinOp(
                                    BinOp::Add,
                                    Operand::Local(LocalId::new(0)),
                                    Operand::Local(LocalId::new(1)),
                                ),
                            },
                        ],
                        term: Terminator::Return(Operand::Local(LocalId::new(2))),
                    },
                    blocks: vec![],
                is_closure: false,
                },
            ],
            entry: "add".to_string(),
        };
        
        let mlir = emit_program(&program).unwrap();
        println!("\n{}", mlir);
        
        // Verify it's well-formed
        assert!(mlir.lines().count() > 5);
    }
}
