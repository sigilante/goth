//! High-level MLIR builder for Goth
//!
//! This module provides a high-level interface for building MLIR from MIR,
//! abstracting over the dialect operations.

use goth_ast::types::Type;
use goth_ast::op::{BinOp, UnaryOp};
use goth_mir::mir::*;

use crate::context::TextMlirContext;
use crate::types::{type_to_mlir_string, is_integer_type, is_float_type};
use crate::dialects::{arith, func, cf, scf, tensor, goth as goth_dialect};
use crate::error::{MlirError, Result};

/// High-level MLIR builder
///
/// This provides a convenient interface for emitting MLIR from MIR,
/// handling the mapping between MIR constructs and MLIR dialect operations.
pub struct MlirBuilder<'a> {
    ctx: &'a mut TextMlirContext,
}

impl<'a> MlirBuilder<'a> {
    /// Create a new MLIR builder
    pub fn new(ctx: &'a mut TextMlirContext) -> Self {
        Self { ctx }
    }

    /// Get the underlying context
    pub fn context(&mut self) -> &mut TextMlirContext {
        self.ctx
    }

    /// Emit a constant value
    pub fn emit_constant(&mut self, constant: &Constant, ty: &Type) -> Result<String> {
        match constant {
            Constant::Int(n) => {
                let code = arith::emit_constant_int(self.ctx, *n, ty)?;
                self.ctx.emit(&code);
                Ok(format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1))
            }
            Constant::Float(f) => {
                let code = arith::emit_constant_float(self.ctx, *f, ty)?;
                self.ctx.emit(&code);
                Ok(format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1))
            }
            Constant::Bool(b) => {
                let code = arith::emit_constant_bool(self.ctx, *b)?;
                self.ctx.emit(&code);
                Ok(format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1))
            }
            Constant::String(s) => {
                // String constants need special handling
                let ssa = self.ctx.fresh_ssa();
                let code = format!(
                    "{}{} = arith.constant \"{}\" : !llvm.ptr\n",
                    self.ctx.indent_str(),
                    ssa,
                    s.escape_default()
                );
                self.ctx.emit(&code);
                Ok(ssa)
            }
            Constant::Unit => {
                // Unit is represented as empty tuple
                let ssa = self.ctx.fresh_ssa();
                let code = format!(
                    "{}// {} = unit\n",
                    self.ctx.indent_str(),
                    ssa
                );
                self.ctx.emit(&code);
                Ok(ssa)
            }
        }
    }

    /// Emit an operand (constant or local reference)
    pub fn emit_operand(&mut self, op: &Operand) -> Result<String> {
        match op {
            Operand::Const(c) => {
                let ty = self.infer_constant_type(c);
                self.emit_constant(c, &ty)
            }
            Operand::Local(local) => self.ctx.get_ssa(local),
        }
    }

    /// Infer the type of a constant
    fn infer_constant_type(&self, c: &Constant) -> Type {
        match c {
            Constant::Int(_) => Type::Prim(goth_ast::types::PrimType::I64),
            Constant::Float(_) => Type::Prim(goth_ast::types::PrimType::F64),
            Constant::Bool(_) => Type::Prim(goth_ast::types::PrimType::Bool),
            Constant::String(_) => Type::Prim(goth_ast::types::PrimType::String),
            Constant::Unit => Type::Tuple(vec![]),
        }
    }

    /// Emit a binary operation
    pub fn emit_binop(
        &mut self,
        op: &BinOp,
        lhs: &str,
        rhs: &str,
        ty: &Type,
    ) -> Result<String> {
        let code = arith::emit_binop(self.ctx, op, lhs, rhs, ty)?;
        self.ctx.emit(&code);
        // Return the SSA value that was just created
        Ok(format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1))
    }

    // ========== SCF (Structured Control Flow) Operations ==========

    /// Create an scf.if builder for structured conditionals
    pub fn if_builder(&mut self, result_types: Vec<Type>) -> scf::IfBuilder<'_> {
        scf::IfBuilder::new(self.ctx, result_types)
    }

    /// Create an scf.for builder for counted loops
    pub fn for_builder(&mut self) -> scf::ForBuilder<'_> {
        scf::ForBuilder::new(self.ctx)
    }

    /// Emit a simple scf.if with then/else values (no region building needed)
    pub fn emit_scf_if(
        &mut self,
        condition: &str,
        then_value: &str,
        else_value: &str,
        result_type: &Type,
    ) -> Result<String> {
        let code = scf::emit_if_complete(self.ctx, condition, then_value, else_value, result_type)?;
        self.ctx.emit(&code);
        Ok(format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1))
    }

    /// Emit scf.yield to return values from a region
    pub fn emit_scf_yield(&mut self, values: &[String], types: &[Type]) -> Result<()> {
        let code = scf::emit_yield(self.ctx, values, types)?;
        self.ctx.emit(&code);
        Ok(())
    }

    /// Emit a unary operation
    pub fn emit_unary(&mut self, op: &UnaryOp, operand: &str, ty: &Type) -> Result<String> {
        let code = match op {
            UnaryOp::Neg => {
                if is_integer_type(ty) {
                    arith::emit_negi(self.ctx, operand, ty)?
                } else if is_float_type(ty) {
                    arith::emit_negf(self.ctx, operand, ty)?
                } else {
                    return Err(MlirError::UnsupportedOp(format!("Negation on {:?}", ty)));
                }
            }
            UnaryOp::Not => arith::emit_not(self.ctx, operand)?,
            UnaryOp::Sqrt | UnaryOp::Floor | UnaryOp::Ceil => {
                let op_name = match op {
                    UnaryOp::Sqrt => "math.sqrt",
                    UnaryOp::Floor => "math.floor",
                    UnaryOp::Ceil => "math.ceil",
                    _ => unreachable!(),
                };
                let ssa = self.ctx.fresh_ssa();
                let mlir_ty = type_to_mlir_string(ty)?;
                format!(
                    "{}{} = {} {} : {}\n",
                    self.ctx.indent_str(),
                    ssa,
                    op_name,
                    operand,
                    mlir_ty
                )
            }
            UnaryOp::Sum => {
                goth_dialect::emit_reduce(self.ctx, operand, ReduceOp::Sum, ty)?
            }
            UnaryOp::Prod => {
                goth_dialect::emit_reduce(self.ctx, operand, ReduceOp::Prod, ty)?
            }
            _ => {
                return Err(MlirError::UnsupportedOp(format!("{:?}", op)));
            }
        };

        self.ctx.emit(&code);
        Ok(format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1))
    }

    /// Emit a statement
    pub fn emit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        let ssa = match &stmt.rhs {
            Rhs::Use(op) => {
                self.emit_operand(op)?
            }

            Rhs::Const(c) => {
                self.emit_constant(c, &stmt.ty)?
            }

            Rhs::BinOp(op, left, right) => {
                let lhs = self.emit_operand(left)?;
                let rhs = self.emit_operand(right)?;
                self.emit_binop(op, &lhs, &rhs, &stmt.ty)?
            }

            Rhs::UnaryOp(op, operand) => {
                let op_ssa = self.emit_operand(operand)?;
                self.emit_unary(op, &op_ssa, &stmt.ty)?
            }

            Rhs::Tuple(ops) => {
                let mut op_ssas = Vec::new();
                for op in ops {
                    op_ssas.push(self.emit_operand(op)?);
                }
                // Tuple creation - use builtin conversion for now
                let ssa = self.ctx.fresh_ssa();
                let ty_str = type_to_mlir_string(&stmt.ty)?;
                let code = format!(
                    "{}{} = builtin.unrealized_conversion_cast {} : ... -> {}\n",
                    self.ctx.indent_str(),
                    ssa,
                    op_ssas.join(", "),
                    ty_str
                );
                self.ctx.emit(&code);
                ssa
            }

            Rhs::TupleField(tup, idx) => {
                let tup_ssa = self.emit_operand(tup)?;
                let ssa = self.ctx.fresh_ssa();
                let ty_str = type_to_mlir_string(&stmt.ty)?;
                let code = format!(
                    "{}{} = builtin.unrealized_conversion_cast {}[{}] : ... -> {}\n",
                    self.ctx.indent_str(),
                    ssa,
                    tup_ssa,
                    idx,
                    ty_str
                );
                self.ctx.emit(&code);
                ssa
            }

            Rhs::Array(elems) => {
                let mut elem_ssas = Vec::new();
                for elem in elems {
                    elem_ssas.push(self.emit_operand(elem)?);
                }
                let code = tensor::emit_from_elements(self.ctx, &elem_ssas, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::ArrayFill { size, value } => {
                let _size_ssa = self.emit_operand(size)?;
                let value_ssa = self.emit_operand(value)?;
                // Use tensor.splat or similar for filled arrays
                let ty_str = crate::types::type_to_mlir_string(&stmt.ty)?;
                let code = format!("{}%arr = tensor.splat {} : {}\n",
                    self.ctx.indent_str(), value_ssa, ty_str);
                self.ctx.emit(&code);
                "%arr".to_string()
            }

            Rhs::Index(arr, idx) => {
                let arr_ssa = self.emit_operand(arr)?;
                let idx_ssa = self.emit_operand(idx)?;
                let code = tensor::emit_extract(self.ctx, &arr_ssa, &[idx_ssa], &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::Iota(n) => {
                let n_ssa = self.emit_operand(n)?;
                let code = goth_dialect::emit_iota(self.ctx, &n_ssa, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::Range(start, end) => {
                let start_ssa = self.emit_operand(start)?;
                let end_ssa = self.emit_operand(end)?;
                let code = goth_dialect::emit_range(self.ctx, &start_ssa, &end_ssa, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::TensorMap { tensor, func } => {
                let tensor_ssa = self.emit_operand(tensor)?;
                let func_ssa = self.emit_operand(func)?;
                let code = goth_dialect::emit_map(self.ctx, &tensor_ssa, &func_ssa, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::TensorFilter { tensor, pred } => {
                let tensor_ssa = self.emit_operand(tensor)?;
                let pred_ssa = self.emit_operand(pred)?;
                let code = goth_dialect::emit_filter(self.ctx, &tensor_ssa, &pred_ssa, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::TensorReduce { tensor, op } => {
                let tensor_ssa = self.emit_operand(tensor)?;
                let code = goth_dialect::emit_reduce(self.ctx, &tensor_ssa, *op, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::TensorZip { left, right } => {
                let left_ssa = self.emit_operand(left)?;
                let right_ssa = self.emit_operand(right)?;
                let code = goth_dialect::emit_zip(self.ctx, &left_ssa, &right_ssa, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::MakeClosure { func: func_name, captures } => {
                let mut cap_ssas = Vec::new();
                for cap in captures {
                    cap_ssas.push(self.emit_operand(cap)?);
                }
                let code = goth_dialect::emit_make_closure(self.ctx, func_name, &cap_ssas, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::ClosureCall { closure, args } => {
                let closure_ssa = self.emit_operand(closure)?;
                let mut arg_ssas = Vec::new();
                let mut arg_types = Vec::new();
                for arg in args {
                    arg_ssas.push(self.emit_operand(arg)?);
                    // TODO: Get proper types
                    arg_types.push(Type::Prim(goth_ast::types::PrimType::I64));
                }
                let code = func::emit_call_indirect(
                    self.ctx,
                    &closure_ssa,
                    &arg_ssas,
                    &arg_types,
                    &stmt.ty,
                )?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::Call { func: func_name, args, arg_tys } => {
                let mut arg_ssas = Vec::new();
                for arg in args {
                    arg_ssas.push(self.emit_operand(arg)?);
                }
                let code = func::emit_call(self.ctx, func_name, &arg_ssas, arg_tys, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::Prim { name, args } => {
                let mut arg_ssas = Vec::new();
                for arg in args {
                    arg_ssas.push(self.emit_operand(arg)?);
                }
                let code = goth_dialect::emit_prim(self.ctx, name, &arg_ssas, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::MakeVariant { tag, constructor, payload } => {
                let payload_ssa = payload.as_ref()
                    .map(|p| self.emit_operand(p))
                    .transpose()?;
                let code = goth_dialect::emit_make_variant(
                    self.ctx,
                    *tag,
                    constructor,
                    payload_ssa.as_deref(),
                    &stmt.ty,
                )?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::GetTag(variant) => {
                let variant_ssa = self.emit_operand(variant)?;
                let code = goth_dialect::emit_get_tag(self.ctx, &variant_ssa);
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::GetPayload(variant) => {
                let variant_ssa = self.emit_operand(variant)?;
                let code = goth_dialect::emit_get_payload(self.ctx, &variant_ssa, &stmt.ty)?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::Uncertain { value, uncertainty } => {
                let value_ssa = self.emit_operand(value)?;
                let uncertainty_ssa = self.emit_operand(uncertainty)?;
                let code = goth_dialect::emit_uncertain(
                    self.ctx,
                    &value_ssa,
                    &uncertainty_ssa,
                    &stmt.ty,
                )?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }

            Rhs::ContractCheck { predicate, message, is_precondition } => {
                let pred_ssa = self.emit_operand(predicate)?;
                let code = goth_dialect::emit_contract_check(
                    self.ctx,
                    &pred_ssa,
                    message,
                    *is_precondition,
                );
                self.ctx.emit(&code);
                // Contract checks don't produce a value
                self.ctx.fresh_ssa()
            }

            Rhs::Slice { array, start, end } => {
                let arr_ssa = self.emit_operand(array)?;
                let start_ssa = match start {
                    Some(s) => self.emit_operand(s)?,
                    None => {
                        // Default to 0
                        self.emit_constant(&Constant::Int(0), &Type::Prim(goth_ast::types::PrimType::I64))?
                    }
                };
                let end_ssa = match end {
                    Some(e) => self.emit_operand(e)?,
                    None => {
                        // Need to get array length
                        tensor::emit_dim(self.ctx, &arr_ssa, "0");
                        format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
                    }
                };

                // Compute size = end - start
                let size_ssa = self.ctx.fresh_ssa();
                self.ctx.emit(&format!(
                    "{}{} = arith.subi {}, {} : index\n",
                    self.ctx.indent_str(),
                    size_ssa,
                    end_ssa,
                    start_ssa
                ));

                let code = tensor::emit_extract_slice(
                    self.ctx,
                    &arr_ssa,
                    &[start_ssa],
                    &[size_ssa],
                    &["1".to_string()],
                    &stmt.ty,
                )?;
                self.ctx.emit(&code);
                format!("%{}", self.ctx.fresh_ssa().strip_prefix('%').unwrap_or("0").parse::<usize>().unwrap_or(0) - 1)
            }
        };

        // Register the SSA value for this local
        self.ctx.register_local(stmt.dest, ssa, stmt.ty.clone());

        Ok(())
    }

    /// Emit a block terminator
    pub fn emit_terminator(&mut self, term: &Terminator) -> Result<()> {
        match term {
            Terminator::Return(op) => {
                let ret_ssa = self.emit_operand(op)?;
                // Get the type from the local if possible (clone to release borrow)
                let ty = if let Operand::Local(local) = op {
                    self.ctx.get_local_type(local).cloned()
                } else {
                    None
                };

                let ty = ty.unwrap_or_else(|| {
                    match op {
                        Operand::Const(c) => self.infer_constant_type(c),
                        Operand::Local(_) => Type::Prim(goth_ast::types::PrimType::I64), // Default
                    }
                });
                let code = func::emit_return(self.ctx, &ret_ssa, &ty)?;
                self.ctx.emit(&code);
            }

            Terminator::Goto(block_id) => {
                let code = cf::emit_br(self.ctx, *block_id);
                self.ctx.emit(&code);
            }

            Terminator::If { cond, then_block, else_block } => {
                let cond_ssa = self.emit_operand(cond)?;
                let code = cf::emit_cond_br(self.ctx, &cond_ssa, *then_block, *else_block);
                self.ctx.emit(&code);
            }

            Terminator::Switch { scrutinee, cases, default } => {
                let scrut_ssa = self.emit_operand(scrutinee)?;
                let int_cases: Vec<(i64, BlockId)> = cases.iter()
                    .filter_map(|(c, b)| {
                        if let Constant::Int(n) = c {
                            Some((*n, *b))
                        } else {
                            None
                        }
                    })
                    .collect();
                let code = cf::emit_switch(self.ctx, &scrut_ssa, &int_cases, *default);
                self.ctx.emit(&code);
            }

            Terminator::Unreachable => {
                self.ctx.emit_line("// unreachable");
            }
        }

        Ok(())
    }

    /// Emit a complete block
    pub fn emit_block(&mut self, block: &Block, label: Option<&str>) -> Result<()> {
        // Emit label if provided
        if let Some(lbl) = label {
            func::emit_block_label(self.ctx, lbl);
        }

        // Emit statements
        for stmt in &block.stmts {
            self.emit_stmt(stmt)?;
        }

        // Emit terminator
        self.emit_terminator(&block.term)?;

        Ok(())
    }

    /// Emit a complete function
    pub fn emit_function(&mut self, func: &Function) -> Result<()> {
        // Start function
        func::emit_function_header(self.ctx, func)?;

        // Emit entry block
        if func.blocks.is_empty() {
            self.emit_block(&func.body, None)?;
        } else {
            self.emit_block(&func.body, Some("entry"))?;

            // Emit additional blocks
            for (block_id, block) in &func.blocks {
                let label = format!("bb{}", block_id.0);
                self.emit_block(block, Some(&label))?;
            }
        }

        // End function
        func::emit_function_footer(self.ctx);

        Ok(())
    }

    /// Emit a complete program
    pub fn emit_program(&mut self, program: &Program) -> Result<()> {
        self.ctx.emit_line("module {");
        self.ctx.push_indent();

        for func in &program.functions {
            self.emit_function(func)?;
            self.ctx.emit("\n");
        }

        self.ctx.pop_indent();
        self.ctx.emit_line("}");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;

    #[test]
    fn test_emit_simple_function() {
        let mut ctx = TextMlirContext::new();
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
                            Operand::Local(LocalId::new(0)),
                            Operand::Const(Constant::Int(1)),
                        ),
                    },
                ],
                term: Terminator::Return(Operand::Local(LocalId::new(1))),
            },
            blocks: vec![],
            is_closure: false,
        };

        let mut builder = MlirBuilder::new(&mut ctx);
        builder.emit_function(&func).unwrap();

        let output = ctx.into_output();
        assert!(output.contains("func.func @add_one"));
        assert!(output.contains("arith.addi"));
        assert!(output.contains("func.return"));
    }

    #[test]
    fn test_emit_program() {
        let mut ctx = TextMlirContext::new();
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

        let mut builder = MlirBuilder::new(&mut ctx);
        builder.emit_program(&program).unwrap();

        let output = ctx.into_output();
        assert!(output.contains("module {"));
        assert!(output.contains("func.func @main"));
        assert!(output.contains("arith.constant 42"));
        assert!(output.contains("}"));
    }
}
