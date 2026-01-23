//! Func dialect operations for Goth MLIR emission
//!
//! The func dialect provides function-related operations:
//! - func.func: Function definition
//! - func.return: Return from function
//! - func.call: Direct function call
//! - func.call_indirect: Indirect call through function pointer

use goth_ast::types::Type;
use goth_mir::mir::{Function, Block, Terminator, LocalId};
use crate::context::TextMlirContext;
use crate::types::type_to_mlir_string;
use crate::error::{MlirError, Result};

/// Emit a function signature (without body)
pub fn emit_function_signature(func: &Function) -> Result<String> {
    let param_types: Result<Vec<_>> = func.params.iter()
        .map(|ty| type_to_mlir_string(ty))
        .collect();
    let param_types = param_types?;

    let ret_type = type_to_mlir_string(&func.ret_ty)?;

    let params_str = param_types.iter()
        .enumerate()
        .map(|(i, ty)| format!("%arg{}: {}", i, ty))
        .collect::<Vec<_>>()
        .join(", ");

    Ok(format!("func.func @{}({}) -> {}", func.name, params_str, ret_type))
}

/// Emit a function header (opening line)
pub fn emit_function_header(
    ctx: &mut TextMlirContext,
    func: &Function,
) -> Result<()> {
    let param_types: Result<Vec<_>> = func.params.iter()
        .map(|ty| type_to_mlir_string(ty))
        .collect();
    let param_types = param_types?;

    let ret_type = type_to_mlir_string(&func.ret_ty)?;

    // Emit function header with parameters
    let mut header = format!("func.func @{}(", func.name);

    for (i, ty) in param_types.iter().enumerate() {
        if i > 0 {
            header.push_str(", ");
        }
        let param_ssa = ctx.fresh_ssa();
        header.push_str(&format!("{}: {}", param_ssa, ty));

        // Register parameter as local
        let local_id = LocalId::new(i as u32);
        ctx.register_local(local_id, param_ssa.clone(), func.params[i].clone());
    }

    header.push_str(&format!(") -> {} {{\n", ret_type));
    ctx.emit(&header);
    ctx.push_indent();

    Ok(())
}

/// Emit a function footer (closing brace)
pub fn emit_function_footer(ctx: &mut TextMlirContext) {
    ctx.pop_indent();
    ctx.emit_line("}");
}

/// Emit a return statement
pub fn emit_return(ctx: &mut TextMlirContext, value: &str, ty: &Type) -> Result<String> {
    let mlir_ty = type_to_mlir_string(ty)?;
    Ok(format!(
        "{}func.return {} : {}\n",
        ctx.indent_str(), value, mlir_ty
    ))
}

/// Emit a direct function call
pub fn emit_call(
    ctx: &mut TextMlirContext,
    func_name: &str,
    args: &[String],
    arg_types: &[Type],
    ret_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();

    let arg_types_str: Result<Vec<_>> = arg_types.iter()
        .map(|ty| type_to_mlir_string(ty))
        .collect();
    let arg_types_str = arg_types_str?;

    let ret_type_str = type_to_mlir_string(ret_type)?;

    Ok(format!(
        "{}{} = func.call @{}({}) : ({}) -> {}\n",
        ctx.indent_str(),
        ssa,
        func_name,
        args.join(", "),
        arg_types_str.join(", "),
        ret_type_str
    ))
}

/// Emit an indirect function call (through function pointer/closure)
pub fn emit_call_indirect(
    ctx: &mut TextMlirContext,
    closure: &str,
    args: &[String],
    arg_types: &[Type],
    ret_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();

    let arg_types_str: Result<Vec<_>> = arg_types.iter()
        .map(|ty| type_to_mlir_string(ty))
        .collect();
    let arg_types_str = arg_types_str?;

    let ret_type_str = type_to_mlir_string(ret_type)?;

    Ok(format!(
        "{}{} = func.call_indirect {}({}) : ({}) -> {}\n",
        ctx.indent_str(),
        ssa,
        closure,
        args.join(", "),
        arg_types_str.join(", "),
        ret_type_str
    ))
}

/// Emit a block label
pub fn emit_block_label(ctx: &mut TextMlirContext, label: &str) {
    // Block labels are at indent level - 1
    let saved_indent = ctx.indent_str();
    ctx.pop_indent();
    ctx.emit(&format!("{}^{}:\n", ctx.indent_str(), label));
    ctx.push_indent();
}

/// Emit a block label with block arguments
pub fn emit_block_label_with_args(
    ctx: &mut TextMlirContext,
    label: &str,
    args: &[(String, Type)],
) -> Result<()> {
    let args_str: Result<Vec<_>> = args.iter()
        .map(|(name, ty)| {
            let ty_str = type_to_mlir_string(ty)?;
            Ok(format!("{}: {}", name, ty_str))
        })
        .collect();
    let args_str = args_str?;

    ctx.pop_indent();
    ctx.emit(&format!("{}^{}({}):\n", ctx.indent_str(), label, args_str.join(", ")));
    ctx.push_indent();

    Ok(())
}

/// Builder for function emission
pub struct FunctionBuilder<'a> {
    ctx: &'a mut TextMlirContext,
    func: &'a Function,
    has_multiple_blocks: bool,
}

impl<'a> FunctionBuilder<'a> {
    /// Create a new function builder
    pub fn new(ctx: &'a mut TextMlirContext, func: &'a Function) -> Self {
        let has_multiple_blocks = !func.blocks.is_empty();
        Self {
            ctx,
            func,
            has_multiple_blocks,
        }
    }

    /// Start building the function
    pub fn begin(&mut self) -> Result<()> {
        emit_function_header(self.ctx, self.func)
    }

    /// Finish building the function
    pub fn end(&mut self) {
        emit_function_footer(self.ctx);
    }

    /// Get whether this function has multiple blocks
    pub fn has_multiple_blocks(&self) -> bool {
        self.has_multiple_blocks
    }

    /// Get the context
    pub fn context(&mut self) -> &mut TextMlirContext {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_mir::mir::{Block, Terminator, Operand, Constant};

    fn make_simple_function() -> Function {
        Function {
            name: "test".to_string(),
            params: vec![Type::Prim(PrimType::I64)],
            ret_ty: Type::Prim(PrimType::I64),
            body: Block::with_return(Operand::Local(LocalId::new(0))),
            blocks: vec![],
            is_closure: false,
        }
    }

    #[test]
    fn test_emit_function_signature() {
        let func = make_simple_function();
        let sig = emit_function_signature(&func).unwrap();
        assert!(sig.contains("func.func @test"));
        assert!(sig.contains("i64"));
    }

    #[test]
    fn test_emit_function_header() {
        let mut ctx = TextMlirContext::new();
        let func = make_simple_function();
        emit_function_header(&mut ctx, &func).unwrap();

        let output = ctx.output();
        assert!(output.contains("func.func @test"));
        assert!(output.contains("%0: i64"));
        assert!(output.contains("-> i64"));
    }

    #[test]
    fn test_emit_return() {
        let mut ctx = TextMlirContext::new();
        let code = emit_return(&mut ctx, "%0", &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("func.return %0"));
        assert!(code.contains("i64"));
    }

    #[test]
    fn test_emit_call() {
        let mut ctx = TextMlirContext::new();
        let code = emit_call(
            &mut ctx,
            "add",
            &["%0".to_string(), "%1".to_string()],
            &[Type::Prim(PrimType::I64), Type::Prim(PrimType::I64)],
            &Type::Prim(PrimType::I64),
        ).unwrap();

        assert!(code.contains("func.call @add"));
        assert!(code.contains("%0, %1"));
    }

    #[test]
    fn test_emit_call_indirect() {
        let mut ctx = TextMlirContext::new();
        let code = emit_call_indirect(
            &mut ctx,
            "%closure",
            &["%arg".to_string()],
            &[Type::Prim(PrimType::I64)],
            &Type::Prim(PrimType::I64),
        ).unwrap();

        assert!(code.contains("func.call_indirect %closure"));
    }

    #[test]
    fn test_function_builder() {
        let mut ctx = TextMlirContext::new();
        let func = make_simple_function();

        {
            let mut builder = FunctionBuilder::new(&mut ctx, &func);
            builder.begin().unwrap();
            // Would emit body here
            builder.end();
        }

        let output = ctx.into_output();
        assert!(output.contains("func.func @test"));
        assert!(output.contains("}"));
    }
}
