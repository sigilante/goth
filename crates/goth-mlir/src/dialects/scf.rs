//! SCF (Structured Control Flow) dialect operations for Goth MLIR emission
//!
//! The scf dialect provides structured control flow constructs that are
//! easier to analyze and optimize than unstructured cf operations:
//!
//! - scf.if: Conditional execution with then/else regions
//! - scf.for: Counted loop with induction variable
//! - scf.while: General loop with condition
//! - scf.yield: Return values from a region
//! - scf.condition: Specify condition for while loop continuation
//!
//! These operations maintain SSA form through region-based value passing,
//! making them more amenable to optimization passes.

use goth_ast::types::Type;
use crate::context::TextMlirContext;
use crate::types::type_to_mlir_string;
use crate::error::Result;

/// Emit an scf.if operation (conditional with optional results)
///
/// The if operation returns values from both branches via scf.yield.
///
/// Example MLIR:
/// ```mlir
/// %result = scf.if %cond -> (i64) {
///   %then_val = ...
///   scf.yield %then_val : i64
/// } else {
///   %else_val = ...
///   scf.yield %else_val : i64
/// }
/// ```
pub fn emit_if_start(
    ctx: &mut TextMlirContext,
    condition: &str,
    result_types: &[Type],
) -> Result<String> {
    let result_ssa = if result_types.is_empty() {
        String::new()
    } else {
        format!("{} = ", ctx.fresh_ssa())
    };

    let result_type_str = if result_types.is_empty() {
        String::new()
    } else {
        let types: Result<Vec<_>> = result_types.iter()
            .map(|t| type_to_mlir_string(t))
            .collect();
        format!(" -> ({})", types?.join(", "))
    };

    Ok(format!(
        "{}{}scf.if {}{} {{\n",
        ctx.indent_str(),
        result_ssa,
        condition,
        result_type_str
    ))
}

/// Emit the else branch of an scf.if
pub fn emit_else(ctx: &mut TextMlirContext) -> String {
    format!("{}}} else {{\n", ctx.indent_str())
}

/// Emit the closing brace of an scf.if
pub fn emit_if_end(ctx: &mut TextMlirContext) -> String {
    format!("{}}}\n", ctx.indent_str())
}

/// Emit a complete scf.if operation inline (for simple cases)
pub fn emit_if_complete(
    ctx: &mut TextMlirContext,
    condition: &str,
    then_value: &str,
    else_value: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = scf.if {} -> ({}) {{\n{}  scf.yield {} : {}\n{}}} else {{\n{}  scf.yield {} : {}\n{}}}\n",
        ctx.indent_str(),
        ssa,
        condition,
        ty_str,
        ctx.indent_str(),
        then_value,
        ty_str,
        ctx.indent_str(),
        ctx.indent_str(),
        else_value,
        ty_str,
        ctx.indent_str()
    ))
}

/// Emit an scf.for loop (counted loop)
///
/// Example MLIR:
/// ```mlir
/// %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i64) {
///   %new_acc = arith.addi %acc, %i : i64
///   scf.yield %new_acc : i64
/// }
/// ```
pub fn emit_for_start(
    ctx: &mut TextMlirContext,
    lower_bound: &str,
    upper_bound: &str,
    step: &str,
    iter_args: &[(String, String, Type)], // (name, init_value, type)
) -> Result<String> {
    let induction_var = ctx.fresh_ssa();

    let iter_args_str = if iter_args.is_empty() {
        String::new()
    } else {
        let args: Result<Vec<_>> = iter_args.iter()
            .map(|(name, init, ty)| {
                let ty_str = type_to_mlir_string(ty)?;
                Ok(format!("{} = {}", name, init))
            })
            .collect();
        format!(" iter_args({})", args?.join(", "))
    };

    let result_type_str = if iter_args.is_empty() {
        String::new()
    } else {
        let types: Result<Vec<_>> = iter_args.iter()
            .map(|(_, _, ty)| type_to_mlir_string(ty))
            .collect();
        format!(" -> ({})", types?.join(", "))
    };

    let result_ssa = if iter_args.is_empty() {
        String::new()
    } else {
        format!("{} = ", ctx.fresh_ssa())
    };

    Ok(format!(
        "{}{}scf.for {} = {} to {} step {}{}{} {{\n",
        ctx.indent_str(),
        result_ssa,
        induction_var,
        lower_bound,
        upper_bound,
        step,
        iter_args_str,
        result_type_str
    ))
}

/// Emit an scf.for loop end
pub fn emit_for_end(ctx: &mut TextMlirContext) -> String {
    format!("{}}}\n", ctx.indent_str())
}

/// Emit an scf.while loop
///
/// Example MLIR:
/// ```mlir
/// %result = scf.while (%arg0 = %init) : (i64) -> (i64) {
///   %cond = arith.cmpi slt, %arg0, %limit : i64
///   scf.condition(%cond) %arg0 : i64
/// } do {
/// ^bb0(%arg1: i64):
///   %next = arith.addi %arg1, %c1 : i64
///   scf.yield %next : i64
/// }
/// ```
pub fn emit_while_start(
    ctx: &mut TextMlirContext,
    iter_args: &[(String, String, Type)], // (name, init_value, type)
    result_types: &[Type],
) -> Result<String> {
    let iter_args_str = if iter_args.is_empty() {
        String::new()
    } else {
        let args: Vec<String> = iter_args.iter()
            .map(|(name, init, _)| format!("{} = {}", name, init))
            .collect();
        format!("({})", args.join(", "))
    };

    let input_types_str = if iter_args.is_empty() {
        "()".to_string()
    } else {
        let types: Result<Vec<_>> = iter_args.iter()
            .map(|(_, _, ty)| type_to_mlir_string(ty))
            .collect();
        format!("({})", types?.join(", "))
    };

    let result_types_str = if result_types.is_empty() {
        "()".to_string()
    } else {
        let types: Result<Vec<_>> = result_types.iter()
            .map(|ty| type_to_mlir_string(ty))
            .collect();
        format!("({})", types?.join(", "))
    };

    let result_ssa = if result_types.is_empty() {
        String::new()
    } else {
        format!("{} = ", ctx.fresh_ssa())
    };

    Ok(format!(
        "{}{}scf.while {} : {} -> {} {{\n",
        ctx.indent_str(),
        result_ssa,
        iter_args_str,
        input_types_str,
        result_types_str
    ))
}

/// Emit the do block of an scf.while
pub fn emit_while_do(
    ctx: &mut TextMlirContext,
    block_args: &[(String, Type)],
) -> Result<String> {
    let args_str = if block_args.is_empty() {
        String::new()
    } else {
        let args: Result<Vec<_>> = block_args.iter()
            .map(|(name, ty)| {
                let ty_str = type_to_mlir_string(ty)?;
                Ok(format!("{}: {}", name, ty_str))
            })
            .collect();
        format!("^bb0({}):\n", args?.join(", "))
    };

    Ok(format!(
        "{}}} do {{\n{}{}",
        ctx.indent_str(),
        ctx.indent_str(),
        args_str
    ))
}

/// Emit scf.while end
pub fn emit_while_end(ctx: &mut TextMlirContext) -> String {
    format!("{}}}\n", ctx.indent_str())
}

/// Emit scf.yield (return values from a region)
///
/// Example MLIR:
/// ```mlir
/// scf.yield %value : i64
/// ```
pub fn emit_yield(
    ctx: &mut TextMlirContext,
    values: &[String],
    types: &[Type],
) -> Result<String> {
    if values.is_empty() {
        return Ok(format!("{}scf.yield\n", ctx.indent_str()));
    }

    let type_strs: Result<Vec<_>> = types.iter()
        .map(|t| type_to_mlir_string(t))
        .collect();

    Ok(format!(
        "{}scf.yield {} : {}\n",
        ctx.indent_str(),
        values.join(", "),
        type_strs?.join(", ")
    ))
}

/// Emit scf.condition (for while loop condition)
///
/// Example MLIR:
/// ```mlir
/// scf.condition(%cond) %arg0 : i64
/// ```
pub fn emit_condition(
    ctx: &mut TextMlirContext,
    condition: &str,
    values: &[String],
    types: &[Type],
) -> Result<String> {
    if values.is_empty() {
        return Ok(format!(
            "{}scf.condition({})\n",
            ctx.indent_str(),
            condition
        ));
    }

    let type_strs: Result<Vec<_>> = types.iter()
        .map(|t| type_to_mlir_string(t))
        .collect();

    Ok(format!(
        "{}scf.condition({}) {} : {}\n",
        ctx.indent_str(),
        condition,
        values.join(", "),
        type_strs?.join(", ")
    ))
}

/// Emit scf.parallel (parallel loop for multi-dimensional iteration)
///
/// Example MLIR:
/// ```mlir
/// scf.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1) {
///   ...
///   scf.yield
/// }
/// ```
pub fn emit_parallel_start(
    ctx: &mut TextMlirContext,
    induction_vars: &[String],
    lower_bounds: &[String],
    upper_bounds: &[String],
    steps: &[String],
) -> String {
    format!(
        "{}scf.parallel ({}) = ({}) to ({}) step ({}) {{\n",
        ctx.indent_str(),
        induction_vars.join(", "),
        lower_bounds.join(", "),
        upper_bounds.join(", "),
        steps.join(", ")
    )
}

/// Emit scf.parallel end
pub fn emit_parallel_end(ctx: &mut TextMlirContext) -> String {
    format!("{}}}\n", ctx.indent_str())
}

/// Emit scf.reduce (reduction in parallel loop)
pub fn emit_reduce(
    ctx: &mut TextMlirContext,
    operand: &str,
    result_type: &Type,
) -> Result<String> {
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}scf.reduce({}) : {} {{\n",
        ctx.indent_str(),
        operand,
        ty_str
    ))
}

/// Emit scf.reduce.return
pub fn emit_reduce_return(
    ctx: &mut TextMlirContext,
    value: &str,
    result_type: &Type,
) -> Result<String> {
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}scf.reduce.return {} : {}\n{}}}\n",
        ctx.indent_str(),
        value,
        ty_str,
        ctx.indent_str()
    ))
}

/// Helper struct for building scf.if operations
pub struct IfBuilder<'a> {
    /// The MLIR context for emitting code
    pub ctx: &'a mut TextMlirContext,
    result_types: Vec<Type>,
}

impl<'a> IfBuilder<'a> {
    /// Create a new if builder
    pub fn new(ctx: &'a mut TextMlirContext, result_types: Vec<Type>) -> Self {
        Self { ctx, result_types }
    }

    /// Start the if operation
    pub fn begin(&mut self, condition: &str) -> Result<()> {
        let code = emit_if_start(self.ctx, condition, &self.result_types)?;
        self.ctx.emit(&code);
        self.ctx.push_indent();
        Ok(())
    }

    /// Transition to else branch
    pub fn else_branch(&mut self) {
        self.ctx.pop_indent();
        let code = emit_else(self.ctx);
        self.ctx.emit(&code);
        self.ctx.push_indent();
    }

    /// End the if operation
    pub fn end(&mut self) {
        self.ctx.pop_indent();
        let code = emit_if_end(self.ctx);
        self.ctx.emit(&code);
    }

    /// Emit yield in current region
    pub fn yield_values(&mut self, values: &[String]) -> Result<()> {
        let code = emit_yield(self.ctx, values, &self.result_types)?;
        self.ctx.emit(&code);
        Ok(())
    }
}

/// Helper struct for building scf.for operations
pub struct ForBuilder<'a> {
    /// The MLIR context for emitting code
    pub ctx: &'a mut TextMlirContext,
    iter_arg_types: Vec<Type>,
}

impl<'a> ForBuilder<'a> {
    /// Create a new for builder
    pub fn new(ctx: &'a mut TextMlirContext) -> Self {
        Self {
            ctx,
            iter_arg_types: Vec::new(),
        }
    }

    /// Start the for loop
    pub fn begin(
        &mut self,
        lower_bound: &str,
        upper_bound: &str,
        step: &str,
        iter_args: &[(String, String, Type)],
    ) -> Result<()> {
        self.iter_arg_types = iter_args.iter().map(|(_, _, ty)| ty.clone()).collect();
        let code = emit_for_start(self.ctx, lower_bound, upper_bound, step, iter_args)?;
        self.ctx.emit(&code);
        self.ctx.push_indent();
        Ok(())
    }

    /// End the for loop
    pub fn end(&mut self) {
        self.ctx.pop_indent();
        let code = emit_for_end(self.ctx);
        self.ctx.emit(&code);
    }

    /// Emit yield in loop body
    pub fn yield_values(&mut self, values: &[String]) -> Result<()> {
        let code = emit_yield(self.ctx, values, &self.iter_arg_types)?;
        self.ctx.emit(&code);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;

    #[test]
    fn test_emit_if_start() {
        let mut ctx = TextMlirContext::new();
        let code = emit_if_start(&mut ctx, "%cond", &[Type::Prim(PrimType::I64)]).unwrap();
        assert!(code.contains("scf.if %cond"));
        assert!(code.contains("-> (i64)"));
    }

    #[test]
    fn test_emit_if_complete() {
        let mut ctx = TextMlirContext::new();
        let code = emit_if_complete(
            &mut ctx,
            "%cond",
            "%then_val",
            "%else_val",
            &Type::Prim(PrimType::I64),
        ).unwrap();
        assert!(code.contains("scf.if %cond"));
        assert!(code.contains("scf.yield %then_val"));
        assert!(code.contains("scf.yield %else_val"));
        assert!(code.contains("else"));
    }

    #[test]
    fn test_emit_for_start() {
        let mut ctx = TextMlirContext::new();
        let code = emit_for_start(
            &mut ctx,
            "%lb",
            "%ub",
            "%step",
            &[("%acc".to_string(), "%init".to_string(), Type::Prim(PrimType::I64))],
        ).unwrap();
        assert!(code.contains("scf.for"));
        assert!(code.contains("%lb to %ub step %step"));
        assert!(code.contains("iter_args"));
    }

    #[test]
    fn test_emit_while_start() {
        let mut ctx = TextMlirContext::new();
        let code = emit_while_start(
            &mut ctx,
            &[("%arg".to_string(), "%init".to_string(), Type::Prim(PrimType::I64))],
            &[Type::Prim(PrimType::I64)],
        ).unwrap();
        assert!(code.contains("scf.while"));
        assert!(code.contains("(i64) -> (i64)"));
    }

    #[test]
    fn test_emit_yield() {
        let mut ctx = TextMlirContext::new();
        let code = emit_yield(&mut ctx, &["%val".to_string()], &[Type::Prim(PrimType::I64)]).unwrap();
        assert!(code.contains("scf.yield %val"));
        assert!(code.contains("i64"));
    }

    #[test]
    fn test_emit_condition() {
        let mut ctx = TextMlirContext::new();
        let code = emit_condition(
            &mut ctx,
            "%cond",
            &["%arg".to_string()],
            &[Type::Prim(PrimType::I64)],
        ).unwrap();
        assert!(code.contains("scf.condition(%cond)"));
        assert!(code.contains("%arg"));
    }

    #[test]
    fn test_emit_parallel() {
        let mut ctx = TextMlirContext::new();
        let code = emit_parallel_start(
            &mut ctx,
            &["%i".to_string(), "%j".to_string()],
            &["%0".to_string(), "%0".to_string()],
            &["%n".to_string(), "%m".to_string()],
            &["%1".to_string(), "%1".to_string()],
        );
        assert!(code.contains("scf.parallel"));
        assert!(code.contains("(%i, %j)"));
    }

    #[test]
    fn test_if_builder() {
        let mut ctx = TextMlirContext::new();

        {
            let mut builder = IfBuilder::new(&mut ctx, vec![Type::Prim(PrimType::I64)]);
            builder.begin("%cond").unwrap();
            // Then branch - emit directly through builder's context
            builder.ctx.emit("  // then branch\n");
            builder.yield_values(&["%then_val".to_string()]).unwrap();
            builder.else_branch();
            // Else branch
            builder.ctx.emit("  // else branch\n");
            builder.yield_values(&["%else_val".to_string()]).unwrap();
            builder.end();
        }

        let output = ctx.into_output();
        assert!(output.contains("scf.if %cond"));
        assert!(output.contains("else"));
        assert!(output.contains("scf.yield"));
    }

    #[test]
    fn test_for_builder() {
        let mut ctx = TextMlirContext::new();

        {
            let mut builder = ForBuilder::new(&mut ctx);
            builder.begin(
                "%c0",
                "%n",
                "%c1",
                &[("%acc".to_string(), "%init".to_string(), Type::Prim(PrimType::I64))],
            ).unwrap();
            // Emit through builder's context
            builder.ctx.emit("  // loop body\n");
            builder.yield_values(&["%new_acc".to_string()]).unwrap();
            builder.end();
        }

        let output = ctx.into_output();
        assert!(output.contains("scf.for"));
        assert!(output.contains("iter_args"));
        assert!(output.contains("scf.yield"));
    }
}
