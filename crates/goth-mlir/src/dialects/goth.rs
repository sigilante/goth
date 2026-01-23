//! Custom Goth dialect for domain-specific MLIR operations
//!
//! The Goth dialect provides operations for Goth-specific semantics:
//! - goth.iota: Generate sequence [0, 1, ..., n-1]
//! - goth.range: Generate sequence [start, ..., end-1]
//! - goth.map: Apply function elementwise to tensor
//! - goth.filter: Filter tensor elements by predicate
//! - goth.reduce_{sum,prod,min,max}: Reduction operations
//! - goth.zip: Zip two tensors together
//!
//! These operations will be lowered to linalg or other standard dialects
//! in a later pass.

use goth_ast::types::Type;
use goth_mir::mir::ReduceOp;
use crate::context::TextMlirContext;
use crate::types::type_to_mlir_string;
use crate::error::Result;

/// Emit iota operation: generate [0, 1, 2, ..., n-1]
pub fn emit_iota(
    ctx: &mut TextMlirContext,
    size: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.iota {} : {}\n",
        ctx.indent_str(),
        ssa,
        size,
        ty_str
    ))
}

/// Emit range operation: generate [start, start+1, ..., end-1]
pub fn emit_range(
    ctx: &mut TextMlirContext,
    start: &str,
    end: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.range {}, {} : {}\n",
        ctx.indent_str(),
        ssa,
        start,
        end,
        ty_str
    ))
}

/// Emit map operation: apply function to each element
pub fn emit_map(
    ctx: &mut TextMlirContext,
    tensor: &str,
    func: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.map {}, {} : {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        func,
        ty_str
    ))
}

/// Emit filter operation: filter elements by predicate
pub fn emit_filter(
    ctx: &mut TextMlirContext,
    tensor: &str,
    pred: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.filter {}, {} : {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        pred,
        ty_str
    ))
}

/// Emit reduce operation
pub fn emit_reduce(
    ctx: &mut TextMlirContext,
    tensor: &str,
    op: ReduceOp,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    let op_name = match op {
        ReduceOp::Sum => "reduce_sum",
        ReduceOp::Prod => "reduce_prod",
        ReduceOp::Min => "reduce_min",
        ReduceOp::Max => "reduce_max",
    };

    Ok(format!(
        "{}{} = goth.{} {} : {}\n",
        ctx.indent_str(),
        ssa,
        op_name,
        tensor,
        ty_str
    ))
}

/// Emit zip operation: zip two tensors together
pub fn emit_zip(
    ctx: &mut TextMlirContext,
    left: &str,
    right: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.zip {}, {} : {}\n",
        ctx.indent_str(),
        ssa,
        left,
        right,
        ty_str
    ))
}

/// Emit variant construction
pub fn emit_make_variant(
    ctx: &mut TextMlirContext,
    tag: u32,
    constructor: &str,
    payload: Option<&str>,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    let payload_str = payload.map(|p| format!(", {}", p)).unwrap_or_default();

    Ok(format!(
        "{}{} = goth.make_variant {} \"{}\"{}  : {}\n",
        ctx.indent_str(),
        ssa,
        tag,
        constructor,
        payload_str,
        ty_str
    ))
}

/// Emit get_tag operation
pub fn emit_get_tag(
    ctx: &mut TextMlirContext,
    variant: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = goth.get_tag {} : i32\n",
        ctx.indent_str(),
        ssa,
        variant
    )
}

/// Emit get_payload operation
pub fn emit_get_payload(
    ctx: &mut TextMlirContext,
    variant: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.get_payload {} : {}\n",
        ctx.indent_str(),
        ssa,
        variant,
        ty_str
    ))
}

/// Emit uncertain value: value ± uncertainty
pub fn emit_uncertain(
    ctx: &mut TextMlirContext,
    value: &str,
    uncertainty: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.uncertain {}, {} : {}\n",
        ctx.indent_str(),
        ssa,
        value,
        uncertainty,
        ty_str
    ))
}

/// Emit primitive operation (built-in function)
pub fn emit_prim(
    ctx: &mut TextMlirContext,
    name: &str,
    args: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = goth.{} {} : {}\n",
        ctx.indent_str(),
        ssa,
        name,
        args.join(", "),
        ty_str
    ))
}

/// Emit contract check (for preconditions/postconditions)
pub fn emit_contract_check(
    ctx: &mut TextMlirContext,
    predicate: &str,
    message: &str,
    is_precondition: bool,
) -> String {
    let kind = if is_precondition { "pre" } else { "post" };

    format!(
        "{}goth.contract_check_{} {}, \"{}\"\n",
        ctx.indent_str(),
        kind,
        predicate,
        message.escape_default()
    )
}

/// Emit closure creation
pub fn emit_make_closure(
    ctx: &mut TextMlirContext,
    func_name: &str,
    captures: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    let captures_str = if captures.is_empty() {
        String::new()
    } else {
        format!(" [{}]", captures.join(", "))
    };

    Ok(format!(
        "{}{} = goth.make_closure @{}{} : {}\n",
        ctx.indent_str(),
        ssa,
        func_name,
        captures_str,
        ty_str
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_ast::shape::{Shape, Dim};

    fn tensor_i64() -> Type {
        Type::Tensor(
            Shape(vec![Dim::Var("n".into())]),
            Box::new(Type::Prim(PrimType::I64)),
        )
    }

    #[test]
    fn test_emit_iota() {
        let mut ctx = TextMlirContext::new();
        let code = emit_iota(&mut ctx, "%n", &tensor_i64()).unwrap();
        assert!(code.contains("goth.iota %n"));
    }

    #[test]
    fn test_emit_range() {
        let mut ctx = TextMlirContext::new();
        let code = emit_range(&mut ctx, "%start", "%end", &tensor_i64()).unwrap();
        assert!(code.contains("goth.range %start, %end"));
    }

    #[test]
    fn test_emit_map() {
        let mut ctx = TextMlirContext::new();
        let code = emit_map(&mut ctx, "%tensor", "%func", &tensor_i64()).unwrap();
        assert!(code.contains("goth.map %tensor, %func"));
    }

    #[test]
    fn test_emit_filter() {
        let mut ctx = TextMlirContext::new();
        let code = emit_filter(&mut ctx, "%tensor", "%pred", &tensor_i64()).unwrap();
        assert!(code.contains("goth.filter %tensor, %pred"));
    }

    #[test]
    fn test_emit_reduce() {
        let mut ctx = TextMlirContext::new();

        let code = emit_reduce(&mut ctx, "%tensor", ReduceOp::Sum, &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("goth.reduce_sum"));

        let code = emit_reduce(&mut ctx, "%tensor", ReduceOp::Prod, &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("goth.reduce_prod"));

        let code = emit_reduce(&mut ctx, "%tensor", ReduceOp::Min, &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("goth.reduce_min"));

        let code = emit_reduce(&mut ctx, "%tensor", ReduceOp::Max, &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("goth.reduce_max"));
    }

    #[test]
    fn test_emit_zip() {
        let mut ctx = TextMlirContext::new();
        let code = emit_zip(&mut ctx, "%a", "%b", &tensor_i64()).unwrap();
        assert!(code.contains("goth.zip %a, %b"));
    }

    #[test]
    fn test_emit_make_closure() {
        let mut ctx = TextMlirContext::new();
        let fn_type = Type::func(Type::Prim(PrimType::I64), Type::Prim(PrimType::I64));
        let code = emit_make_closure(
            &mut ctx,
            "lambda_0",
            &["%x".into(), "%y".into()],
            &fn_type,
        ).unwrap();
        assert!(code.contains("goth.make_closure @lambda_0"));
        assert!(code.contains("[%x, %y]"));
    }

    #[test]
    fn test_emit_prim() {
        let mut ctx = TextMlirContext::new();
        let code = emit_prim(
            &mut ctx,
            "print",
            &["%msg".into()],
            &Type::Tuple(vec![]),
        ).unwrap();
        assert!(code.contains("goth.print %msg"));
    }
}
