//! Tensor dialect operations for Goth MLIR emission
//!
//! The tensor dialect provides tensor operations:
//! - tensor.from_elements: Create tensor from scalar elements
//! - tensor.extract: Extract element from tensor
//! - tensor.insert: Insert element into tensor
//! - tensor.dim: Get dimension size
//! - tensor.rank: Get tensor rank
//! - tensor.reshape: Reshape tensor
//! - tensor.empty: Create uninitialized tensor
//! - tensor.generate: Create tensor with computed values

use goth_ast::types::Type;
use crate::context::TextMlirContext;
use crate::types::type_to_mlir_string;
use crate::error::Result;

/// Emit tensor creation from elements
pub fn emit_from_elements(
    ctx: &mut TextMlirContext,
    elements: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.from_elements {} : {}\n",
        ctx.indent_str(),
        ssa,
        elements.join(", "),
        ty_str
    ))
}

/// Emit tensor element extraction
pub fn emit_extract(
    ctx: &mut TextMlirContext,
    tensor: &str,
    indices: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.extract {}[{}] : {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        indices.join(", "),
        ty_str
    ))
}

/// Emit tensor element insertion
pub fn emit_insert(
    ctx: &mut TextMlirContext,
    value: &str,
    tensor: &str,
    indices: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.insert {} into {}[{}] : {}\n",
        ctx.indent_str(),
        ssa,
        value,
        tensor,
        indices.join(", "),
        ty_str
    ))
}

/// Emit tensor dimension query
pub fn emit_dim(
    ctx: &mut TextMlirContext,
    tensor: &str,
    dim_index: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = tensor.dim {}, {} : index\n",
        ctx.indent_str(),
        ssa,
        tensor,
        dim_index
    )
}

/// Emit tensor rank query
pub fn emit_rank(ctx: &mut TextMlirContext, tensor: &str) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = tensor.rank {} : index\n",
        ctx.indent_str(),
        ssa,
        tensor
    )
}

/// Emit empty tensor creation
pub fn emit_empty(
    ctx: &mut TextMlirContext,
    dims: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.empty({}) : {}\n",
        ctx.indent_str(),
        ssa,
        dims.join(", "),
        ty_str
    ))
}

/// Emit tensor reshape
pub fn emit_reshape(
    ctx: &mut TextMlirContext,
    tensor: &str,
    shape: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.reshape {} : {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        ty_str
    ))
}

/// Emit tensor concatenation
pub fn emit_concat(
    ctx: &mut TextMlirContext,
    tensors: &[String],
    dim: i64,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.concat dim({}) {} : {}\n",
        ctx.indent_str(),
        ssa,
        dim,
        tensors.join(", "),
        ty_str
    ))
}

/// Emit tensor slice extraction
pub fn emit_extract_slice(
    ctx: &mut TextMlirContext,
    tensor: &str,
    offsets: &[String],
    sizes: &[String],
    strides: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.extract_slice {}[{}][{}][{}] : {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        offsets.join(", "),
        sizes.join(", "),
        strides.join(", "),
        ty_str
    ))
}

/// Emit tensor slice insertion
pub fn emit_insert_slice(
    ctx: &mut TextMlirContext,
    source: &str,
    dest: &str,
    offsets: &[String],
    sizes: &[String],
    strides: &[String],
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.insert_slice {} into {}[{}][{}][{}] : {}\n",
        ctx.indent_str(),
        ssa,
        source,
        dest,
        offsets.join(", "),
        sizes.join(", "),
        strides.join(", "),
        ty_str
    ))
}

/// Emit tensor pad operation
pub fn emit_pad(
    ctx: &mut TextMlirContext,
    tensor: &str,
    low: &[String],
    high: &[String],
    pad_value: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;

    Ok(format!(
        "{}{} = tensor.pad {} low[{}] high[{}] {{ yield {} }} : {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        low.join(", "),
        high.join(", "),
        pad_value,
        ty_str
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_ast::shape::{Shape, Dim};

    fn tensor_i64_10() -> Type {
        Type::Tensor(
            Shape(vec![Dim::Const(10)]),
            Box::new(Type::Prim(PrimType::I64)),
        )
    }

    #[test]
    fn test_emit_from_elements() {
        let mut ctx = TextMlirContext::new();
        let code = emit_from_elements(
            &mut ctx,
            &["%0".into(), "%1".into(), "%2".into()],
            &tensor_i64_10(),
        ).unwrap();

        assert!(code.contains("tensor.from_elements"));
        assert!(code.contains("%0, %1, %2"));
    }

    #[test]
    fn test_emit_extract() {
        let mut ctx = TextMlirContext::new();
        let code = emit_extract(
            &mut ctx,
            "%tensor",
            &["%idx".into()],
            &Type::Prim(PrimType::I64),
        ).unwrap();

        assert!(code.contains("tensor.extract %tensor[%idx]"));
    }

    #[test]
    fn test_emit_insert() {
        let mut ctx = TextMlirContext::new();
        let code = emit_insert(
            &mut ctx,
            "%val",
            "%tensor",
            &["%idx".into()],
            &tensor_i64_10(),
        ).unwrap();

        assert!(code.contains("tensor.insert %val into %tensor[%idx]"));
    }

    #[test]
    fn test_emit_dim() {
        let mut ctx = TextMlirContext::new();
        let code = emit_dim(&mut ctx, "%tensor", "%0");
        assert!(code.contains("tensor.dim %tensor, %0"));
    }

    #[test]
    fn test_emit_empty() {
        let mut ctx = TextMlirContext::new();
        let code = emit_empty(&mut ctx, &["10".into()], &tensor_i64_10()).unwrap();
        assert!(code.contains("tensor.empty(10)"));
    }
}
