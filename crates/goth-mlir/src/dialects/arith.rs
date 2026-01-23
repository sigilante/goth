//! Arith dialect operations for Goth MLIR emission
//!
//! The arith dialect provides basic arithmetic operations:
//! - Integer operations: addi, subi, muli, divsi, remsi
//! - Float operations: addf, subf, mulf, divf, negf
//! - Comparison: cmpi, cmpf
//! - Constants: constant

use goth_ast::op::BinOp;
use goth_ast::types::Type;
use crate::context::TextMlirContext;
use crate::types::{type_to_mlir_string, is_integer_type, is_float_type};
use crate::error::{MlirError, Result};

/// Emit an arithmetic binary operation
///
/// Generates the appropriate arith dialect operation based on operand types.
pub fn emit_binop(
    ctx: &mut TextMlirContext,
    op: &BinOp,
    lhs: &str,
    rhs: &str,
    ty: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let mlir_ty = type_to_mlir_string(ty)?;

    let (op_name, result_ty) = if is_integer_type(ty) {
        (integer_binop_name(op)?, mlir_ty.clone())
    } else if is_float_type(ty) {
        (float_binop_name(op)?, mlir_ty.clone())
    } else {
        return Err(MlirError::UnsupportedType(format!(
            "Binary operation on type: {:?}", ty
        )));
    };

    // Comparison operations return i1 (boolean)
    let result_ty = if is_comparison(op) {
        "i1".to_string()
    } else {
        result_ty
    };

    Ok(format!(
        "{}{} = {} {}, {} : {}\n",
        ctx.indent_str(), ssa, op_name, lhs, rhs, result_ty
    ))
}

/// Get MLIR operation name for integer binary operation
fn integer_binop_name(op: &BinOp) -> Result<&'static str> {
    match op {
        BinOp::Add => Ok("arith.addi"),
        BinOp::Sub => Ok("arith.subi"),
        BinOp::Mul => Ok("arith.muli"),
        BinOp::Div => Ok("arith.divsi"),
        BinOp::Mod => Ok("arith.remsi"),
        BinOp::Lt => Ok("arith.cmpi slt,"),
        BinOp::Gt => Ok("arith.cmpi sgt,"),
        BinOp::Leq => Ok("arith.cmpi sle,"),
        BinOp::Geq => Ok("arith.cmpi sge,"),
        BinOp::Eq => Ok("arith.cmpi eq,"),
        BinOp::Neq => Ok("arith.cmpi ne,"),
        BinOp::And => Ok("arith.andi"),
        BinOp::Or => Ok("arith.ori"),
        _ => Err(MlirError::UnsupportedOp(format!("Integer op: {:?}", op))),
    }
}

/// Get MLIR operation name for floating-point binary operation
fn float_binop_name(op: &BinOp) -> Result<&'static str> {
    match op {
        BinOp::Add => Ok("arith.addf"),
        BinOp::Sub => Ok("arith.subf"),
        BinOp::Mul => Ok("arith.mulf"),
        BinOp::Div => Ok("arith.divf"),
        BinOp::Lt => Ok("arith.cmpf olt,"),
        BinOp::Gt => Ok("arith.cmpf ogt,"),
        BinOp::Leq => Ok("arith.cmpf ole,"),
        BinOp::Geq => Ok("arith.cmpf oge,"),
        BinOp::Eq => Ok("arith.cmpf oeq,"),
        BinOp::Neq => Ok("arith.cmpf one,"),
        _ => Err(MlirError::UnsupportedOp(format!("Float op: {:?}", op))),
    }
}

/// Check if an operation is a comparison
fn is_comparison(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Lt | BinOp::Gt | BinOp::Leq | BinOp::Geq | BinOp::Eq | BinOp::Neq
    )
}

/// Emit an integer constant
pub fn emit_constant_int(ctx: &mut TextMlirContext, value: i64, ty: &Type) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let mlir_ty = type_to_mlir_string(ty)?;

    Ok(format!(
        "{}{} = arith.constant {} : {}\n",
        ctx.indent_str(), ssa, value, mlir_ty
    ))
}

/// Emit a floating-point constant
pub fn emit_constant_float(ctx: &mut TextMlirContext, value: f64, ty: &Type) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let mlir_ty = type_to_mlir_string(ty)?;

    // Format float appropriately for MLIR
    let float_str = if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() { "inf".to_string() } else { "-inf".to_string() }
    } else if value.fract() == 0.0 {
        format!("{:.1}", value) // Ensure decimal point for whole numbers
    } else {
        format!("{}", value)
    };

    Ok(format!(
        "{}{} = arith.constant {} : {}\n",
        ctx.indent_str(), ssa, float_str, mlir_ty
    ))
}

/// Emit a boolean constant
pub fn emit_constant_bool(ctx: &mut TextMlirContext, value: bool) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let val_str = if value { "true" } else { "false" };

    Ok(format!(
        "{}{} = arith.constant {} : i1\n",
        ctx.indent_str(), ssa, val_str
    ))
}

/// Emit integer negation (0 - x)
pub fn emit_negi(ctx: &mut TextMlirContext, operand: &str, ty: &Type) -> Result<String> {
    let mlir_ty = type_to_mlir_string(ty)?;
    let zero_ssa = ctx.fresh_ssa();
    let result_ssa = ctx.fresh_ssa();

    Ok(format!(
        "{}{} = arith.constant 0 : {}\n{}{} = arith.subi {}, {} : {}\n",
        ctx.indent_str(), zero_ssa, mlir_ty,
        ctx.indent_str(), result_ssa, zero_ssa, operand, mlir_ty
    ))
}

/// Emit floating-point negation
pub fn emit_negf(ctx: &mut TextMlirContext, operand: &str, ty: &Type) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let mlir_ty = type_to_mlir_string(ty)?;

    Ok(format!(
        "{}{} = arith.negf {} : {}\n",
        ctx.indent_str(), ssa, operand, mlir_ty
    ))
}

/// Emit boolean NOT (xor with true)
pub fn emit_not(ctx: &mut TextMlirContext, operand: &str) -> Result<String> {
    let one_ssa = ctx.fresh_ssa();
    let result_ssa = ctx.fresh_ssa();

    Ok(format!(
        "{}{} = arith.constant true : i1\n{}{} = arith.xori {}, {} : i1\n",
        ctx.indent_str(), one_ssa,
        ctx.indent_str(), result_ssa, operand, one_ssa
    ))
}

/// Emit type cast/conversion
pub fn emit_cast(
    ctx: &mut TextMlirContext,
    operand: &str,
    from_ty: &Type,
    to_ty: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let from_mlir = type_to_mlir_string(from_ty)?;
    let to_mlir = type_to_mlir_string(to_ty)?;

    // Determine the appropriate cast operation
    let cast_op = match (is_integer_type(from_ty), is_float_type(from_ty),
                         is_integer_type(to_ty), is_float_type(to_ty)) {
        (true, false, false, true) => "arith.sitofp", // int -> float
        (false, true, true, false) => "arith.fptosi", // float -> int
        (true, false, true, false) => "arith.extsi",  // int -> wider int (or trunci)
        (false, true, false, true) => "arith.extf",   // float -> wider float
        _ => return Err(MlirError::UnsupportedOp(format!(
            "Cast from {:?} to {:?}", from_ty, to_ty
        ))),
    };

    Ok(format!(
        "{}{} = {} {} : {} to {}\n",
        ctx.indent_str(), ssa, cast_op, operand, from_mlir, to_mlir
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;

    #[test]
    fn test_emit_binop_int_add() {
        let mut ctx = TextMlirContext::new();
        let code = emit_binop(&mut ctx, &BinOp::Add, "%0", "%1", &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("arith.addi"));
        assert!(code.contains("i64"));
    }

    #[test]
    fn test_emit_binop_float_mul() {
        let mut ctx = TextMlirContext::new();
        let code = emit_binop(&mut ctx, &BinOp::Mul, "%0", "%1", &Type::Prim(PrimType::F64)).unwrap();
        assert!(code.contains("arith.mulf"));
        assert!(code.contains("f64"));
    }

    #[test]
    fn test_emit_binop_comparison() {
        let mut ctx = TextMlirContext::new();
        let code = emit_binop(&mut ctx, &BinOp::Lt, "%0", "%1", &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("arith.cmpi slt,"));
        assert!(code.contains("i1")); // Comparisons return boolean
    }

    #[test]
    fn test_emit_constant_int() {
        let mut ctx = TextMlirContext::new();
        let code = emit_constant_int(&mut ctx, 42, &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("arith.constant 42"));
        assert!(code.contains("i64"));
    }

    #[test]
    fn test_emit_constant_float() {
        let mut ctx = TextMlirContext::new();
        let code = emit_constant_float(&mut ctx, 3.14, &Type::Prim(PrimType::F64)).unwrap();
        assert!(code.contains("arith.constant 3.14"));
        assert!(code.contains("f64"));
    }

    #[test]
    fn test_emit_constant_bool() {
        let mut ctx = TextMlirContext::new();
        let code = emit_constant_bool(&mut ctx, true).unwrap();
        assert!(code.contains("arith.constant true"));
        assert!(code.contains("i1"));
    }

    #[test]
    fn test_emit_negation() {
        let mut ctx = TextMlirContext::new();

        // Integer negation
        let code = emit_negi(&mut ctx, "%0", &Type::Prim(PrimType::I64)).unwrap();
        assert!(code.contains("arith.constant 0"));
        assert!(code.contains("arith.subi"));

        // Float negation
        let mut ctx2 = TextMlirContext::new();
        let code = emit_negf(&mut ctx2, "%0", &Type::Prim(PrimType::F64)).unwrap();
        assert!(code.contains("arith.negf"));
    }

    #[test]
    fn test_emit_not() {
        let mut ctx = TextMlirContext::new();
        let code = emit_not(&mut ctx, "%0").unwrap();
        assert!(code.contains("arith.constant true"));
        assert!(code.contains("arith.xori"));
    }
}
