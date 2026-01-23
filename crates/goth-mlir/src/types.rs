//! Type conversion from Goth types to MLIR types
//!
//! This module handles the mapping between Goth's type system and MLIR's type system.
//!
//! Type mappings:
//! - `I64` → `i64`
//! - `F64` → `f64`
//! - `Bool` → `i1`
//! - `String` → `!llvm.ptr` (pointer to null-terminated string)
//! - `Tensor<shape, elem>` → `tensor<shape x elem>`
//! - `Fn(A, B)` → `(A) -> B`
//! - `Tuple(A, B, C)` → `tuple<A, B, C>`
//! - `Unit` → `()`

use goth_ast::types::{Type, PrimType, TupleField};
use goth_ast::shape::{Shape, Dim};
use crate::error::{MlirError, Result};

#[cfg(feature = "melior")]
use melior::{
    ir::Type as MlirType,
    Context,
};

/// Convert a Goth type to MLIR type string representation
///
/// This is the text-based version that generates MLIR type syntax as a string.
pub fn type_to_mlir_string(ty: &Type) -> Result<String> {
    match ty {
        // Primitive types
        Type::Prim(PrimType::I64) => Ok("i64".to_string()),
        Type::Prim(PrimType::I32) => Ok("i32".to_string()),
        Type::Prim(PrimType::I16) => Ok("i16".to_string()),
        Type::Prim(PrimType::I8) => Ok("i8".to_string()),
        Type::Prim(PrimType::U64) => Ok("i64".to_string()), // MLIR doesn't have unsigned
        Type::Prim(PrimType::U32) => Ok("i32".to_string()),
        Type::Prim(PrimType::U16) => Ok("i16".to_string()),
        Type::Prim(PrimType::U8) => Ok("i8".to_string()),
        Type::Prim(PrimType::F64) => Ok("f64".to_string()),
        Type::Prim(PrimType::F32) => Ok("f32".to_string()),
        Type::Prim(PrimType::Bool) => Ok("i1".to_string()),
        Type::Prim(PrimType::String) => Ok("!llvm.ptr".to_string()),
        Type::Prim(PrimType::Char) => Ok("i32".to_string()), // UTF-32 code point
        Type::Prim(PrimType::Byte) => Ok("i8".to_string()),
        Type::Prim(PrimType::Nat) => Ok("i64".to_string()),  // Natural numbers as i64
        Type::Prim(PrimType::Int) => Ok("i64".to_string()),  // Arbitrary precision as i64

        // Unit type (empty tuple)
        Type::Tuple(fields) if fields.is_empty() => Ok("()".to_string()),

        // Tuple types
        Type::Tuple(fields) => {
            let field_types: Result<Vec<_>> = fields.iter()
                .map(|f| type_to_mlir_string(&f.ty))
                .collect();
            Ok(format!("tuple<{}>", field_types?.join(", ")))
        }

        // Tensor types
        Type::Tensor(shape, elem) => {
            let elem_ty = type_to_mlir_string(elem)?;
            let shape_str = shape_to_mlir_string(shape);

            if shape_str.is_empty() {
                Ok(format!("tensor<{}>", elem_ty))
            } else {
                Ok(format!("tensor<{}x{}>", shape_str, elem_ty))
            }
        }

        // Function types
        Type::Fn(arg, ret) => {
            let arg_ty = type_to_mlir_string(arg)?;
            let ret_ty = type_to_mlir_string(ret)?;
            Ok(format!("({}) -> {}", arg_ty, ret_ty))
        }

        // Type variables - resolve to concrete MLIR types
        Type::Var(name) => {
            match name.as_ref() {
                // Integer-like type variables
                "I" | "Int" | "ℤ" | "N" | "Nat" | "ℕ" => Ok("i64".to_string()),
                // Float-like type variables
                "F" | "Float" | "ℝ" => Ok("f64".to_string()),
                // Boolean-like type variables
                "B" | "Bool" => Ok("i1".to_string()),
                // Default to i64 for unknown type variables
                _ => Ok("i64".to_string()),
            }
        }

        // Variant/sum types - for now, use i64 for tag + max payload size
        Type::Variant(_) => {
            // Variants are represented as {tag: i32, payload: max_payload_type}
            // For now, use a generic representation
            Ok("!goth.variant".to_string())
        }

        // Forall types - should be monomorphized before reaching MLIR
        Type::Forall(_, inner) => type_to_mlir_string(inner),

        // Existential types - should be monomorphized before reaching MLIR
        Type::Exists(_, inner) => type_to_mlir_string(inner),

        // Uncertain types - represented as tuple of (value, uncertainty)
        Type::Uncertain(value_ty, uncertainty_ty) => {
            let val_ty = type_to_mlir_string(value_ty)?;
            let unc_ty = type_to_mlir_string(uncertainty_ty)?;
            Ok(format!("tuple<{}, {}>", val_ty, unc_ty))
        }

        // Interval types - the interval constraint is erased at runtime
        Type::Interval(inner, _interval_set) => type_to_mlir_string(inner),

        // Effectful types - the effect is erased at runtime
        Type::Effectful(inner, _effects) => type_to_mlir_string(inner),

        // Refinement types - the predicate is erased at runtime
        Type::Refinement { base, .. } => type_to_mlir_string(base),

        // Type application - should be resolved before reaching MLIR
        Type::App(ty, _args) => type_to_mlir_string(ty),

        // Optional types - represented as a variant
        Type::Option(_) => Ok("!goth.option".to_string()),

        // Hole (inference placeholder) - should not reach MLIR
        Type::Hole => Err(MlirError::UnsupportedType("Type hole reached MLIR".to_string())),
    }
}

/// Convert a shape to MLIR dimension string
fn shape_to_mlir_string(shape: &Shape) -> String {
    shape.0.iter()
        .map(|dim| match dim {
            Dim::Const(n) => n.to_string(),
            Dim::Var(_) => "?".to_string(),
            Dim::BinOp(_, _, _) => "?".to_string(), // Dynamic/computed dimension
        })
        .collect::<Vec<_>>()
        .join("x")
}

/// Check if a type is an integer type
pub fn is_integer_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::I64) => true,
        Type::Var(name) => matches!(name.as_ref(), "I" | "Int" | "ℤ" | "N" | "Nat" | "ℕ"),
        _ => false,
    }
}

/// Check if a type is a floating-point type
pub fn is_float_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::F64) => true,
        Type::Var(name) => matches!(name.as_ref(), "F" | "Float" | "ℝ"),
        _ => false,
    }
}

/// Check if a type is a boolean type
pub fn is_bool_type(ty: &Type) -> bool {
    match ty {
        Type::Prim(PrimType::Bool) => true,
        Type::Var(name) => matches!(name.as_ref(), "B" | "Bool"),
        _ => false,
    }
}

/// Check if a type is a tensor type
pub fn is_tensor_type(ty: &Type) -> bool {
    matches!(ty, Type::Tensor(_, _))
}

/// Get the element type of a tensor, if applicable
pub fn tensor_element_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Tensor(_, elem) => Some(elem),
        _ => None,
    }
}

/// Get the shape of a tensor, if applicable
pub fn tensor_shape(ty: &Type) -> Option<&Shape> {
    match ty {
        Type::Tensor(shape, _) => Some(shape),
        _ => None,
    }
}

// Melior-based type conversion (when feature is enabled)
#[cfg(feature = "melior")]
pub mod melior_types {
    use super::*;

    /// Convert a Goth type to a proper MLIR type using melior
    pub fn convert_type<'ctx>(ctx: &'ctx Context, ty: &Type) -> Result<MlirType<'ctx>> {
        match ty {
            // Primitive types
            Type::Prim(PrimType::I64) => Ok(MlirType::integer(ctx, 64)),
            Type::Prim(PrimType::F64) => Ok(MlirType::float64(ctx)),
            Type::Prim(PrimType::Bool) => Ok(MlirType::integer(ctx, 1)),

            // Unit type
            Type::Tuple(fields) if fields.is_empty() => {
                Ok(MlirType::tuple(ctx, &[]))
            }

            // Tensor types
            Type::Tensor(shape, elem) => {
                let elem_ty = convert_type(ctx, elem)?;
                let dims = convert_shape(shape);
                Ok(MlirType::ranked_tensor(&dims, elem_ty))
            }

            // Function types
            Type::Fn(arg, ret) => {
                let arg_ty = convert_type(ctx, arg)?;
                let ret_ty = convert_type(ctx, ret)?;
                Ok(MlirType::function(&[arg_ty], &[ret_ty]))
            }

            // Type variables
            Type::Var(name) => {
                match name.as_ref() {
                    "I" | "Int" | "ℤ" | "N" | "Nat" | "ℕ" => Ok(MlirType::integer(ctx, 64)),
                    "F" | "Float" | "ℝ" => Ok(MlirType::float64(ctx)),
                    "B" | "Bool" => Ok(MlirType::integer(ctx, 1)),
                    _ => Ok(MlirType::integer(ctx, 64)),
                }
            }

            _ => Err(MlirError::UnsupportedType(format!("{:?}", ty))),
        }
    }

    /// Convert a shape to MLIR dimensions
    fn convert_shape(shape: &Shape) -> Vec<i64> {
        shape.0.iter()
            .map(|dim| match dim {
                Dim::Const(n) => *n as i64,
                Dim::Var(_) => -1, // Dynamic dimension
                Dim::BinOp(_, _, _) => -1, // Computed dimension
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::shape::Shape;

    #[test]
    fn test_prim_types() {
        assert_eq!(type_to_mlir_string(&Type::Prim(PrimType::I64)).unwrap(), "i64");
        assert_eq!(type_to_mlir_string(&Type::Prim(PrimType::F64)).unwrap(), "f64");
        assert_eq!(type_to_mlir_string(&Type::Prim(PrimType::Bool)).unwrap(), "i1");
        assert_eq!(type_to_mlir_string(&Type::Prim(PrimType::String)).unwrap(), "!llvm.ptr");
    }

    #[test]
    fn test_unit_type() {
        let unit = Type::Tuple(vec![]);
        assert_eq!(type_to_mlir_string(&unit).unwrap(), "()");
    }

    #[test]
    fn test_tuple_type() {
        let tuple = Type::Tuple(vec![
            TupleField { label: None, ty: Type::Prim(PrimType::I64) },
            TupleField { label: None, ty: Type::Prim(PrimType::F64) },
        ]);
        assert_eq!(type_to_mlir_string(&tuple).unwrap(), "tuple<i64, f64>");
    }

    #[test]
    fn test_tensor_type() {
        // 1D tensor of i64 with size 10
        let tensor = Type::Tensor(
            Shape(vec![Dim::Const(10)]),
            Box::new(Type::Prim(PrimType::I64)),
        );
        assert_eq!(type_to_mlir_string(&tensor).unwrap(), "tensor<10xi64>");

        // 2D tensor with dynamic first dimension
        let tensor_2d = Type::Tensor(
            Shape(vec![Dim::Var("n".into()), Dim::Const(5)]),
            Box::new(Type::Prim(PrimType::F64)),
        );
        assert_eq!(type_to_mlir_string(&tensor_2d).unwrap(), "tensor<?x5xf64>");
    }

    #[test]
    fn test_function_type() {
        let fn_type = Type::func(
            Type::Prim(PrimType::I64),
            Type::Prim(PrimType::F64),
        );
        assert_eq!(type_to_mlir_string(&fn_type).unwrap(), "(i64) -> f64");
    }

    #[test]
    fn test_type_variables() {
        assert_eq!(type_to_mlir_string(&Type::Var("I".into())).unwrap(), "i64");
        assert_eq!(type_to_mlir_string(&Type::Var("F".into())).unwrap(), "f64");
        assert_eq!(type_to_mlir_string(&Type::Var("Bool".into())).unwrap(), "i1");
    }

    #[test]
    fn test_is_integer_type() {
        assert!(is_integer_type(&Type::Prim(PrimType::I64)));
        assert!(is_integer_type(&Type::Var("I".into())));
        assert!(is_integer_type(&Type::Var("Int".into())));
        assert!(!is_integer_type(&Type::Prim(PrimType::F64)));
    }

    #[test]
    fn test_is_float_type() {
        assert!(is_float_type(&Type::Prim(PrimType::F64)));
        assert!(is_float_type(&Type::Var("F".into())));
        assert!(!is_float_type(&Type::Prim(PrimType::I64)));
    }
}
