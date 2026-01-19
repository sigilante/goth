//! Types for built-in operations

use goth_ast::types::{Type, PrimType, TypeParam, TypeParamKind};
use goth_ast::op::{BinOp, UnaryOp};
use goth_ast::shape::{Shape, Dim};
use crate::error::{TypeError, TypeResult};

/// Get type of a binary operation
pub fn binop_type(op: BinOp, left: &Type, right: &Type) -> TypeResult<Type> {
    use BinOp::*;
    
    match op {
        // Arithmetic: numeric × numeric → numeric
        Add | Sub | Mul | Div | Mod | Pow => {
            numeric_binop(op, left, right)
        }
        
        // Comparison: T × T → Bool
        Eq | Neq | Lt | Gt | Leq | Geq => {
            comparison_op(left, right)
        }
        
        // Logical: Bool × Bool → Bool
        And | Or => {
            logical_op(left, right)
        }
        
        // Map: [n]T × (T → U) → [n]U
        Map => {
            map_type(left, right)
        }
        
        // Filter: [n]T × (T → Bool) → [?]T
        Filter => {
            filter_type(left, right)
        }
        
        // ZipWith: [n]T × [n]U → [n]⟨T, U⟩
        ZipWith => {
            zip_type(left, right)
        }
        
        // Compose: (B → C) × (A → B) → (A → C)
        Compose => {
            compose_type(left, right)
        }
        
        // Concat: [n]T × [m]T → [n+m]T
        Concat => {
            concat_type(left, right)
        }
        
        // Bind: [n]T × (T → [m]U) → [n×m]U (monadic bind for arrays)
        Bind => {
            bind_type(left, right)
        }
        
        // Custom operators - not yet supported
        Custom(name) => {
            Err(TypeError::InvalidBinOp {
                op: format!("custom({})", name),
                left: left.clone(),
                right: right.clone(),
            })
        }

        // Uncertainty: numeric × numeric → numeric
        BinOp::PlusMinus => {
            let value_ty = numeric_binop(op, left, right)?;
            Ok(Type::Uncertain(Box::new(value_ty.clone()), Box::new(value_ty)))
        }
    }
}

fn numeric_binop(op: BinOp, left: &Type, right: &Type) -> TypeResult<Type> {
    match (left, right) {
        // Scalar × Scalar
        (Type::Prim(p1), Type::Prim(p2)) if p1.is_numeric() && p2.is_numeric() => {
            // Promote to wider type
            Ok(Type::Prim(promote_numeric(*p1, *p2)))
        }

        // Tensor × Tensor (element-wise)
        (Type::Tensor(sh1, el1), Type::Tensor(sh2, el2)) => {
            // Use proper shape unification instead of structural equality
            use crate::shapes::unify_shapes;
            let _shape_subst = unify_shapes(sh1, sh2).map_err(|_| {
                TypeError::ShapeMismatch {
                    expected: sh1.clone(),
                    found: sh2.clone(),
                }
            })?;

            // After unification succeeds, use the first shape
            // (they're now known to be compatible)
            let result_elem = numeric_binop(op, el1, el2)?;
            Ok(Type::Tensor(sh1.clone(), Box::new(result_elem)))
        }

        // Scalar × Tensor or Tensor × Scalar (broadcasting)
        (Type::Prim(p), Type::Tensor(sh, el)) | (Type::Tensor(sh, el), Type::Prim(p))
            if p.is_numeric() => {
            let result_elem = numeric_binop(op, &Type::Prim(*p), el)?;
            Ok(Type::Tensor(sh.clone(), Box::new(result_elem)))
        }

        _ => Err(TypeError::InvalidBinOp {
            op: format!("{:?}", op),
            left: left.clone(),
            right: right.clone(),
        })
    }
}

fn promote_numeric(p1: PrimType, p2: PrimType) -> PrimType {
    use PrimType::*;
    match (p1, p2) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        _ => p1,
    }
}

fn comparison_op(_left: &Type, _right: &Type) -> TypeResult<Type> {
    // Types must be unifiable (we'll check during unification)
    // For now, just accept if both are the same primitive or same tensor
    Ok(Type::Prim(PrimType::Bool))
}

fn logical_op(left: &Type, right: &Type) -> TypeResult<Type> {
    match (left, right) {
        (Type::Prim(PrimType::Bool), Type::Prim(PrimType::Bool)) => {
            Ok(Type::Prim(PrimType::Bool))
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "logical".to_string(),
            left: left.clone(),
            right: right.clone(),
        })
    }
}

fn map_type(arr: &Type, func: &Type) -> TypeResult<Type> {
    match (arr, func) {
        (Type::Tensor(shape, _elem), Type::Fn(_arg, ret)) => {
            // elem must unify with arg (checked during unification)
            Ok(Type::Tensor(shape.clone(), ret.clone()))
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "map".to_string(),
            left: arr.clone(),
            right: func.clone(),
        })
    }
}

fn filter_type(arr: &Type, pred: &Type) -> TypeResult<Type> {
    match (arr, pred) {
        (Type::Tensor(_, elem), Type::Fn(_arg, _ret)) => {
            // arg must unify with elem, ret must be Bool
            // Result has unknown length (dynamic)
            Ok(Type::Tensor(
                Shape(vec![Dim::Var("?".into())]),  // Unknown length
                elem.clone(),
            ))
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "filter".to_string(),
            left: arr.clone(),
            right: pred.clone(),
        })
    }
}

fn zip_type(left: &Type, right: &Type) -> TypeResult<Type> {
    match (left, right) {
        (Type::Tensor(sh1, el1), Type::Tensor(sh2, el2)) => {
            // Use proper shape unification
            use crate::shapes::unify_shapes;
            let _shape_subst = unify_shapes(sh1, sh2).map_err(|_| {
                TypeError::ShapeMismatch {
                    expected: sh1.clone(),
                    found: sh2.clone(),
                }
            })?;

            Ok(Type::Tensor(
                sh1.clone(),
                Box::new(Type::tuple(vec![*el1.clone(), *el2.clone()])),
            ))
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "zip".to_string(),
            left: left.clone(),
            right: right.clone(),
        })
    }
}

fn compose_type(f: &Type, g: &Type) -> TypeResult<Type> {
    match (f, g) {
        (Type::Fn(_b1, c), Type::Fn(a, _b2)) => {
            // b1 must unify with b2
            Ok(Type::Fn(a.clone(), c.clone()))
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "compose".to_string(),
            left: f.clone(),
            right: g.clone(),
        })
    }
}

fn concat_type(left: &Type, right: &Type) -> TypeResult<Type> {
    match (left, right) {
        (Type::Tensor(sh1, el1), Type::Tensor(sh2, _el2)) => {
            // Elements must unify
            // Result shape is sum of dimensions
            if sh1.rank() != 1 || sh2.rank() != 1 {
                return Err(TypeError::InvalidBinOp {
                    op: "concat".to_string(),
                    left: left.clone(),
                    right: right.clone(),
                });
            }
            let new_dim = Dim::BinOp(
                Box::new(sh1.0[0].clone()),
                goth_ast::shape::DimOp::Add,
                Box::new(sh2.0[0].clone()),
            );
            Ok(Type::Tensor(Shape(vec![new_dim]), el1.clone()))
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "concat".to_string(),
            left: left.clone(),
            right: right.clone(),
        })
    }
}

fn bind_type(arr: &Type, func: &Type) -> TypeResult<Type> {
    match (arr, func) {
        (Type::Tensor(_, _elem), Type::Fn(_arg, ret)) => {
            match ret.as_ref() {
                Type::Tensor(_, inner_elem) => {
                    Ok(Type::Tensor(
                        Shape(vec![Dim::Var("?".into())]),  // Unknown flattened length
                        inner_elem.clone(),
                    ))
                }
                _ => Err(TypeError::InvalidBinOp {
                    op: "bind".to_string(),
                    left: arr.clone(),
                    right: func.clone(),
                })
            }
        }
        _ => Err(TypeError::InvalidBinOp {
            op: "bind".to_string(),
            left: arr.clone(),
            right: func.clone(),
        })
    }
}

/// Get type of a unary operation
pub fn unaryop_type(op: UnaryOp, operand: &Type) -> TypeResult<Type> {
    use UnaryOp::*;
    
    match op {
        // Negation: numeric → numeric
        Neg => {
            match operand {
                Type::Prim(p) if p.is_numeric() => Ok(operand.clone()),
                Type::Tensor(sh, el) => {
                    let inner = unaryop_type(Neg, el)?;
                    Ok(Type::Tensor(sh.clone(), Box::new(inner)))
                }
                _ => Err(TypeError::InvalidUnaryOp {
                    op: "neg".to_string(),
                    operand: operand.clone(),
                })
            }
        }
        
        // Logical not: Bool → Bool
        Not => {
            match operand {
                Type::Prim(PrimType::Bool) => Ok(Type::Prim(PrimType::Bool)),
                _ => Err(TypeError::InvalidUnaryOp {
                    op: "not".to_string(),
                    operand: operand.clone(),
                })
            }
        }
        
        // Sum: [n]T → T (where T is numeric)
        Sum | Prod => {
            match operand {
                Type::Tensor(_, elem) => {
                    if let Type::Prim(p) = elem.as_ref() {
                        if p.is_numeric() {
                            return Ok(*elem.clone());
                        }
                    }
                    Err(TypeError::InvalidUnaryOp {
                        op: format!("{:?}", op),
                        operand: operand.clone(),
                    })
                }
                _ => Err(TypeError::InvalidUnaryOp {
                    op: format!("{:?}", op),
                    operand: operand.clone(),
                })
            }
        }
        
        // Scan: [n]T → [n]T
        Scan => {
            match operand {
                Type::Tensor(_sh, elem) => {
                    if let Type::Prim(p) = elem.as_ref() {
                        if p.is_numeric() {
                            return Ok(operand.clone());
                        }
                    }
                    Err(TypeError::InvalidUnaryOp {
                        op: "scan".to_string(),
                        operand: operand.clone(),
                    })
                }
                _ => Err(TypeError::InvalidUnaryOp {
                    op: "scan".to_string(),
                    operand: operand.clone(),
                })
            }
        }

        // Sqrt, Floor, Ceil, and math functions: T → F64 (where T is numeric)
        UnaryOp::Sqrt | UnaryOp::Floor | UnaryOp::Ceil
        | UnaryOp::Gamma | UnaryOp::Ln | UnaryOp::Exp
        | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Abs => {
            match operand {
                Type::Prim(p) if p.is_numeric() => Ok(Type::Prim(PrimType::F64)),
                Type::Tensor(sh, el) => {
                    let inner = unaryop_type(op, el)?;
                    Ok(Type::Tensor(sh.clone(), Box::new(inner)))
                }
                _ => Err(TypeError::InvalidUnaryOp {
                    op: format!("{:?}", op),
                    operand: operand.clone(),
                })
            }
        }
    }
}

/// Get type of a primitive function
pub fn primitive_type(name: &str) -> Option<Type> {
    match name {
        "sqrt" | "exp" | "ln" | "sin" | "cos" | "tan" | 
        "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" |
        "floor" | "ceil" | "round" | "abs" => {
            Some(Type::func(Type::Prim(PrimType::F64), Type::Prim(PrimType::F64)))
        }
        "atan2" | "pow" | "min" | "max" => {
            Some(Type::func_n(
                [Type::Prim(PrimType::F64), Type::Prim(PrimType::F64)],
                Type::Prim(PrimType::F64),
            ))
        }
        "length" => {
            // ∀n α. [n]α → I64
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    Type::Prim(PrimType::I64),
                )),
            ))
        }
        _ => None,
    }
}