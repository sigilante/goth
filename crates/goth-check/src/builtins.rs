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

        // Write: String × String → ()
        Write => {
            write_type(left, right)
        }

        // Read: String × _ → String (left is path, right is ignored or used for pipeline)
        Read => {
            read_type(left, right)
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

/// Write: String × String → ()
fn write_type(_content: &Type, _path: &Type) -> TypeResult<Type> {
    // For now, accept any types and return Unit
    // A more strict version would check that both are String
    Ok(Type::unit())
}

/// Read: String × _ → String
fn read_type(_path: &Type, _right: &Type) -> TypeResult<Type> {
    // Returns the file contents as a String (char tensor)
    Ok(Type::Tensor(
        Shape(vec![Dim::Var("?".into())]),
        Box::new(Type::Prim(PrimType::Char)),
    ))
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
        
        // Sum: [n]T → T (where T is numeric, type variable, or hole)
        Sum | Prod => {
            match operand {
                Type::Tensor(_, elem) => {
                    match elem.as_ref() {
                        Type::Prim(p) if p.is_numeric() => Ok(*elem.clone()),
                        // Allow type variables and holes (assume numeric at runtime)
                        Type::Var(_) | Type::Hole => Ok(*elem.clone()),
                        _ => Err(TypeError::InvalidUnaryOp {
                            op: format!("{:?}", op),
                            operand: operand.clone(),
                        })
                    }
                }
                _ => Err(TypeError::InvalidUnaryOp {
                    op: format!("{:?}", op),
                    operand: operand.clone(),
                })
            }
        }

        // Scan: [n]T → [n]T (where T is numeric, type variable, or hole)
        Scan => {
            match operand {
                Type::Tensor(_sh, elem) => {
                    match elem.as_ref() {
                        Type::Prim(p) if p.is_numeric() => Ok(operand.clone()),
                        // Allow type variables and holes (assume numeric at runtime)
                        Type::Var(_) | Type::Hole => Ok(operand.clone()),
                        _ => Err(TypeError::InvalidUnaryOp {
                            op: "scan".to_string(),
                            operand: operand.clone(),
                        })
                    }
                }
                _ => Err(TypeError::InvalidUnaryOp {
                    op: "scan".to_string(),
                    operand: operand.clone(),
                })
            }
        }

        // Sqrt, Floor, Ceil, Round, and math functions: T → F64 (where T is numeric)
        UnaryOp::Sqrt | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round
        | UnaryOp::Gamma | UnaryOp::Ln | UnaryOp::Log10 | UnaryOp::Log2 | UnaryOp::Exp
        | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan
        | UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan
        | UnaryOp::Sinh | UnaryOp::Cosh | UnaryOp::Tanh
        | UnaryOp::Abs | UnaryOp::Sign => {
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
        "length" | "len" => {
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
        // Matrix operations with shape checking
        "dot" | "·" => {
            // ∀n. [n]F64 → [n]F64 → F64 (dot product)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::F64))),
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::F64))),
                    ],
                    Type::Prim(PrimType::F64),
                )),
            ))
        }
        "matmul" => {
            // ∀m n p. [m n]F64 → [n p]F64 → [m p]F64 (matrix multiplication)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "m".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "p".into(), kind: TypeParamKind::Shape },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Tensor(Shape(vec![Dim::Var("m".into()), Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::F64))),
                        Type::Tensor(Shape(vec![Dim::Var("n".into()), Dim::Var("p".into())]), Box::new(Type::Prim(PrimType::F64))),
                    ],
                    Type::Tensor(Shape(vec![Dim::Var("m".into()), Dim::Var("p".into())]), Box::new(Type::Prim(PrimType::F64))),
                )),
            ))
        }
        "transpose" | "⍉" => {
            // ∀m n. [m n]F64 → [n m]F64 (matrix transpose)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "m".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("m".into()), Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::F64))),
                    Type::Tensor(Shape(vec![Dim::Var("n".into()), Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::F64))),
                )),
            ))
        }
        "norm" => {
            // ∀n. [n]F64 → F64 (vector norm)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::F64))),
                    Type::Prim(PrimType::F64),
                )),
            ))
        }
        // Sequence generation
        "iota" | "ι" | "⍳" => {
            // I64 → [n]I64 (generate 0..n-1)
            Some(Type::func(
                Type::Prim(PrimType::I64),
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::I64))),
            ))
        }
        "range" => {
            // I64 → I64 → [m]I64 (generate start..end-1)
            Some(Type::func_n(
                [Type::Prim(PrimType::I64), Type::Prim(PrimType::I64)],
                Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::I64))),
            ))
        }
        // Type conversions
        "toInt" => {
            // ∀α. α → I64 (convert any type to integer)
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func(Type::Var("α".into()), Type::Prim(PrimType::I64))),
            ))
        }
        "toFloat" => {
            // ∀α. α → F64 (convert any type to float)
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func(Type::Var("α".into()), Type::Prim(PrimType::F64))),
            ))
        }
        "toBool" => {
            // ∀α. α → Bool (convert any type to boolean)
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func(Type::Var("α".into()), Type::Prim(PrimType::Bool))),
            ))
        }
        "toChar" => {
            // I64 → Char (convert integer to character)
            Some(Type::func(Type::Prim(PrimType::I64), Type::Prim(PrimType::Char)))
        }
        "parseInt" => {
            // String → I64 (parse string as integer)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Prim(PrimType::I64),
            ))
        }
        "parseFloat" => {
            // String → F64 (parse string as float)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Prim(PrimType::F64),
            ))
        }
        "toString" => {
            // ∀α. α → String (convert any type to string)
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func(
                    Type::Var("α".into()),
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                )),
            ))
        }
        "chars" => {
            // String → [n]Char (convert string to array of characters)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
            ))
        }
        // Aggregation
        "sum" | "Σ" => {
            // ∀n α. [n]α → α (sum of numeric array)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    Type::Var("α".into()),
                )),
            ))
        }
        "prod" | "Π" => {
            // ∀n α. [n]α → α (product of numeric array)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    Type::Var("α".into()),
                )),
            ))
        }
        // Array operations
        "reverse" => {
            // ∀n α. [n]α → [n]α (reverse array)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                )),
            ))
        }
        "take" => {
            // ∀n m α. I64 → [n]α → [m]α (take first k elements)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "m".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Prim(PrimType::I64),
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    ],
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Var("α".into()))),
                )),
            ))
        }
        "drop" => {
            // ∀n m α. I64 → [n]α → [m]α (drop first k elements)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "m".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Prim(PrimType::I64),
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    ],
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Var("α".into()))),
                )),
            ))
        }
        "concat" | "⧺" => {
            // ∀n m p α. [n]α → [m]α → [p]α (concatenate arrays)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "m".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "p".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                        Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Var("α".into()))),
                    ],
                    Type::Tensor(Shape(vec![Dim::Var("p".into())]), Box::new(Type::Var("α".into()))),
                )),
            ))
        }
        // I/O
        "print" => {
            // ∀α. α → Unit (print value)
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func(Type::Var("α".into()), Type::unit())),
            ))
        }
        "readLine" => {
            // Unit → String (read line from stdin)
            Some(Type::func(
                Type::unit(),
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
            ))
        }
        "readFile" => {
            // String → String (read file contents)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
            ))
        }
        "writeFile" => {
            // String → String → Unit (write content to file)
            Some(Type::func_n(
                [
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                ],
                Type::unit(),
            ))
        }
        // TUI primitives
        "write" => {
            // ∀α. α → Unit (write without newline)
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func(Type::Var("α".into()), Type::unit())),
            ))
        }
        "flush" => {
            // Unit → Unit (flush stdout)
            Some(Type::func(Type::unit(), Type::unit()))
        }
        "readKey" => {
            // Unit → Int (read single key code)
            Some(Type::func(Type::unit(), Type::Prim(PrimType::Int)))
        }
        "rawModeEnter" => {
            // Unit → Unit (enter raw terminal mode)
            Some(Type::func(Type::unit(), Type::unit()))
        }
        "rawModeExit" => {
            // Unit → Unit (exit raw terminal mode)
            Some(Type::func(Type::unit(), Type::unit()))
        }
        "sleep" => {
            // Int → Unit (sleep for milliseconds)
            Some(Type::func(Type::Prim(PrimType::Int), Type::unit()))
        }
        // String splitting (for wc-like operations)
        "lines" => {
            // String → [m]String (split by newlines)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Tensor(
                    Shape(vec![Dim::Var("m".into())]),
                    Box::new(Type::Tensor(Shape(vec![Dim::Var("k".into())]), Box::new(Type::Prim(PrimType::Char)))),
                ),
            ))
        }
        "words" => {
            // String → [m]String (split by whitespace)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Tensor(
                    Shape(vec![Dim::Var("m".into())]),
                    Box::new(Type::Tensor(Shape(vec![Dim::Var("k".into())]), Box::new(Type::Prim(PrimType::Char)))),
                ),
            ))
        }
        "bytes" => {
            // String → [m]I64 (UTF-8 byte values)
            Some(Type::func(
                Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::I64))),
            ))
        }
        // String comparison
        "strEq" => {
            // String → String → Bool
            Some(Type::func_n(
                [
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                ],
                Type::Prim(PrimType::Bool),
            ))
        }
        "startsWith" => {
            // String → String → Bool (check if string starts with prefix)
            Some(Type::func_n(
                [
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                ],
                Type::Prim(PrimType::Bool),
            ))
        }
        "endsWith" => {
            // String → String → Bool (check if string ends with suffix)
            Some(Type::func_n(
                [
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                ],
                Type::Prim(PrimType::Bool),
            ))
        }
        "contains" => {
            // String → String → Bool (check if string contains substring)
            Some(Type::func_n(
                [
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                ],
                Type::Prim(PrimType::Bool),
            ))
        }
        // Stream values (stdout, stderr) — typed as Unit for now
        "stdout" | "stderr" => {
            Some(Type::unit())
        }
        // String concatenation (⧺ is now handled by concat which preserves Char type)
        "strConcat" => {
            // String → String → String (concatenate two strings)
            Some(Type::func_n(
                [
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Prim(PrimType::Char))),
                    Type::Tensor(Shape(vec![Dim::Var("m".into())]), Box::new(Type::Prim(PrimType::Char))),
                ],
                Type::Tensor(Shape(vec![Dim::Var("p".into())]), Box::new(Type::Prim(PrimType::Char))),
            ))
        }
        // Replicate: create an array of n copies of a value
        "replicate" => {
            // ∀α. I64 → α → [n]α
            Some(Type::Forall(
                vec![TypeParam { name: "α".into(), kind: TypeParamKind::Type }],
                Box::new(Type::func_n(
                    [Type::Prim(PrimType::I64), Type::Var("α".into())],
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                )),
            ))
        }
        // Shape query
        "shape" => {
            // ∀n α. [n]α → [1]I64 (get shape as array)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func(
                    Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                    Type::Tensor(Shape(vec![Dim::constant(1)]), Box::new(Type::Prim(PrimType::I64))),
                )),
            ))
        }
        // Index into array
        "index" => {
            // ∀n α. [n]α → I64 → α (get element at index)
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                        Type::Prim(PrimType::I64),
                    ],
                    Type::Var("α".into()),
                )),
            ))
        }
        // Zip two arrays
        "zip" => {
            // ∀n α β. [n]α → [n]β → [n]⟨α, β⟩
            Some(Type::Forall(
                vec![
                    TypeParam { name: "n".into(), kind: TypeParamKind::Shape },
                    TypeParam { name: "α".into(), kind: TypeParamKind::Type },
                    TypeParam { name: "β".into(), kind: TypeParamKind::Type },
                ],
                Box::new(Type::func_n(
                    [
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("α".into()))),
                        Type::Tensor(Shape(vec![Dim::Var("n".into())]), Box::new(Type::Var("β".into()))),
                    ],
                    Type::Tensor(
                        Shape(vec![Dim::Var("n".into())]),
                        Box::new(Type::tuple(vec![Type::Var("α".into()), Type::Var("β".into())])),
                    ),
                )),
            ))
        }
        _ => None,
    }
}