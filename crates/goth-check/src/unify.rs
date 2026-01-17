//! Type unification algorithm

use goth_ast::types::{Type, PrimType};
use goth_ast::shape::{Shape, Dim};
use crate::error::{TypeError, TypeResult};
use crate::subst::{Subst, apply_type, apply_dim};

/// Unify two types, returning a substitution that makes them equal
pub fn unify(t1: &Type, t2: &Type) -> TypeResult<Subst> {
    match (t1, t2) {
        // Identical primitives
        (Type::Prim(p1), Type::Prim(p2)) if p1 == p2 => {
            Ok(Subst::new())
        }

        // Type variable on left
        (Type::Var(v), ty) => {
            unify_var(v, ty)
        }

        // Type variable on right
        (ty, Type::Var(v)) => {
            unify_var(v, ty)
        }

        // Function types
        (Type::Fn(a1, r1), Type::Fn(a2, r2)) => {
            let s1 = unify(a1, a2)?;
            let r1_sub = apply_type(&s1, r1);
            let r2_sub = apply_type(&s1, r2);
            let s2 = unify(&r1_sub, &r2_sub)?;
            Ok(Subst::compose(s1, s2))
        }

        // Tensor types: unify shapes AND element types
        (Type::Tensor(sh1, el1), Type::Tensor(sh2, el2)) => {
            let s1 = unify_shapes(sh1, sh2)?;
            let el1_sub = apply_type(&s1, el1);
            let el2_sub = apply_type(&s1, el2);
            let s2 = unify(&el1_sub, &el2_sub)?;
            Ok(Subst::compose(s1, s2))
        }

        // Tuple types: unify element-wise
        (Type::Tuple(fs1), Type::Tuple(fs2)) => {
            if fs1.len() != fs2.len() {
                return Err(TypeError::TupleArityMismatch {
                    expected: fs1.len(),
                    found: fs2.len(),
                });
            }
            let mut subst = Subst::new();
            for (f1, f2) in fs1.iter().zip(fs2.iter()) {
                let s = unify(&apply_type(&subst, &f1.ty), &apply_type(&subst, &f2.ty))?;
                subst = Subst::compose(subst, s);
            }
            Ok(subst)
        }

        // Option types
        (Type::Option(t1), Type::Option(t2)) => {
            unify(t1, t2)
        }

        // Uncertain types
        (Type::Uncertain(t1), Type::Uncertain(t2)) => {
            unify(t1, t2)
        }

        // Holes unify with anything
        (Type::Hole, _) | (_, Type::Hole) => {
            Ok(Subst::new())
        }

        // Mismatch
        _ => Err(TypeError::UnificationFailure {
            t1: t1.clone(),
            t2: t2.clone(),
        })
    }
}

/// Unify a type variable with a type
fn unify_var(var: &str, ty: &Type) -> TypeResult<Subst> {
    // Check if ty is the same variable
    if let Type::Var(v) = ty {
        if var == v.as_ref() {
            return Ok(Subst::new());
        }
    }

    // Occurs check: var must not appear free in ty
    if occurs_in(var, ty) {
        return Err(TypeError::InfiniteType {
            var: var.to_string(),
            ty: ty.clone(),
        });
    }

    Ok(Subst::singleton_type(var, ty.clone()))
}

/// Check if a type variable occurs in a type
fn occurs_in(var: &str, ty: &Type) -> bool {
    match ty {
        Type::Var(v) => var == v.as_ref(),
        Type::Fn(a, r) => occurs_in(var, a) || occurs_in(var, r),
        Type::Tensor(_, elem) => occurs_in(var, elem),
        Type::Tuple(fields) => fields.iter().any(|f| occurs_in(var, &f.ty)),
        Type::Option(t) | Type::Uncertain(t) => occurs_in(var, t),
        Type::Forall(params, body) => {
            // Don't look through bound variables
            if params.iter().any(|p| p.name.as_ref() == var) {
                false
            } else {
                occurs_in(var, body)
            }
        }
        _ => false,
    }
}

/// Unify two shapes
pub fn unify_shapes(sh1: &Shape, sh2: &Shape) -> TypeResult<Subst> {
    if sh1.rank() != sh2.rank() {
        return Err(TypeError::RankMismatch {
            expected: sh1.rank(),
            found: sh2.rank(),
        });
    }

    let mut subst = Subst::new();
    for (i, (d1, d2)) in sh1.0.iter().zip(&sh2.0).enumerate() {
        let d1_sub = apply_dim(&subst, d1);
        let d2_sub = apply_dim(&subst, d2);
        let s = unify_dims(&d1_sub, &d2_sub, i)?;
        subst = Subst::compose(subst, s);
    }
    Ok(subst)
}

/// Unify two dimensions
fn unify_dims(d1: &Dim, d2: &Dim, position: usize) -> TypeResult<Subst> {
    match (d1, d2) {
        // Same constant
        (Dim::Const(n1), Dim::Const(n2)) if n1 == n2 => {
            Ok(Subst::new())
        }

        // Variable on left
        (Dim::Var(v), dim) => {
            if let Dim::Var(v2) = dim {
                if v.as_ref() == v2.as_ref() {
                    return Ok(Subst::new());
                }
            }
            Ok(Subst::singleton_shape(v.as_ref(), dim.clone()))
        }

        // Variable on right
        (dim, Dim::Var(v)) => {
            Ok(Subst::singleton_shape(v.as_ref(), dim.clone()))
        }

        // Mismatched constants
        (Dim::Const(n1), Dim::Const(n2)) => {
            Err(TypeError::DimMismatch {
                position,
                expected: n1.to_string(),
                found: n2.to_string(),
            })
        }

        // BinOp cases need more sophisticated solving
        // For now, just check structural equality
        (Dim::BinOp(l1, op1, r1), Dim::BinOp(l2, op2, r2)) if op1 == op2 => {
            let s1 = unify_dims(l1, l2, position)?;
            let r1_sub = apply_dim(&s1, r1);
            let r2_sub = apply_dim(&s1, r2);
            let s2 = unify_dims(&r1_sub, &r2_sub, position)?;
            Ok(Subst::compose(s1, s2))
        }

        _ => Err(TypeError::DimMismatch {
            position,
            expected: format!("{}", d1),
            found: format!("{}", d2),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_primitives() {
        let s = unify(&Type::Prim(PrimType::F64), &Type::Prim(PrimType::F64)).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn test_unify_var() {
        let s = unify(&Type::Var("α".into()), &Type::Prim(PrimType::F64)).unwrap();
        assert_eq!(s.types.get("α"), Some(&Type::Prim(PrimType::F64)));
    }

    #[test]
    fn test_unify_functions() {
        let t1 = Type::Fn(
            Box::new(Type::Var("α".into())),
            Box::new(Type::Prim(PrimType::Bool)),
        );
        let t2 = Type::Fn(
            Box::new(Type::Prim(PrimType::F64)),
            Box::new(Type::Prim(PrimType::Bool)),
        );
        let s = unify(&t1, &t2).unwrap();
        assert_eq!(s.types.get("α"), Some(&Type::Prim(PrimType::F64)));
    }

    #[test]
    fn test_unify_shapes() {
        let sh1 = Shape(vec![Dim::Var("n".into())]);
        let sh2 = Shape(vec![Dim::Const(10)]);
        let s = unify_shapes(&sh1, &sh2).unwrap();
        assert_eq!(s.shapes.get("n"), Some(&Dim::Const(10)));
    }

    #[test]
    fn test_occurs_check() {
        let t1 = Type::Var("α".into());
        let t2 = Type::Fn(Box::new(Type::Var("α".into())), Box::new(Type::Prim(PrimType::Bool)));
        assert!(unify(&t1, &t2).is_err());
    }
}