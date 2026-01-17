//! Type checking (checking mode)

use goth_ast::expr::Expr;
use goth_ast::types::{Type, PrimType};

use crate::context::Context;
use crate::error::{TypeError, TypeResult};
use crate::infer::{infer, pattern_types};
use crate::unify::unify;

/// Check that an expression has the expected type
pub fn check(ctx: &mut Context, expr: &Expr, expected: &Type) -> TypeResult<()> {
    match expr {
        // ============ Lambda ============
        
        Expr::Lam(body) => {
            match expected {
                Type::Fn(arg_ty, ret_ty) => {
                    ctx.with_binding(*arg_ty.clone(), |ctx| {
                        check(ctx, body, ret_ty)
                    })
                }
                _ => Err(TypeError::Mismatch {
                    expected: expected.clone(),
                    found: Type::Fn(Box::new(Type::Hole), Box::new(Type::Hole)),
                })
            }
        }
        
        Expr::LamN(n, body) => {
            // Unroll expected type to find n argument types
            let mut current = expected;
            let mut arg_types = Vec::new();
            
            for _ in 0..*n {
                match current {
                    Type::Fn(arg, ret) => {
                        arg_types.push(*arg.clone());
                        current = ret;
                    }
                    _ => {
                        return Err(TypeError::Mismatch {
                            expected: expected.clone(),
                            found: Type::Hole, // Can't construct proper type
                        });
                    }
                }
            }
            
            ctx.with_bindings(&arg_types, |ctx| {
                check(ctx, body, current)
            })
        }
        
        // ============ If ============
        
        Expr::If { cond, then_, else_ } => {
            check(ctx, cond, &Type::Prim(PrimType::Bool))?;
            check(ctx, then_, expected)?;
            check(ctx, else_, expected)?;
            Ok(())
        }
        
        // ============ Let ============
        
        Expr::Let { pattern, value, body } => {
            let val_ty = infer(ctx, value)?;
            let bindings = pattern_types(pattern, &val_ty)?;
            ctx.with_bindings(&bindings, |ctx| {
                check(ctx, body, expected)
            })
        }
        
        // ============ Match ============
        
        Expr::Match { scrutinee, arms } => {
            let scrut_ty = infer(ctx, scrutinee)?;
            
            for arm in arms {
                let bindings = pattern_types(&arm.pattern, &scrut_ty)?;
                ctx.with_bindings(&bindings, |ctx| {
                    if let Some(guard) = &arm.guard {
                        check(ctx, guard, &Type::Prim(PrimType::Bool))?;
                    }
                    check(ctx, &arm.body, expected)
                })?;
            }
            
            Ok(())
        }
        
        // ============ Array (with known element type) ============
        
        Expr::Array(exprs) => {
            match expected {
                Type::Tensor(_, elem) => {
                    for e in exprs {
                        check(ctx, e, elem)?;
                    }
                    Ok(())
                }
                _ => {
                    // Fall back to inference
                    let inferred = infer(ctx, expr)?;
                    subsumes(expected, &inferred)
                }
            }
        }
        
        // ============ Tuple (with known field types) ============
        
        Expr::Tuple(exprs) => {
            match expected {
                Type::Tuple(fields) => {
                    if exprs.len() != fields.len() {
                        return Err(TypeError::TupleArityMismatch {
                            expected: fields.len(),
                            found: exprs.len(),
                        });
                    }
                    for (e, f) in exprs.iter().zip(fields) {
                        check(ctx, e, &f.ty)?;
                    }
                    Ok(())
                }
                _ => {
                    let inferred = infer(ctx, expr)?;
                    subsumes(expected, &inferred)
                }
            }
        }
        
        // ============ Hole ============
        
        Expr::Hole => Ok(()),  // Hole accepts any type
        
        // ============ Default: infer and compare ============
        
        _ => {
            let inferred = infer(ctx, expr)?;
            subsumes(expected, &inferred)
        }
    }
}

/// Check that `actual` type is a subtype of (or equal to) `expected`
fn subsumes(expected: &Type, actual: &Type) -> TypeResult<()> {
    // For now, just use unification
    // Later, add proper subtyping for refinements
    let _subst = unify(expected, actual)?;
    Ok(())
}