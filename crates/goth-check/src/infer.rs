//! Type inference (synthesis mode)

use goth_ast::expr::{Expr, FieldAccess};
use goth_ast::types::{Type, PrimType};
use goth_ast::literal::Literal;
use goth_ast::pattern::Pattern;

use crate::context::Context;
use crate::error::{TypeError, TypeResult};
use crate::check::check;
use crate::unify::unify;
use crate::subst::{Subst, apply_type};
use crate::builtins::{binop_type, unaryop_type, primitive_type};

/// Infer (synthesize) the type of an expression
pub fn infer(ctx: &mut Context, expr: &Expr) -> TypeResult<Type> {
    match expr {
        // ============ Atoms ============
        
        Expr::Idx(i) => {
            ctx.lookup_index(*i)
                .cloned()
                .ok_or(TypeError::UnboundIndex(*i))
        }
        
        Expr::Name(name) => {
            ctx.lookup_global(name)
                .cloned()
                .ok_or_else(|| TypeError::UndefinedName(name.to_string()))
        }
        
        Expr::Lit(lit) => Ok(literal_type(lit)),
        
        Expr::Prim(name) => {
            primitive_type(name)
                .ok_or_else(|| TypeError::UndefinedName(name.to_string()))
        }
        
        // ============ Lambda ============
        
        // Lambda without annotation cannot be inferred
        Expr::Lam(_) | Expr::LamN(_, _) => {
            Err(TypeError::CannotInferLambda)
        }
        
        // ============ Application ============
        
        Expr::App(func, arg) => {
            let func_ty = infer(ctx, func)?;
            infer_app(ctx, &func_ty, arg)
        }
        
        // ============ Let ============
        
        Expr::Let { pattern, value, body } => {
            let val_ty = infer(ctx, value)?;
            let bindings = pattern_types(pattern, &val_ty)?;
            ctx.with_bindings(&bindings, |ctx| infer(ctx, body))
        }
        
        Expr::LetRec { bindings, body } => {
            // For recursive bindings, we need type annotations
            // For now, assume all are lambdas with holes
            let placeholder_types: Vec<Type> = bindings.iter()
                .map(|_| Type::Hole)
                .collect();
            
            ctx.with_bindings(&placeholder_types, |ctx| {
                infer(ctx, body)
            })
        }
        
        // ============ If ============
        
        Expr::If { cond, then_, else_ } => {
            check(ctx, cond, &Type::Prim(PrimType::Bool))?;
            let then_ty = infer(ctx, then_)?;
            let else_ty = infer(ctx, else_)?;
            
            let subst = unify(&then_ty, &else_ty)?;
            Ok(apply_type(&subst, &then_ty))
        }
        
        // ============ Match ============
        
        Expr::Match { scrutinee, arms } => {
            let scrut_ty = infer(ctx, scrutinee)?;
            
            if arms.is_empty() {
                return Err(TypeError::NonExhaustiveMatch);
            }
            
            // Infer type from first arm, check rest against it
            let first_arm = &arms[0];
            let bindings = pattern_types(&first_arm.pattern, &scrut_ty)?;
            let result_ty = ctx.with_bindings(&bindings, |ctx| {
                if let Some(guard) = &first_arm.guard {
                    check(ctx, guard, &Type::Prim(PrimType::Bool))?;
                }
                infer(ctx, &first_arm.body)
            })?;
            
            // Check remaining arms
            for arm in arms.iter().skip(1) {
                let bindings = pattern_types(&arm.pattern, &scrut_ty)?;
                ctx.with_bindings(&bindings, |ctx| {
                    if let Some(guard) = &arm.guard {
                        check(ctx, guard, &Type::Prim(PrimType::Bool))?;
                    }
                    check(ctx, &arm.body, &result_ty)
                })?;
            }
            
            Ok(result_ty)
        }
        
        // ============ Binary/Unary Operations ============

        Expr::BinOp(op, left, right) => {
            use goth_ast::op::BinOp;

            // Special handling for Map and Filter with lambda right operand
            match (op, right.as_ref()) {
                (BinOp::Map, Expr::Lam(body)) => {
                    // arr ↦ λ→ body: infer arr type, use element as lambda arg
                    let left_ty = infer(ctx, left)?;
                    match &left_ty {
                        Type::Tensor(shape, elem_ty) => {
                            // Infer body with element type bound to ₀
                            let ret_ty = ctx.with_binding(*elem_ty.clone(), |ctx| {
                                infer(ctx, body)
                            })?;
                            // Result is tensor with same shape, new element type
                            Ok(Type::Tensor(shape.clone(), Box::new(ret_ty)))
                        }
                        _ => Err(TypeError::NotATensor(left_ty))
                    }
                }

                (BinOp::Filter, Expr::Lam(body)) => {
                    // arr ▸ λ→ body: infer arr type, lambda must return Bool
                    let left_ty = infer(ctx, left)?;
                    match &left_ty {
                        Type::Tensor(_, elem_ty) => {
                            // Infer body, should be Bool
                            let pred_ty = ctx.with_binding(*elem_ty.clone(), |ctx| {
                                infer(ctx, body)
                            })?;
                            check(ctx, &Expr::Lit(Literal::True), &pred_ty)?;
                            // Result has unknown length but same element type
                            Ok(Type::Tensor(
                                goth_ast::shape::Shape(vec![goth_ast::shape::Dim::Var("m".into())]),
                                elem_ty.clone(),
                            ))
                        }
                        _ => Err(TypeError::NotATensor(left_ty))
                    }
                }

                _ => {
                    // Default: infer both sides
                    let left_ty = infer(ctx, left)?;
                    let right_ty = infer(ctx, right)?;
                    binop_type(op.clone(), &left_ty, &right_ty)
                }
            }
        }
        
        Expr::UnaryOp(op, operand) => {
            let operand_ty = infer(ctx, operand)?;
            unaryop_type(*op, &operand_ty)
        }
        
        Expr::Norm(inner) => {
            let inner_ty = infer(ctx, inner)?;
            // Norm of tensor is scalar
            match inner_ty {
                Type::Tensor(_, elem) => Ok(*elem),
                _ => Err(TypeError::NotATensor(inner_ty)),
            }
        }
        
        // ============ Data Construction ============
        
        Expr::Tuple(exprs) => {
            let types: Vec<Type> = exprs.iter()
                .map(|e| infer(ctx, e))
                .collect::<TypeResult<_>>()?;
            Ok(Type::tuple(types))
        }
        
        Expr::Record(fields) => {
            let typed_fields = fields.iter()
                .map(|(name, expr)| {
                    let ty = infer(ctx, expr)?;
                    Ok(goth_ast::types::TupleField {
                        label: Some(name.clone()),
                        ty,
                    })
                })
                .collect::<TypeResult<_>>()?;
            Ok(Type::Tuple(typed_fields))
        }
        
        Expr::Array(exprs) => {
            if exprs.is_empty() {
                // Empty array needs annotation
                return Ok(Type::Tensor(
                    goth_ast::shape::Shape(vec![goth_ast::shape::Dim::Const(0)]),
                    Box::new(Type::Hole),
                ));
            }
            
            let elem_ty = infer(ctx, &exprs[0])?;
            for e in exprs.iter().skip(1) {
                check(ctx, e, &elem_ty)?;
            }
            
            Ok(Type::Tensor(
                goth_ast::shape::Shape(vec![goth_ast::shape::Dim::Const(exprs.len() as u64)]),
                Box::new(elem_ty),
            ))
        }
        
        Expr::ArrayFill { shape, value } => {
            let val_ty = infer(ctx, value)?;
            let dims: Vec<goth_ast::shape::Dim> = shape.iter()
                .map(|e| {
                    // Shape expressions should be integers
                    check(ctx, e, &Type::Prim(PrimType::I64))?;
                    // For now, mark as unknown; could try to evaluate constants
                    Ok(goth_ast::shape::Dim::Var("?".into()))
                })
                .collect::<TypeResult<_>>()?;
            
            Ok(Type::Tensor(goth_ast::shape::Shape(dims), Box::new(val_ty)))
        }
        
        Expr::Variant { constructor, payload } => {
            // Without a type annotation, we can't know the full variant type
            // Return a partial variant type
            let payload_ty = match payload {
                Some(p) => Some(infer(ctx, p)?),
                None => None,
            };
            Ok(Type::Variant(vec![goth_ast::types::VariantArm {
                name: constructor.clone(),
                payload: payload_ty,
            }]))
        }
        
        // ============ Access ============
        
        Expr::Field(base, access) => {
            let base_ty = infer(ctx, base)?;
            field_access_type(&base_ty, access)
        }
        
        Expr::Index(base, indices) => {
            let base_ty = infer(ctx, base)?;
            match base_ty {
                Type::Tensor(shape, elem) => {
                    if indices.len() != shape.rank() {
                        return Err(TypeError::WrongIndexCount {
                            expected: shape.rank(),
                            found: indices.len(),
                        });
                    }
                    // All indices should be integers
                    for idx in indices {
                        check(ctx, idx, &Type::Prim(PrimType::I64))?;
                    }
                    Ok(*elem)
                }
                _ => Err(TypeError::CannotIndex(base_ty))
            }
        }
        
        Expr::Slice { array, start, end } => {
            let arr_ty = infer(ctx, array)?;
            match arr_ty {
                Type::Tensor(_, elem) => {
                    if let Some(s) = start {
                        check(ctx, s, &Type::Prim(PrimType::I64))?;
                    }
                    if let Some(e) = end {
                        check(ctx, e, &Type::Prim(PrimType::I64))?;
                    }
                    // Result has unknown length
                    Ok(Type::Tensor(
                        goth_ast::shape::Shape(vec![goth_ast::shape::Dim::Var("?".into())]),
                        elem,
                    ))
                }
                _ => Err(TypeError::CannotIndex(arr_ty))
            }
        }
        
        // ============ Annotation ============
        
        Expr::Annot(expr, ty) => {
            check(ctx, expr, ty)?;
            Ok(ty.clone())
        }
        
        Expr::Cast { expr, target, .. } => {
            let _ = infer(ctx, expr)?;
            Ok(target.clone())
        }
        
        // ============ Special ============
        
        Expr::Hole => Ok(Type::Hole),
        
        Expr::Disabled(_) => Ok(Type::unit()),
        
        _ => {
            // Fallback for unhandled cases
            Ok(Type::Hole)
        }
    }
}

/// Infer result type of function application
fn infer_app(ctx: &mut Context, func_ty: &Type, arg: &Expr) -> TypeResult<Type> {
    match func_ty {
        Type::Fn(param_ty, ret_ty) => {
            check(ctx, arg, param_ty)?;
            Ok(*ret_ty.clone())
        }
        
        Type::Forall(params, body) => {
            // Instantiate with fresh type variables
            // For now, just use holes
            let mut subst = Subst::new();
            for p in params {
                match p.kind {
                    goth_ast::types::TypeParamKind::Type => {
                        subst.types.insert(p.name.to_string(), Type::Hole);
                    }
                    goth_ast::types::TypeParamKind::Shape => {
                        subst.shapes.insert(p.name.to_string(), goth_ast::shape::Dim::Var("?".into()));
                    }
                    _ => {}
                }
            }
            let instantiated = apply_type(&subst, body);
            infer_app(ctx, &instantiated, arg)
        }
        
        _ => Err(TypeError::NotAFunction(func_ty.clone()))
    }
}

/// Get type of a literal
fn literal_type(lit: &Literal) -> Type {
    match lit {
        Literal::Int(_) => Type::Prim(PrimType::I64),
        Literal::Float(_) => Type::Prim(PrimType::F64),
        Literal::True | Literal::False => Type::Prim(PrimType::Bool),
        Literal::Char(_) => Type::Prim(PrimType::Char),
        Literal::String(_) => Type::Tensor(
            goth_ast::shape::Shape(vec![goth_ast::shape::Dim::Var("?".into())]),
            Box::new(Type::Prim(PrimType::Char)),
        ),
        Literal::Unit => Type::unit(),
    }
}

/// Extract types for pattern bindings
pub fn pattern_types(pattern: &Pattern, ty: &Type) -> TypeResult<Vec<Type>> {
    match pattern {
        Pattern::Wildcard => Ok(vec![]),
        
        Pattern::Var(_) => Ok(vec![ty.clone()]),
        
        Pattern::Lit(_) => Ok(vec![]),
        
        Pattern::Tuple(pats) => {
            match ty {
                Type::Tuple(fields) => {
                    if pats.len() != fields.len() {
                        return Err(TypeError::TupleArityMismatch {
                            expected: pats.len(),
                            found: fields.len(),
                        });
                    }
                    let mut types = Vec::new();
                    for (pat, field) in pats.iter().zip(fields) {
                        types.extend(pattern_types(pat, &field.ty)?);
                    }
                    Ok(types)
                }
                _ => Err(TypeError::PatternMismatch {
                    expected: "tuple".to_string(),
                    found: ty.clone(),
                })
            }
        }
        
        Pattern::Array(pats) => {
            match ty {
                Type::Tensor(_, elem) => {
                    let mut types = Vec::new();
                    for pat in pats {
                        types.extend(pattern_types(pat, elem)?);
                    }
                    Ok(types)
                }
                _ => Err(TypeError::PatternMismatch {
                    expected: "array".to_string(),
                    found: ty.clone(),
                })
            }
        }
        
        Pattern::ArraySplit { head, tail } => {
            match ty {
                Type::Tensor(_, elem) => {
                    let mut types = Vec::new();
                    for pat in head {
                        types.extend(pattern_types(pat, elem)?);
                    }
                    // Tail is an array
                    types.extend(pattern_types(tail, ty)?);
                    Ok(types)
                }
                _ => Err(TypeError::PatternMismatch {
                    expected: "array".to_string(),
                    found: ty.clone(),
                })
            }
        }
        
        Pattern::Variant { payload: Some(p), .. } => {
            // Need variant type info to know payload type
            pattern_types(p, &Type::Hole)
        }
        
        Pattern::Variant { payload: None, .. } => Ok(vec![]),
        
        Pattern::Typed(inner, _) => {
            pattern_types(inner, ty)
        }
        
        Pattern::Or(p1, _) => {
            // Both branches should bind same names
            pattern_types(p1, ty)
        }
        
        Pattern::Guard(inner, _) => {
            pattern_types(inner, ty)
        }
    }
}

/// Get type of field access
fn field_access_type(base_ty: &Type, access: &FieldAccess) -> TypeResult<Type> {
    match (base_ty, access) {
        (Type::Tuple(fields), FieldAccess::Index(i)) => {
            let idx = *i as usize;
            if idx < fields.len() {
                Ok(fields[idx].ty.clone())
            } else {
                Err(TypeError::FieldNotFound {
                    field: i.to_string(),
                    ty: base_ty.clone(),
                })
            }
        }
        
        (Type::Tuple(fields), FieldAccess::Named(name)) => {
            for field in fields {
                if let Some(label) = &field.label {
                    if label.as_ref() == name.as_ref() {
                        return Ok(field.ty.clone());
                    }
                }
            }
            Err(TypeError::FieldNotFound {
                field: name.to_string(),
                ty: base_ty.clone(),
            })
        }
        
        _ => Err(TypeError::FieldNotFound {
            field: format!("{:?}", access),
            ty: base_ty.clone(),
        })
    }
}