//! Type and shape substitutions

use std::collections::HashMap;
use goth_ast::types::Type;
use goth_ast::shape::{Shape, Dim};

/// Substitution mapping type variables to types
#[derive(Debug, Clone, Default)]
pub struct Subst {
    /// Type variable substitutions: α → Type
    pub types: HashMap<String, Type>,
    /// Shape variable substitutions: n → Dim
    pub shapes: HashMap<String, Dim>,
}

impl Subst {
    pub fn new() -> Self {
        Subst::default()
    }

    /// Create substitution with single type binding
    pub fn singleton_type(var: impl Into<String>, ty: Type) -> Self {
        let mut s = Subst::new();
        s.types.insert(var.into(), ty);
        s
    }

    /// Create substitution with single shape binding
    pub fn singleton_shape(var: impl Into<String>, dim: Dim) -> Self {
        let mut s = Subst::new();
        s.shapes.insert(var.into(), dim);
        s
    }

    /// Check if substitution is empty
    pub fn is_empty(&self) -> bool {
        self.types.is_empty() && self.shapes.is_empty()
    }

    /// Compose two substitutions: apply s2 after s1
    /// (s2 ∘ s1)(x) = s2(s1(x))
    pub fn compose(s1: Subst, s2: Subst) -> Subst {
        let mut result = Subst::new();
        
        // Apply s2 to all bindings in s1
        for (var, ty) in s1.types {
            result.types.insert(var, apply_type(&s2, &ty));
        }
        
        // Add all bindings from s2 (overwriting if present)
        for (var, ty) in &s2.types {
            result.types.insert(var.clone(), ty.clone());
        }

        // Same for shapes
        for (var, dim) in s1.shapes {
            result.shapes.insert(var, apply_dim(&s2, &dim));
        }
        for (var, dim) in &s2.shapes {
            result.shapes.insert(var.clone(), dim.clone());
        }

        result
    }
}

/// Apply substitution to a type
pub fn apply_type(subst: &Subst, ty: &Type) -> Type {
    match ty {
        Type::Var(v) => {
            subst.types.get(v.as_ref()).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Fn(arg, ret) => {
            Type::Fn(
                Box::new(apply_type(subst, arg)),
                Box::new(apply_type(subst, ret)),
            )
        }
        Type::Tensor(shape, elem) => {
            Type::Tensor(
                apply_shape(subst, shape),
                Box::new(apply_type(subst, elem)),
            )
        }
        Type::Tuple(fields) => {
            Type::Tuple(
                fields.iter().map(|f| goth_ast::types::TupleField {
                    label: f.label.clone(),
                    ty: apply_type(subst, &f.ty),
                }).collect()
            )
        }
        Type::Forall(params, body) => {
            // Don't substitute bound variables
            let mut inner_subst = subst.clone();
            for p in params {
                inner_subst.types.remove(p.name.as_ref());
            }
            Type::Forall(params.clone(), Box::new(apply_type(&inner_subst, body)))
        }
        Type::Option(inner) => {
            Type::Option(Box::new(apply_type(subst, inner)))
        }
        Type::Uncertain(val, unc) => {
            Type::Uncertain(
                Box::new(apply_type(subst, val)),
                Box::new(apply_type(subst, unc))
            )
        }
        Type::Refinement { name, base, predicate } => {
            Type::Refinement {
                name: name.clone(),
                base: Box::new(apply_type(subst, base)),
                predicate: predicate.clone(), // Predicates need separate handling
            }
        }
        // Primitives and holes don't change
        Type::Prim(_) | Type::Hole => ty.clone(),
        // TODO: Handle remaining cases
        _ => ty.clone(),
    }
}

/// Apply substitution to a shape
pub fn apply_shape(subst: &Subst, shape: &Shape) -> Shape {
    Shape(shape.0.iter().map(|d| apply_dim(subst, d)).collect())
}

/// Apply substitution to a dimension
pub fn apply_dim(subst: &Subst, dim: &Dim) -> Dim {
    match dim {
        Dim::Var(v) => {
            subst.shapes.get(v.as_ref()).cloned().unwrap_or_else(|| dim.clone())
        }
        Dim::Const(_) => dim.clone(),
        Dim::BinOp(l, op, r) => {
            Dim::BinOp(
                Box::new(apply_dim(subst, l)),
                *op,
                Box::new(apply_dim(subst, r)),
            )
        }
    }
}