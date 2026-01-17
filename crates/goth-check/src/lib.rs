//! Goth Type Checker
//!
//! Bidirectional type checking for the Goth programming language.
//!
//! ## Example
//!
//! ```rust
//! use goth_check::TypeChecker;
//! use goth_parse::prelude::*;
//! use goth_ast::types::PrimType;
//!
//! let expr = parse_expr("1 + 2").unwrap();
//! let resolved = resolve_expr(expr);
//!
//! let mut checker = TypeChecker::new();
//! let ty = checker.infer(&resolved).unwrap();
//! // Type should be I64
//! ```

pub mod error;
pub mod subst;
pub mod unify;
pub mod context;
pub mod builtins;
pub mod infer;
pub mod check;

pub use error::{TypeError, TypeResult};
pub use context::Context;
pub use infer::infer;
pub use check::check;

use goth_ast::expr::Expr;
use goth_ast::types::Type;
use goth_ast::decl::{Module, Decl};

/// Type checker instance
pub struct TypeChecker {
    pub ctx: Context,
}

impl TypeChecker {
    /// Create a new type checker with empty context
    pub fn new() -> Self {
        let mut ctx = Context::new();
        
        // Add built-in primitives
        for prim in ["sqrt", "exp", "ln", "sin", "cos", "tan", 
                     "asin", "acos", "atan", "sinh", "cosh", "tanh",
                     "floor", "ceil", "round", "abs", "length"] {
            if let Some(ty) = builtins::primitive_type(prim) {
                ctx.define_global(prim, ty);
            }
        }
        
        TypeChecker { ctx }
    }

    /// Infer the type of an expression
    pub fn infer(&mut self, expr: &Expr) -> TypeResult<Type> {
        infer::infer(&mut self.ctx, expr)
    }

    /// Check that an expression has the expected type
    pub fn check(&mut self, expr: &Expr, expected: &Type) -> TypeResult<()> {
        check::check(&mut self.ctx, expr, expected)
    }

    /// Type check a module, returning types for each declaration
    pub fn check_module(&mut self, module: &Module) -> TypeResult<Vec<(String, Type)>> {
        let mut results = Vec::new();
        
        for decl in &module.decls {
            match decl {
                Decl::Let(let_decl) => {
                    let ty = if let Some(ann_ty) = &let_decl.type_ {
                        self.check(&let_decl.value, ann_ty)?;
                        ann_ty.clone()
                    } else {
                        self.infer(&let_decl.value)?
                    };
                    self.ctx.define_global(let_decl.name.to_string(), ty.clone());
                    results.push((let_decl.name.to_string(), ty));
                }
                
                Decl::Fn(fn_decl) => {
                    // Function signature is the declared type
                    let ty = fn_decl.signature.clone();
                    
                    // Check body against return type
                    // For λ→ body with signature A → B, we check:
                    //   body : B  (in context with arg : A)
                    if let Type::Fn(arg_ty, ret_ty) = &ty {
                        self.ctx.with_binding(*arg_ty.clone(), |ctx| {
                            check::check(ctx, &fn_decl.body, ret_ty)
                        })?;
                    }
                    
                    self.ctx.define_global(fn_decl.name.to_string(), ty.clone());
                    results.push((fn_decl.name.to_string(), ty));
                }
                
                Decl::Type(_) => {
                    // Type declarations don't produce values
                }
                
                _ => {}
            }
        }
        
        Ok(results)
    }

    /// Define a global binding
    pub fn define(&mut self, name: impl Into<String>, ty: Type) {
        self.ctx.define_global(name, ty);
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_ast::literal::Literal;

    #[test]
    fn test_infer_int() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Lit(Literal::Int(42));
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::I64));
    }

    #[test]
    fn test_infer_float() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Lit(Literal::Float(3.14));
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::F64));
    }

    #[test]
    fn test_infer_bool() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Lit(Literal::True);
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::Bool));
    }

    #[test]
    fn test_infer_addition() {
        use goth_ast::op::BinOp;
        
        let mut checker = TypeChecker::new();
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::I64));
    }

    #[test]
    fn test_check_lambda() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Lam(Box::new(Expr::Idx(0)));  // λ→ ₀
        let ty = Type::func(Type::Prim(PrimType::F64), Type::Prim(PrimType::F64));
        checker.check(&expr, &ty).unwrap();
    }

    #[test]
    fn test_infer_array() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
        ]);
        let ty = checker.infer(&expr).unwrap();
        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape.rank(), 1);
                assert_eq!(*elem, Type::Prim(PrimType::I64));
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_infer_tuple() {
        let mut checker = TypeChecker::new();
        let expr = Expr::Tuple(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::True),
        ]);
        let ty = checker.infer(&expr).unwrap();
        match ty {
            Type::Tuple(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].ty, Type::Prim(PrimType::I64));
                assert_eq!(fields[1].ty, Type::Prim(PrimType::Bool));
            }
            _ => panic!("Expected tuple type"),
        }
    }

    #[test]
    fn test_infer_let() {
        use goth_ast::pattern::Pattern;
        
        let mut checker = TypeChecker::new();
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::Idx(0)),  // x
        };
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::I64));
    }

    #[test]
    fn test_infer_if() {
        let mut checker = TypeChecker::new();
        let expr = Expr::If {
            cond: Box::new(Expr::Lit(Literal::True)),
            then_: Box::new(Expr::Lit(Literal::Int(1))),
            else_: Box::new(Expr::Lit(Literal::Int(2))),
        };
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::I64));
    }

    #[test]
    fn test_type_mismatch() {
        use goth_ast::op::BinOp;
        
        let mut checker = TypeChecker::new();
        let expr = Expr::BinOp(
            BinOp::And,  // Requires Bool operands
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        assert!(checker.infer(&expr).is_err());
    }
}