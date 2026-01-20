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
pub mod shapes;

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
                     "floor", "ceil", "round", "abs", "length",
                     "dot", "·", "matmul", "transpose", "⍉", "norm",
                     "iota", "ι", "⍳", "range",
                     // Type conversions
                     "toInt", "toFloat", "toBool", "toChar", "toString", "chars",
                     "parseInt", "parseFloat",
                     // Aggregation
                     "sum", "Σ", "prod", "Π",
                     // Array operations
                     "reverse", "take", "drop", "concat", "⧺",
                     // I/O
                     "print", "readLine"] {
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

                    // Define function globally BEFORE checking body (enables recursion)
                    self.ctx.define_global(fn_decl.name.to_string(), ty.clone());

                    // Check body against return type
                    // For multi-argument functions F → F → F, we need to:
                    // 1. Collect all argument types
                    // 2. Push them all onto the context
                    // 3. Check body against final return type

                    let mut arg_types = Vec::new();
                    let mut current = &ty;

                    // Collect all argument types by traversing the chain
                    while let Type::Fn(arg_ty, ret_ty) = current {
                        arg_types.push(*arg_ty.clone());
                        current = ret_ty;
                    }

                    // current is now the final return type
                    let final_ret_ty = current;

                    // Check body with all arguments in context
                    self.ctx.with_bindings(&arg_types, |ctx| {
                        check::check(ctx, &fn_decl.body, final_ret_ty)
                    })?;

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
    use goth_ast::op::UnaryOp;
    use goth_ast::op::BinOp;

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
        let mut checker = TypeChecker::new();
        let expr = Expr::BinOp(
            BinOp::And,  // Requires Bool operands
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        assert!(checker.infer(&expr).is_err());
    }

    #[test]
    fn test_infer_sqrt() {
        let mut checker = TypeChecker::new();
        let expr = Expr::UnaryOp(
            UnaryOp::Sqrt,
            Box::new(Expr::Lit(Literal::Float(16.0))),
        );
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::F64));
    }

    #[test]
    fn test_infer_floor() {
        let mut checker = TypeChecker::new();
        let expr = Expr::UnaryOp(
            UnaryOp::Floor,
            Box::new(Expr::Lit(Literal::Float(3.7))),
        );
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::F64));
    }

    #[test]
    fn test_infer_ceil() {
        let mut checker = TypeChecker::new();
        let expr = Expr::UnaryOp(
            UnaryOp::Ceil,
            Box::new(Expr::Lit(Literal::Float(3.2))),
        );
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::F64));
    }

    #[test]
    fn test_infer_uncertain() {
        let mut checker = TypeChecker::new();
        let expr = Expr::BinOp(
            BinOp::PlusMinus,
            Box::new(Expr::Lit(Literal::Float(10.5))),
            Box::new(Expr::Lit(Literal::Float(0.3))),
        );
        let ty = checker.infer(&expr).unwrap();
        match ty {
            Type::Uncertain(val, unc) => {
                assert_eq!(*val, Type::Prim(PrimType::F64));
                assert_eq!(*unc, Type::Prim(PrimType::F64));
            }
            _ => panic!("Expected Uncertain type, got {:?}", ty),
        }
    }

    #[test]
    fn test_check_multi_arg_function() {
        use goth_ast::decl::{Module, Decl, FnDecl};
        use goth_ast::effect::Effects;

        let mut checker = TypeChecker::new();

        // Create: add : I64 → I64 → I64, body: ₀ + ₁
        let fn_type = Type::func(
            Type::Prim(PrimType::I64),
            Type::func(
                Type::Prim(PrimType::I64),
                Type::Prim(PrimType::I64)
            )
        );

        let body = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),  // ₀
            Box::new(Expr::Idx(1)),  // ₁
        );

        let fn_decl = FnDecl {
            name: "add".into(),
            signature: fn_type.clone(),
            effects: Effects::pure(),
            body,
            preconditions: vec![],
            postconditions: vec![],
            constraints: vec![],
            type_params: vec![],
        };
        
        let module = Module {
            decls: vec![Decl::Fn(fn_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module);
        assert!(result.is_ok());
        
        // Verify function was added to context
        let ty = checker.ctx.lookup_global("add").unwrap();
        assert_eq!(ty, &fn_type);
    }

    #[test]
    fn test_check_three_arg_function() {
        use goth_ast::decl::{Module, Decl, FnDecl};
        use goth_ast::effect::Effects;

        let mut checker = TypeChecker::new();

        // Create: add3 : I64 → I64 → I64 → I64, body: ₀ + ₁ + ₂
        let fn_type = Type::func(
            Type::Prim(PrimType::I64),
            Type::func(
                Type::Prim(PrimType::I64),
                Type::func(
                    Type::Prim(PrimType::I64),
                    Type::Prim(PrimType::I64)
                )
            )
        );

        let body = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),  // ₀
                Box::new(Expr::Idx(1)),  // ₁
            )),
            Box::new(Expr::Idx(2)),  // ₂
        );

        let fn_decl = FnDecl {
            name: "add3".into(),
            signature: fn_type.clone(),
            effects: Effects::pure(),
            body,
            preconditions: vec![],
            postconditions: vec![],
            constraints: vec![],
            type_params: vec![],
        };
        
        let module = Module {
            decls: vec![Decl::Fn(fn_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_arg_with_operators() {
        use goth_ast::decl::{Module, Decl, FnDecl};
        use goth_ast::effect::Effects;

        let mut checker = TypeChecker::new();

        // Create: pythag : F64 → F64 → F64, body: √(₀ × ₀ + ₁ × ₁)
        let fn_type = Type::func(
            Type::Prim(PrimType::F64),
            Type::func(
                Type::Prim(PrimType::F64),
                Type::Prim(PrimType::F64)
            )
        );

        let body = Expr::UnaryOp(
            UnaryOp::Sqrt,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Idx(0)),  // ₀
                    Box::new(Expr::Idx(0)),  // ₀
                )),
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Idx(1)),  // ₁
                    Box::new(Expr::Idx(1)),  // ₁
                )),
            )),
        );

        let fn_decl = FnDecl {
            name: "pythag".into(),
            signature: fn_type.clone(),
            effects: Effects::pure(),
            body,
            preconditions: vec![],
            postconditions: vec![],
            constraints: vec![],
            type_params: vec![],
        };
        
        let module = Module {
            decls: vec![Decl::Fn(fn_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module);
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_multi_arg_wrong_arity() {
        use goth_ast::decl::{Module, Decl, FnDecl};
        use goth_ast::effect::Effects;

        let mut checker = TypeChecker::new();

        // Create: bad : I64 → I64 → I64, body: ₀ + ₂ (₂ out of bounds!)
        let fn_type = Type::func(
            Type::Prim(PrimType::I64),
            Type::func(
                Type::Prim(PrimType::I64),
                Type::Prim(PrimType::I64)
            )
        );

        let body = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),  // ₀
            Box::new(Expr::Idx(2)),  // ₂ - should be out of bounds!
        );

        let fn_decl = FnDecl {
            name: "bad".into(),
            signature: fn_type.clone(),
            effects: Effects::pure(),
            body,
            preconditions: vec![],
            postconditions: vec![],
            constraints: vec![],
            type_params: vec![],
        };
        
        let module = Module {
            decls: vec![Decl::Fn(fn_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module);
        // This should fail - ₂ is out of bounds for 2-arg function
        assert!(result.is_err());
    }

    #[test]
    fn test_let_declaration() {
        use goth_ast::decl::{Module, Decl, LetDecl};
        
        let mut checker = TypeChecker::new();
        
        let let_decl = LetDecl {
            name: "x".into(),
            type_: None,
            value: Expr::Lit(Literal::Int(42)),
        };
        
        let module = Module {
            decls: vec![Decl::Let(let_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "x");
        assert_eq!(result[0].1, Type::Prim(PrimType::I64));
    }

    #[test]
    fn test_let_with_type_annotation() {
        use goth_ast::decl::{Module, Decl, LetDecl};
        
        let mut checker = TypeChecker::new();
        
        let let_decl = LetDecl {
            name: "x".into(),
            type_: Some(Type::Prim(PrimType::I64)),
            value: Expr::Lit(Literal::Int(42)),
        };
        
        let module = Module {
            decls: vec![Decl::Let(let_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_let_with_wrong_type_annotation() {
        use goth_ast::decl::{Module, Decl, LetDecl};
        
        let mut checker = TypeChecker::new();
        
        let let_decl = LetDecl {
            name: "x".into(),
            type_: Some(Type::Prim(PrimType::Bool)),  // Wrong!
            value: Expr::Lit(Literal::Int(42)),       // This is I64
        };
        
        let module = Module {
            decls: vec![Decl::Let(let_decl)],
            name: Some("test".into()),
        };
        
        let result = checker.check_module(&module);
        assert!(result.is_err());
    }

    #[test]
    fn test_function_partial_application() {
        let mut checker = TypeChecker::new();
        
        // Define: add : I64 → I64 → I64
        let add_type = Type::func(
            Type::Prim(PrimType::I64),
            Type::func(
                Type::Prim(PrimType::I64),
                Type::Prim(PrimType::I64)
            )
        );
        checker.define("add", add_type);
        
        // Expression: add 5
        let expr = Expr::App(
            Box::new(Expr::Name("add".into())),
            Box::new(Expr::Lit(Literal::Int(5))),
        );
        
        let ty = checker.infer(&expr).unwrap();
        
        // Should be: I64 → I64
        match ty {
            Type::Fn(arg, ret) => {
                assert_eq!(*arg, Type::Prim(PrimType::I64));
                assert_eq!(*ret, Type::Prim(PrimType::I64));
            }
            _ => panic!("Expected function type, got {:?}", ty),
        }
    }

    #[test]
    fn test_comparison_operators() {
        let mut checker = TypeChecker::new();
        
        for op in [BinOp::Lt, BinOp::Gt, BinOp::Leq, BinOp::Geq, BinOp::Eq, BinOp::Neq] {
            let expr = Expr::BinOp(
                op.clone(),
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(10))),
            );
            
            let ty = checker.infer(&expr).unwrap();
            assert_eq!(ty, Type::Prim(PrimType::Bool), "Failed for operator {:?}", op);
        }
    }

    #[test]
    fn test_logical_operators() {
        let mut checker = TypeChecker::new();
        
        for op in [BinOp::And, BinOp::Or] {
            let expr = Expr::BinOp(
                op.clone(),
                Box::new(Expr::Lit(Literal::True)),
                Box::new(Expr::Lit(Literal::False)),
            );
            
            let ty = checker.infer(&expr).unwrap();
            assert_eq!(ty, Type::Prim(PrimType::Bool), "Failed for operator {:?}", op);
        }
    }

    #[test]
    fn test_negation() {
        let mut checker = TypeChecker::new();
        
        let expr = Expr::UnaryOp(
            UnaryOp::Neg,
            Box::new(Expr::Lit(Literal::Int(5))),
        );
        
        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::I64));
    }

    #[test]
    fn test_not() {
        let mut checker = TypeChecker::new();

        let expr = Expr::UnaryOp(
            UnaryOp::Not,
            Box::new(Expr::Lit(Literal::True)),
        );

        let ty = checker.infer(&expr).unwrap();
        assert_eq!(ty, Type::Prim(PrimType::Bool));
    }

    // ============================================================
    // Shape Checking Tests
    // ============================================================

    #[test]
    fn test_tensor_add_same_shape() {
        // [3]F64 + [3]F64 should work
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        let arr1 = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);
        let arr2 = Expr::Array(vec![
            Expr::Lit(Literal::Float(4.0)),
            Expr::Lit(Literal::Float(5.0)),
            Expr::Lit(Literal::Float(6.0)),
        ]);

        let expr = Expr::BinOp(BinOp::Add, Box::new(arr1), Box::new(arr2));
        let ty = checker.infer(&expr).unwrap();

        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape, Shape(vec![Dim::Const(3)]));
                assert_eq!(*elem, Type::Prim(PrimType::F64));
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_tensor_add_different_shape() {
        // [3]F64 + [5]F64 should FAIL
        let mut checker = TypeChecker::new();

        let arr1 = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);
        let arr2 = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
            Expr::Lit(Literal::Float(4.0)),
            Expr::Lit(Literal::Float(5.0)),
        ]);

        let expr = Expr::BinOp(BinOp::Add, Box::new(arr1), Box::new(arr2));
        let result = checker.infer(&expr);

        assert!(result.is_err(), "Should fail: [3] + [5] shape mismatch");
    }

    #[test]
    fn test_tensor_mul_same_shape() {
        // Element-wise multiplication: [2]F64 × [2]F64 → [2]F64
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        let arr1 = Expr::Array(vec![
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);
        let arr2 = Expr::Array(vec![
            Expr::Lit(Literal::Float(4.0)),
            Expr::Lit(Literal::Float(5.0)),
        ]);

        let expr = Expr::BinOp(BinOp::Mul, Box::new(arr1), Box::new(arr2));
        let ty = checker.infer(&expr).unwrap();

        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape, Shape(vec![Dim::Const(2)]));
                assert_eq!(*elem, Type::Prim(PrimType::F64));
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_tensor_scalar_broadcast() {
        // scalar + [3]F64 should work (broadcasting)
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        let scalar = Expr::Lit(Literal::Float(10.0));
        let arr = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);

        let expr = Expr::BinOp(BinOp::Add, Box::new(scalar), Box::new(arr));
        let ty = checker.infer(&expr).unwrap();

        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape, Shape(vec![Dim::Const(3)]));
                assert_eq!(*elem, Type::Prim(PrimType::F64));
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_tensor_function_type_with_shapes() {
        // Test that we can define functions with shape-annotated types
        use goth_ast::decl::{Module, Decl, FnDecl};
        use goth_ast::shape::{Shape, Dim};
        use goth_ast::effect::Effects;

        let mut checker = TypeChecker::new();

        // Define: id_vec : [n]F64 → [n]F64 (identity on vectors)
        // Body: ₀ (return the argument)
        let param_shape = Shape(vec![Dim::Var("n".into())]);
        let param_type = Type::Tensor(param_shape.clone(), Box::new(Type::Prim(PrimType::F64)));
        let ret_type = Type::Tensor(param_shape, Box::new(Type::Prim(PrimType::F64)));

        let fn_type = Type::func(param_type, ret_type);

        let fn_decl = FnDecl {
            name: "id_vec".into(),
            signature: fn_type.clone(),
            effects: Effects::pure(),
            body: Expr::Idx(0), // λ→ ₀
            preconditions: vec![],
            postconditions: vec![],
            constraints: vec![],
            type_params: vec![],
        };

        let module = Module {
            decls: vec![Decl::Fn(fn_decl)],
            name: Some("test".into()),
        };

        let result = checker.check_module(&module);
        assert!(result.is_ok(), "id_vec : [n]F64 → [n]F64 should type-check");
    }

    #[test]
    fn test_tensor_function_shape_mismatch() {
        // Test that shape mismatches in function signatures are caught
        use goth_ast::decl::{Module, Decl, FnDecl};
        use goth_ast::shape::{Shape, Dim};
        use goth_ast::effect::Effects;

        let mut checker = TypeChecker::new();

        // Define: bad : [3]F64 → [5]F64
        // Body: ₀ (return the argument - but shapes don't match!)
        let param_type = Type::Tensor(
            Shape(vec![Dim::Const(3)]),
            Box::new(Type::Prim(PrimType::F64)),
        );
        let ret_type = Type::Tensor(
            Shape(vec![Dim::Const(5)]),
            Box::new(Type::Prim(PrimType::F64)),
        );

        let fn_type = Type::func(param_type, ret_type);

        let fn_decl = FnDecl {
            name: "bad".into(),
            signature: fn_type.clone(),
            effects: Effects::pure(),
            body: Expr::Idx(0), // λ→ ₀
            preconditions: vec![],
            postconditions: vec![],
            constraints: vec![],
            type_params: vec![],
        };

        let module = Module {
            decls: vec![Decl::Fn(fn_decl)],
            name: Some("test".into()),
        };

        let result = checker.check_module(&module);
        assert!(result.is_err(), "bad : [3]F64 → [5]F64 with body ₀ should fail");
    }

    #[test]
    fn test_tensor_sum_reduction() {
        // Σ [3]F64 → F64 (sum reduces tensor to scalar)
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        let arr = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);

        let expr = Expr::UnaryOp(UnaryOp::Sum, Box::new(arr));
        let ty = checker.infer(&expr).unwrap();

        // Sum should return scalar F64
        assert_eq!(ty, Type::Prim(PrimType::F64));
    }

    #[test]
    fn test_tensor_concat_shapes() {
        // concat [2]F64 [3]F64 → [5]F64 (or [2+3]F64)
        use goth_ast::shape::{Shape, Dim, DimOp};

        let mut checker = TypeChecker::new();

        let arr1 = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
        ]);
        let arr2 = Expr::Array(vec![
            Expr::Lit(Literal::Float(3.0)),
            Expr::Lit(Literal::Float(4.0)),
            Expr::Lit(Literal::Float(5.0)),
        ]);

        let expr = Expr::BinOp(BinOp::Concat, Box::new(arr1), Box::new(arr2));
        let ty = checker.infer(&expr).unwrap();

        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape.rank(), 1);
                assert_eq!(*elem, Type::Prim(PrimType::F64));
                // Shape should be [2+3] = BinOp(Const(2), Add, Const(3))
                match &shape.0[0] {
                    Dim::BinOp(l, DimOp::Add, r) => {
                        assert_eq!(**l, Dim::Const(2));
                        assert_eq!(**r, Dim::Const(3));
                    }
                    _ => panic!("Expected BinOp dimension for concat"),
                }
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_shape_variable_unification() {
        // Test that shape variables unify correctly
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        // Define a function with shape variable: double : [n]F64 → [n]F64
        let n_shape = Shape(vec![Dim::Var("n".into())]);
        let fn_type = Type::func(
            Type::Tensor(n_shape.clone(), Box::new(Type::Prim(PrimType::F64))),
            Type::Tensor(n_shape, Box::new(Type::Prim(PrimType::F64))),
        );
        checker.define("double", fn_type);

        // Apply to [3]F64 array
        let arr = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);

        let expr = Expr::App(
            Box::new(Expr::Name("double".into())),
            Box::new(arr),
        );

        let ty = checker.infer(&expr).unwrap();

        // Result should be [3]F64 (n unified with 3)
        match ty {
            Type::Tensor(shape, elem) => {
                // Note: The shape might still have the variable if we don't
                // apply the substitution, but it should type-check
                assert_eq!(shape.rank(), 1);
                assert_eq!(*elem, Type::Prim(PrimType::F64));
            }
            _ => panic!("Expected tensor type, got {:?}", ty),
        }
    }

    #[test]
    fn test_zip_same_shape() {
        // zip [3]F64 [3]I64 → [3]⟨F64, I64⟩
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        let arr1 = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);
        let arr2 = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
        ]);

        let expr = Expr::BinOp(BinOp::ZipWith, Box::new(arr1), Box::new(arr2));
        let ty = checker.infer(&expr).unwrap();

        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape, Shape(vec![Dim::Const(3)]));
                // elem should be a tuple of (F64, I64)
                match elem.as_ref() {
                    Type::Tuple(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].ty, Type::Prim(PrimType::F64));
                        assert_eq!(fields[1].ty, Type::Prim(PrimType::I64));
                    }
                    _ => panic!("Expected tuple element type"),
                }
            }
            _ => panic!("Expected tensor type"),
        }
    }

    #[test]
    fn test_zip_different_shape() {
        // zip [3]F64 [5]I64 should FAIL
        let mut checker = TypeChecker::new();

        let arr1 = Expr::Array(vec![
            Expr::Lit(Literal::Float(1.0)),
            Expr::Lit(Literal::Float(2.0)),
            Expr::Lit(Literal::Float(3.0)),
        ]);
        let arr2 = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
            Expr::Lit(Literal::Int(4)),
            Expr::Lit(Literal::Int(5)),
        ]);

        let expr = Expr::BinOp(BinOp::ZipWith, Box::new(arr1), Box::new(arr2));
        let result = checker.infer(&expr);

        assert!(result.is_err(), "zip [3] [5] should fail due to shape mismatch");
    }

    #[test]
    fn test_nested_tensor_shape() {
        // [[1,2], [3,4]] should be [2][2]I64 (2x2 matrix)
        use goth_ast::shape::{Shape, Dim};

        let mut checker = TypeChecker::new();

        let row1 = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
        ]);
        let row2 = Expr::Array(vec![
            Expr::Lit(Literal::Int(3)),
            Expr::Lit(Literal::Int(4)),
        ]);
        let matrix = Expr::Array(vec![row1, row2]);

        let ty = checker.infer(&matrix).unwrap();

        match ty {
            Type::Tensor(outer_shape, inner) => {
                assert_eq!(outer_shape, Shape(vec![Dim::Const(2)]));
                match inner.as_ref() {
                    Type::Tensor(inner_shape, elem) => {
                        assert_eq!(*inner_shape, Shape(vec![Dim::Const(2)]));
                        assert_eq!(**elem, Type::Prim(PrimType::I64));
                    }
                    _ => panic!("Expected nested tensor"),
                }
            }
            _ => panic!("Expected tensor type"),
        }
    }
}