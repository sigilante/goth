//! # Goth AST
//!
//! Abstract Syntax Tree for the Goth programming language.
//!
//! Goth is an LLM-native programming language featuring:
//! - Dense symbolic notation (APL-influenced)
//! - De Bruijn indices for variable binding
//! - Shape-indexed tensor types
//! - Effect tracking
//! - Interval types for value ranges
//! - Refinement types with pre/postconditions
//!
//! ## Source Formats
//!
//! The AST can be serialized to three formats:
//! - `.goth` - Unicode text (for humans)
//! - `.gast` - JSON AST (for tooling)
//! - `.gbin` - Binary AST (for LLMs and compilation)
//!
//! All three are isomorphic. The compiler accepts any of them.
//!
//! ## Example
//!
//! ```rust
//! use goth_ast::prelude::*;
//!
//! // Build a simple function: ╭─ add : F64 → F64 → F64
//! //                          ╰─ ₀ + ₁
//! let add_fn = FnDecl::simple(
//!     "add",
//!     Type::func_n([Type::f64(), Type::f64()], Type::f64()),
//!     Expr::add(Expr::idx(0), Expr::idx(1)),
//! );
//!
//! // Pretty print
//! let text = pretty::print_fn(&add_fn);
//! println!("{}", text);
//!
//! // Serialize to JSON
//! let module = Module::new(vec![add_fn.into()]);
//! let json = ser::to_json(&module).unwrap();
//! println!("{}", json);
//! ```

pub mod op;
pub mod literal;
pub mod shape;
pub mod effect;
pub mod interval;
pub mod types;
pub mod pattern;
pub mod expr;
pub mod decl;
pub mod pretty;
pub mod ser;

/// Prelude - common imports
pub mod prelude {
    pub use crate::op::{BinOp, UnaryOp, Assoc};
    pub use crate::literal::Literal;
    pub use crate::shape::{Shape, Dim};
    pub use crate::effect::{Effect, Effects};
    pub use crate::interval::{Interval, IntervalSet, Bound, BoundKind};
    pub use crate::types::{Type, PrimType, TupleField, VariantArm, TypeParam, Constraint};
    pub use crate::pattern::Pattern;
    pub use crate::expr::{Expr, MatchArm, FieldAccess, CastKind, DoOp};
    pub use crate::decl::{Module, Decl, FnDecl, TypeDecl, ClassDecl, ImplDecl, LetDecl};
    pub use crate::pretty;
    pub use crate::ser;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_simple_function() {
        // ╭─ add : F64 → F64 → F64
        // ╰─ ₀ + ₁
        let add = FnDecl::simple(
            "add",
            Type::func_n([Type::f64(), Type::f64()], Type::f64()),
            Expr::add(Expr::idx(0), Expr::idx(1)),
        );

        assert_eq!(add.name.as_ref(), "add");
        
        let text = pretty::print_fn(&add);
        assert!(text.contains("add"));
        assert!(text.contains("F64"));
    }

    #[test]
    fn test_factorial() {
        // ╭─ factorial : ℕ → ℕ
        // ╰─ match ₀
        //      0 → 1
        //      n → n × factorial (n - 1)
        let factorial = FnDecl::simple(
            "factorial",
            Type::func(Type::nat(), Type::nat()),
            Expr::match_(
                Expr::idx(0),
                vec![
                    MatchArm::new(Pattern::lit(0i64), Expr::int(1)),
                    MatchArm::new(
                        Pattern::var("n"),
                        Expr::mul(
                            Expr::idx(0),
                            Expr::app(
                                Expr::name("factorial"),
                                Expr::sub(Expr::idx(0), Expr::int(1)),
                            ),
                        ),
                    ),
                ],
            ),
        );

        let text = pretty::print_fn(&factorial);
        assert!(text.contains("factorial"));
    }

    #[test]
    fn test_tensor_type() {
        // [3 4]F64 - 3x4 matrix of floats
        let matrix_type = Type::tensor(
            Shape::concrete(&[3, 4]),
            Type::f64(),
        );
        
        let text = format!("{}", matrix_type);
        assert!(text.contains("[3 4]"));
        assert!(text.contains("F64"));
    }

    #[test]
    fn test_vector_map() {
        // xs ↦ (λ→ ₀ × 2)
        let mapped = Expr::map(
            Expr::name("xs"),
            Expr::lam(Expr::mul(Expr::idx(0), Expr::int(2))),
        );

        let text = format!("{}", mapped);
        assert!(text.contains("↦"));
    }

    #[test]
    fn test_de_bruijn_shift() {
        // λ→ ₀ shifted by 1 should still be λ→ ₀
        // (the index is relative to the lambda)
        let lam = Expr::lam(Expr::idx(0));
        let shifted = lam.shift(0, 1);
        
        match shifted {
            Expr::Lam(body) => match *body {
                Expr::Idx(i) => assert_eq!(i, 0),
                _ => panic!("Expected Idx"),
            },
            _ => panic!("Expected Lam"),
        }
    }

    #[test]
    fn test_de_bruijn_capture() {
        // let x = 5 in λ→ x + ₀
        // The 'x' (which is ₀ in the outer scope) should become ₁ inside the lambda
        let outer_ref = Expr::idx(0); // refers to x
        let shifted = outer_ref.shift(0, 1); // shift for entering lambda
        
        match shifted {
            Expr::Idx(i) => assert_eq!(i, 1),
            _ => panic!("Expected Idx"),
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let module = Module::new(vec![
            FnDecl::simple(
                "id",
                Type::func(Type::var("T"), Type::var("T")),
                Expr::idx(0),
            ).into(),
        ]);

        let json = ser::to_json(&module).unwrap();
        let recovered = ser::from_json(&json).unwrap();
        
        assert_eq!(module, recovered);
    }

    #[test]
    fn test_binary_roundtrip() {
        let module = Module::new(vec![
            FnDecl::simple(
                "const",
                Type::func_n([Type::var("A"), Type::var("B")], Type::var("A")),
                Expr::idx(1), // return first arg (de Bruijn: second from innermost)
            ).into(),
        ]);

        let binary = ser::to_binary(&module).unwrap();
        let recovered = ser::from_binary(&binary).unwrap();
        
        assert_eq!(module, recovered);
    }

    #[test]
    fn test_effects() {
        let io_effect = Effects::single(Effect::Io);
        let rand_effect = Effects::single(Effect::Rand);
        let combined = io_effect.union(&rand_effect);
        
        assert!(combined.contains(&Effect::Io));
        assert!(combined.contains(&Effect::Rand));
        assert!(!combined.is_pure());
    }

    #[test]
    fn test_intervals() {
        let unit = Interval::unit(); // [0..1]
        let positive = Interval::positive(); // (0..∞)
        
        assert!(unit.may_contain_zero());
        assert!(!positive.may_contain_zero());
    }

    #[test]
    fn test_dot_product() {
        // ╭─ dot : [n]F64 → [n]F64 → F64
        // ╰─ ₀ ⊗ ₁ Σ
        let dot = FnDecl::simple(
            "dot",
            Type::func_n(
                [
                    Type::tensor(Shape::symbolic(&["n"]), Type::f64()),
                    Type::tensor(Shape::symbolic(&["n"]), Type::f64()),
                ],
                Type::f64(),
            ),
            Expr::sum(Expr::zip_with(Expr::idx(1), Expr::idx(0))),
        );

        let text = pretty::print_fn(&dot);
        assert!(text.contains("dot"));
    }

    #[test]
    fn test_typeclass() {
        use crate::types::TypeParamKind;
        
        // class Num τ where
        //   (+) : τ → τ → τ
        //   zero : τ
        let num_class = ClassDecl::new(
            "Num",
            TypeParam { name: "τ".into(), kind: TypeParamKind::Type },
        )
        .with_method("add", Type::func_n([Type::var("τ"), Type::var("τ")], Type::var("τ")))
        .with_method("zero", Type::var("τ"));

        assert_eq!(num_class.methods.len(), 2);
    }

    #[test]
    fn test_pattern_binding_count() {
        let wild = Pattern::wildcard();
        let var = Pattern::var("x");
        let tuple = Pattern::tuple(vec![Pattern::var("a"), Pattern::var("b")]);
        let nested = Pattern::tuple(vec![
            Pattern::var("x"),
            Pattern::tuple(vec![Pattern::var("y"), Pattern::wildcard()]),
        ]);

        assert_eq!(wild.binding_count(), 0);
        assert_eq!(var.binding_count(), 1);
        assert_eq!(tuple.binding_count(), 2);
        assert_eq!(nested.binding_count(), 2);
    }

    #[test]
    fn test_disabled_expr() {
        let _enabled = Expr::add(Expr::idx(0), Expr::int(1));
        let disabled = Expr::disabled(Expr::add(Expr::idx(0), Expr::int(2)));
        
        let text = format!("{}", disabled);
        assert!(text.contains("#-"));
        assert!(text.contains("-#"));
    }
}
