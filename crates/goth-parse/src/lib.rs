//! # Goth Parse
//!
//! Parser for the Goth programming language.
//!
//! ## Example
//!
//! ```rust
//! use goth_parse::prelude::*;
//!
//! let expr = parse_expr("λ→ ₀ + 1").unwrap();
//! // Returns: Lam(BinOp(Add, Idx(0), Lit(Int(1))))
//! ```

pub mod lexer;
pub mod parser;
pub mod resolve;
pub mod loader;

pub mod prelude {
    pub use crate::lexer::{Token, Lexer, Loc, Spanned};
    pub use crate::parser::{Parser, ParseError, ParseResult, parse_expr, parse_type, parse_pattern, parse_module};
    pub use crate::resolve::{resolve_expr, resolve_module};
    pub use crate::loader::{Loader, LoadError, LoadResult, load_file, load_source, save_file};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use goth_ast::expr::Expr;
    use goth_ast::literal::Literal;
    use goth_ast::op::BinOp;
    use goth_ast::pattern::Pattern;
    use goth_ast::types::{Type, PrimType};

    // ============ Lexer Tests ============

    #[test]
    fn test_lex_arithmetic() {
        let mut lex = Lexer::new("1 + 2 * 3");
        assert_eq!(lex.next(), Some(Token::Int(1)));
        assert_eq!(lex.next(), Some(Token::Plus));
        assert_eq!(lex.next(), Some(Token::Int(2)));
        assert_eq!(lex.next(), Some(Token::Star));
        assert_eq!(lex.next(), Some(Token::Int(3)));
    }

    #[test]
    fn test_lex_lambda() {
        let mut lex = Lexer::new("λ→ ₀");
        assert_eq!(lex.next(), Some(Token::Lambda));
        assert_eq!(lex.next(), Some(Token::Arrow));
        assert_eq!(lex.next(), Some(Token::Index(0)));
    }

    #[test]
    fn test_lex_ascii_fallback() {
        let mut lex = Lexer::new("\\-> _0");
        assert_eq!(lex.next(), Some(Token::Lambda));
        assert_eq!(lex.next(), Some(Token::Arrow));
        assert_eq!(lex.next(), Some(Token::Index(0)));
    }

    // ============ Expression Parsing ============

    #[test]
    fn test_parse_int() {
        let expr = parse_expr("42").unwrap();
        assert!(matches!(expr, Expr::Lit(Literal::Int(42))));
    }

    #[test]
    fn test_parse_float() {
        let expr = parse_expr("3.14").unwrap();
        assert!(matches!(expr, Expr::Lit(Literal::Float(f)) if (f - 3.14).abs() < 0.001));
    }

    #[test]
    fn test_parse_bool() {
        assert!(matches!(parse_expr("⊤").unwrap(), Expr::Lit(Literal::True)));
        assert!(matches!(parse_expr("true").unwrap(), Expr::Lit(Literal::True)));
        assert!(matches!(parse_expr("⊥").unwrap(), Expr::Lit(Literal::False)));
        assert!(matches!(parse_expr("false").unwrap(), Expr::Lit(Literal::False)));
    }

    #[test]
    fn test_parse_string() {
        let expr = parse_expr(r#""hello""#).unwrap();
        match expr {
            Expr::Lit(Literal::String(s)) => assert_eq!(s.as_ref(), "hello"),
            _ => panic!("Expected string literal"),
        }
    }

    #[test]
    fn test_parse_index() {
        assert!(matches!(parse_expr("₀").unwrap(), Expr::Idx(0)));
        assert!(matches!(parse_expr("₁").unwrap(), Expr::Idx(1)));
        assert!(matches!(parse_expr("_0").unwrap(), Expr::Idx(0)));
        assert!(matches!(parse_expr("_10").unwrap(), Expr::Idx(10)));
    }

    #[test]
    fn test_parse_name() {
        match parse_expr("foo").unwrap() {
            Expr::Name(n) => assert_eq!(n.as_ref(), "foo"),
            _ => panic!("Expected name"),
        }
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_expr("1 + 2").unwrap();
        match expr {
            Expr::BinOp(BinOp::Add, l, r) => {
                assert!(matches!(*l, Expr::Lit(Literal::Int(1))));
                assert!(matches!(*r, Expr::Lit(Literal::Int(2))));
            }
            _ => panic!("Expected BinOp"),
        }
    }

    #[test]
    fn test_parse_precedence() {
        // 1 + 2 * 3 = 1 + (2 * 3)
        let expr = parse_expr("1 + 2 * 3").unwrap();
        match expr {
            Expr::BinOp(BinOp::Add, l, r) => {
                assert!(matches!(*l, Expr::Lit(Literal::Int(1))));
                assert!(matches!(*r, Expr::BinOp(BinOp::Mul, _, _)));
            }
            _ => panic!("Expected Add at top"),
        }
    }

    #[test]
    fn test_parse_parens() {
        // (1 + 2) * 3
        let expr = parse_expr("(1 + 2) * 3").unwrap();
        match expr {
            Expr::BinOp(BinOp::Mul, l, r) => {
                assert!(matches!(*l, Expr::BinOp(BinOp::Add, _, _)));
                assert!(matches!(*r, Expr::Lit(Literal::Int(3))));
            }
            _ => panic!("Expected Mul at top"),
        }
    }

    #[test]
    fn test_parse_lambda_simple() {
        let expr = parse_expr("λ→ ₀").unwrap();
        match expr {
            Expr::Lam(body) => assert!(matches!(*body, Expr::Idx(0))),
            _ => panic!("Expected Lam"),
        }
    }

    #[test]
    fn test_parse_lambda_ascii() {
        let expr = parse_expr("\\-> _0 + 1").unwrap();
        match expr {
            Expr::Lam(body) => assert!(matches!(*body, Expr::BinOp(BinOp::Add, _, _))),
            _ => panic!("Expected Lam"),
        }
    }

    #[test]
    fn test_parse_application() {
        let expr = parse_expr("f x").unwrap();
        match expr {
            Expr::App(func, arg) => {
                assert!(matches!(*func, Expr::Name(_)));
                assert!(matches!(*arg, Expr::Name(_)));
            }
            _ => panic!("Expected App"),
        }
    }

    #[test]
    fn test_parse_chained_application() {
        // f x y = (f x) y
        let expr = parse_expr("f x y").unwrap();
        match expr {
            Expr::App(func, _) => {
                assert!(matches!(*func, Expr::App(_, _)));
            }
            _ => panic!("Expected nested App"),
        }
    }

    #[test]
    fn test_parse_let() {
        let expr = parse_expr("let x = 5 in x + 1").unwrap();
        match expr {
            Expr::Let { pattern, value, body } => {
                assert!(matches!(pattern, Pattern::Var(_)));
                assert!(matches!(*value, Expr::Lit(Literal::Int(5))));
                assert!(matches!(*body, Expr::BinOp(BinOp::Add, _, _)));
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_parse_if() {
        let expr = parse_expr("if ⊤ then 1 else 2").unwrap();
        match expr {
            Expr::If { cond, then_, else_ } => {
                assert!(matches!(*cond, Expr::Lit(Literal::True)));
                assert!(matches!(*then_, Expr::Lit(Literal::Int(1))));
                assert!(matches!(*else_, Expr::Lit(Literal::Int(2))));
            }
            _ => panic!("Expected If"),
        }
    }

    #[test]
    fn test_parse_match() {
        let expr = parse_expr("match x { 0 → 1; _ → 2 }").unwrap();
        match expr {
            Expr::Match { arms, .. } => {
                assert_eq!(arms.len(), 2);
            }
            _ => panic!("Expected Match"),
        }
    }

    #[test]
    fn test_parse_array() {
        let expr = parse_expr("[1, 2, 3]").unwrap();
        match expr {
            Expr::Array(elems) => assert_eq!(elems.len(), 3),
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_parse_tuple() {
        let expr = parse_expr("(1, 2, 3)").unwrap();
        match expr {
            Expr::Tuple(elems) => assert_eq!(elems.len(), 3),
            _ => panic!("Expected Tuple"),
        }
    }

    #[test]
    fn test_parse_angle_tuple() {
        let expr = parse_expr("⟨1, 2⟩").unwrap();
        match expr {
            Expr::Tuple(elems) => assert_eq!(elems.len(), 2),
            _ => panic!("Expected Tuple"),
        }
    }

    #[test]
    fn test_parse_field_access() {
        let expr = parse_expr("x.foo").unwrap();
        match expr {
            Expr::Field(_, goth_ast::expr::FieldAccess::Named(n)) => {
                assert_eq!(n.as_ref(), "foo");
            }
            _ => panic!("Expected Field"),
        }
    }

    #[test]
    fn test_parse_index_access() {
        let expr = parse_expr("x.0").unwrap();
        match expr {
            Expr::Field(_, goth_ast::expr::FieldAccess::Index(0)) => {}
            _ => panic!("Expected Field with index"),
        }
    }

    #[test]
    fn test_parse_map() {
        let expr = parse_expr("[1,2,3] ↦ λ→ ₀ + 1").unwrap();
        match expr {
            Expr::BinOp(BinOp::Map, _, _) => {}
            _ => panic!("Expected Map"),
        }
    }

    #[test]
    fn test_parse_map_ascii() {
        let expr = parse_expr("[1,2,3] -: \\-> _0 + 1").unwrap();
        match expr {
            Expr::BinOp(BinOp::Map, _, _) => {}
            _ => panic!("Expected Map"),
        }
    }

    #[test]
    fn test_parse_sum() {
        let expr = parse_expr("Σ [1,2,3]").unwrap();
        match expr {
            Expr::UnaryOp(goth_ast::op::UnaryOp::Sum, _) => {}
            _ => panic!("Expected Sum"),
        }
    }

    #[test]
    fn test_parse_compose() {
        let expr = parse_expr("f ∘ g").unwrap();
        match expr {
            Expr::BinOp(BinOp::Compose, _, _) => {}
            _ => panic!("Expected Compose"),
        }
    }

    #[test]
    fn test_parse_unary_neg() {
        let expr = parse_expr("-5").unwrap();
        match expr {
            Expr::UnaryOp(goth_ast::op::UnaryOp::Neg, _) => {}
            _ => panic!("Expected Neg"),
        }
    }

    // ============ Pattern Parsing ============

    #[test]
    fn test_parse_pattern_wildcard() {
        let pat = parse_pattern("_").unwrap();
        assert!(matches!(pat, Pattern::Wildcard));
    }

    #[test]
    fn test_parse_pattern_var() {
        let pat = parse_pattern("x").unwrap();
        match pat {
            Pattern::Var(Some(n)) => assert_eq!(n.as_ref(), "x"),
            _ => panic!("Expected Var"),
        }
    }

    #[test]
    fn test_parse_pattern_literal() {
        let pat = parse_pattern("42").unwrap();
        assert!(matches!(pat, Pattern::Lit(Literal::Int(42))));
    }

    #[test]
    fn test_parse_pattern_tuple() {
        let pat = parse_pattern("(x, y)").unwrap();
        match pat {
            Pattern::Tuple(pats) => assert_eq!(pats.len(), 2),
            _ => panic!("Expected Tuple"),
        }
    }

    #[test]
    fn test_parse_pattern_array() {
        let pat = parse_pattern("[x, y, z]").unwrap();
        match pat {
            Pattern::Array(pats) => assert_eq!(pats.len(), 3),
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_parse_pattern_variant() {
        let pat = parse_pattern("Some x").unwrap();
        match pat {
            Pattern::Variant { constructor, payload } => {
                assert_eq!(constructor.as_ref(), "Some");
                assert!(payload.is_some());
            }
            _ => panic!("Expected Variant"),
        }
    }

    #[test]
    fn test_parse_pattern_or() {
        let pat = parse_pattern("0 | 1").unwrap();
        match pat {
            Pattern::Or(_, _) => {}
            _ => panic!("Expected Or"),
        }
    }

    // ============ Type Parsing ============

    #[test]
    fn test_parse_type_prim() {
        assert!(matches!(parse_type("F64").unwrap(), Type::Prim(PrimType::F64)));
        assert!(matches!(parse_type("I32").unwrap(), Type::Prim(PrimType::I32)));
        assert!(matches!(parse_type("Bool").unwrap(), Type::Prim(PrimType::Bool)));
    }

    #[test]
    fn test_parse_type_var() {
        let ty = parse_type("α").unwrap();
        match ty {
            Type::Var(v) => assert_eq!(v.as_ref(), "α"),
            _ => panic!("Expected Var"),
        }
    }

    #[test]
    fn test_parse_type_function() {
        let ty = parse_type("F64 → F64").unwrap();
        match ty {
            Type::Fn(arg, ret) => {
                assert!(matches!(*arg, Type::Prim(PrimType::F64)));
                assert!(matches!(*ret, Type::Prim(PrimType::F64)));
            }
            _ => panic!("Expected Fn"),
        }
    }

    #[test]
    fn test_parse_type_function_chain() {
        // F64 → F64 → F64 = F64 → (F64 → F64)
        let ty = parse_type("F64 → F64 → F64").unwrap();
        match ty {
            Type::Fn(arg, ret) => {
                assert!(matches!(*arg, Type::Prim(PrimType::F64)));
                assert!(matches!(*ret, Type::Fn(_, _)));
            }
            _ => panic!("Expected Fn"),
        }
    }

    #[test]
    fn test_parse_type_tensor() {
        let ty = parse_type("[n]F64").unwrap();
        match ty {
            Type::Tensor(shape, elem) => {
                assert_eq!(shape.rank(), 1);
                assert!(matches!(*elem, Type::Prim(PrimType::F64)));
            }
            _ => panic!("Expected Tensor"),
        }
    }

    #[test]
    fn test_parse_type_matrix() {
        let ty = parse_type("[m n]F64").unwrap();
        match ty {
            Type::Tensor(shape, _) => {
                assert_eq!(shape.rank(), 2);
            }
            _ => panic!("Expected Tensor"),
        }
    }

    #[test]
    fn test_parse_type_tuple() {
        let ty = parse_type("(F64, I32)").unwrap();
        match ty {
            Type::Tuple(fields) => assert_eq!(fields.len(), 2),
            _ => panic!("Expected Tuple"),
        }
    }

    #[test]
    fn test_parse_type_forall() {
        let ty = parse_type("∀ α. α → α").unwrap();
        match ty {
            Type::Forall(params, body) => {
                assert_eq!(params.len(), 1);
                assert!(matches!(*body, Type::Fn(_, _)));
            }
            _ => panic!("Expected Forall"),
        }
    }

    // ============ Declaration Parsing ============

    #[test]
    fn test_parse_let_decl() {
        let module = parse_module("let x = 42", "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_parse_fn_decl() {
        let src = "╭─ id : α → α\n╰─ ₀";
        let module = parse_module(src, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    // ============ Complex Expressions ============

    #[test]
    fn test_parse_complex_map_filter() {
        let expr = parse_expr("[1,2,3,4,5] ↦ λ→ ₀ * 2").unwrap();
        match expr {
            Expr::BinOp(BinOp::Map, arr, _) => {
                assert!(matches!(*arr, Expr::Array(_)));
            }
            _ => panic!("Expected Map"),
        }
    }

    #[test]
    fn test_parse_nested_let() {
        let expr = parse_expr("let x = 1 in let y = 2 in x + y").unwrap();
        match expr {
            Expr::Let { body, .. } => {
                assert!(matches!(*body, Expr::Let { .. }));
            }
            _ => panic!("Expected nested Let"),
        }
    }

    #[test]
    fn test_parse_comparison_chain() {
        let expr = parse_expr("x < y && y < z").unwrap();
        match expr {
            Expr::BinOp(BinOp::And, _, _) => {}
            _ => panic!("Expected And at top"),
        }
    }

    // ============ Error Cases ============

    #[test]
    fn test_parse_error_unexpected() {
        assert!(parse_expr("+ 1").is_err());
    }

    #[test]
    fn test_parse_error_unclosed_paren() {
        assert!(parse_expr("(1 + 2").is_err());
    }

    #[test]
    fn test_parse_error_unclosed_bracket() {
        assert!(parse_expr("[1, 2, 3").is_err());
    }

    // ============ Bug Fix Tests ============
    // Tests for syntax corrections and refinements from session

    #[test]
    fn test_greek_letter_in_pattern() {
        // Greek letters should work in let patterns
        let expr = parse_expr("let μ = 5 in μ").unwrap();
        match expr {
            Expr::Let { pattern, .. } => {
                match pattern {
                    Pattern::Var(Some(name)) => assert_eq!(&*name, "μ"),
                    _ => panic!("Expected var pattern with μ"),
                }
            }
            _ => panic!("Expected let expression"),
        }
    }

    #[test]
    fn test_greek_letter_in_expression() {
        // Greek letters should work as variable references
        let expr = parse_expr("μ + 1").unwrap();
        match expr {
            Expr::BinOp(BinOp::Add, left, _) => {
                match *left {
                    Expr::Name(name) => assert_eq!(&*name, "μ"),
                    _ => panic!("Expected μ as name"),
                }
            }
            _ => panic!("Expected binary operation"),
        }
    }

    #[test]
    fn test_greek_letter_field_access() {
        // Greek letters should work in field access
        let expr = parse_expr("stats.μ").unwrap();
        match expr {
            Expr::Field(_, field) => {
                match field {
                    goth_ast::expr::FieldAccess::Named(name) => assert_eq!(&*name, "μ"),
                    _ => panic!("Expected named field μ"),
                }
            }
            _ => panic!("Expected field access"),
        }
    }

    #[test]
    fn test_back_arrow_let_binding() {
        // ← should work as alternative to =
        let expr1 = parse_expr("let x ← 5 in x").unwrap();
        let expr2 = parse_expr("let x = 5 in x").unwrap();
        // Both should parse to the same structure
        match (expr1, expr2) {
            (Expr::Let { .. }, Expr::Let { .. }) => {
                // Success - both are let expressions
            }
            _ => panic!("Both should be let expressions"),
        }
    }

    #[test]
    fn test_back_arrow_let_rec() {
        // ← should work in let rec
        let expr = parse_expr("let rec f ← λ→ ₀ in f").unwrap();
        match expr {
            Expr::LetRec { bindings, .. } => {
                assert_eq!(bindings.len(), 1);
            }
            _ => panic!("Expected let rec"),
        }
    }

    #[test]
    fn test_sequential_let_bindings() {
        // Semicolons should allow sequential bindings
        let expr = parse_expr("let x ← 1 ; y ← 2 in x + y").unwrap();
        // This should parse as nested lets
        match expr {
            Expr::Let { body, .. } => {
                match *body {
                    Expr::Let { .. } => {
                        // Success - nested let structure
                    }
                    _ => panic!("Expected nested let"),
                }
            }
            _ => panic!("Expected outer let"),
        }
    }

    #[test]
    fn test_sequential_let_three_bindings() {
        // Multiple sequential bindings
        let expr = parse_expr("let x ← 1 ; y ← 2 ; z ← 3 in x + y + z").unwrap();
        // Should parse successfully
        match expr {
            Expr::Let { .. } => {
                // Success
            }
            _ => panic!("Expected let expression"),
        }
    }

    #[test]
    fn test_sequential_let_trailing_semicolon() {
        // Trailing semicolon should be optional
        let expr = parse_expr("let x ← 1 ; y ← 2 ; in x + y").unwrap();
        match expr {
            Expr::Let { .. } => {
                // Success
            }
            _ => panic!("Expected let expression"),
        }
    }

    #[test]
    fn test_expr_application_then_operator() {
        // Critical bug fix: function application followed by operator
        // This was failing with "expected In, got Slash"
        let expr = parse_expr("sum ₀ / len ₀").unwrap();
        match expr {
            Expr::BinOp(BinOp::Div, left, right) => {
                // Left should be application: sum ₀
                match *left {
                    Expr::App(_, _) => {
                        // Success
                    }
                    _ => panic!("Expected application on left"),
                }
                // Right should be application: len ₀
                match *right {
                    Expr::App(_, _) => {
                        // Success
                    }
                    _ => panic!("Expected application on right"),
                }
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_complex_expr_with_application_and_ops() {
        // sum arr / n should parse as (sum arr) / n
        let expr = parse_expr("sum arr / n").unwrap();
        match expr {
            Expr::BinOp(BinOp::Div, _, _) => {
                // Success - parsed as division
            }
            _ => panic!("Expected division operation"),
        }
    }

    #[test]
    fn test_expr_multiple_applications_and_ops() {
        // f x + g y should parse as (f x) + (g y)
        let expr = parse_expr("f x + g y").unwrap();
        match expr {
            Expr::BinOp(BinOp::Add, left, right) => {
                match (*left, *right) {
                    (Expr::App(_, _), Expr::App(_, _)) => {
                        // Success - both sides are applications
                    }
                    _ => panic!("Expected applications on both sides"),
                }
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_normalize_function_body() {
        // The actual normalize function that was failing
        let body = "let arr ← ₀ ; n ← len arr ; μ ← sum arr / n in μ";
        let expr = parse_expr(body).unwrap();
        match expr {
            Expr::Let { .. } => {
                // Success - complex function body parsed
            }
            _ => panic!("Expected let expression"),
        }
    }

    #[test]
    fn test_greek_sequential_bindings() {
        // Combining Greek letters and sequential bindings
        let expr = parse_expr("let μ ← 5 ; σ ← 2 in μ + σ").unwrap();
        match expr {
            Expr::Let { pattern, body, .. } => {
                match pattern {
                    Pattern::Var(Some(name)) => assert_eq!(&*name, "μ"),
                    _ => panic!("Expected μ pattern"),
                }
                match *body {
                    Expr::Let { .. } => {
                        // Success - nested let
                    }
                    _ => panic!("Expected nested let"),
                }
            }
            _ => panic!("Expected let"),
        }
    }

    #[test]
    fn test_function_decl_with_greek() {
        // Function declarations should accept Greek letters
        let source = "╭─ test : F → F\n╰─ let μ ← ₀ in μ";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    // ============ Multi-Argument Function Tests ============

    #[test]
    fn test_multi_arg_function_type_parsing() {
        // Two-argument function type
        let ty = parse_type("F → F → F").unwrap();
        match ty {
            goth_ast::types::Type::Fn(_arg1, ret1) => {
                match *ret1 {
                    goth_ast::types::Type::Fn(_arg2, _ret2) => {
                        // Success - nested function type
                    }
                    _ => panic!("Expected nested function type"),
                }
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_three_arg_function_type_parsing() {
        // Three-argument function type
        let ty = parse_type("F → F → F → F").unwrap();
        match ty {
            goth_ast::types::Type::Fn(_arg1, ret1) => {
                match *ret1 {
                    goth_ast::types::Type::Fn(_arg2, ret2) => {
                        match *ret2 {
                            goth_ast::types::Type::Fn(_arg3, _ret3) => {
                                // Success - triple nested function type
                            }
                            _ => panic!("Expected third function type"),
                        }
                    }
                    _ => panic!("Expected second function type"),
                }
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_multi_arg_function_declaration() {
        // Two-argument function declaration with body using both args
        let source = "╭─ pythag : F → F → F\n╰─ √(₀ × ₀ + ₁ × ₁)";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
        
        match &module.decls[0] {
            goth_ast::decl::Decl::Fn(fn_decl) => {
                assert_eq!(&*fn_decl.name, "pythag");
                // Verify type signature parsed correctly
                match &fn_decl.signature {
                    goth_ast::types::Type::Fn(_arg1, ret) => {
                        match ret.as_ref() {
                            goth_ast::types::Type::Fn(_arg2, _ret) => {
                                // Success - two arrows
                            }
                            _ => panic!("Expected second arrow"),
                        }
                    }
                    _ => panic!("Expected function type"),
                }
            }
            _ => panic!("Expected function declaration"),
        }
    }

    #[test]
    fn test_three_arg_function_declaration() {
        // Three-argument function
        let source = "╭─ add3 : F → F → F → F\n╰─ ₀ + ₁ + ₂";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_tuple_arg_function_type() {
        // Single tuple argument (not curried)
        let ty = parse_type("⟨F, F⟩ → F").unwrap();
        match ty {
            goth_ast::types::Type::Fn(arg, _ret) => {
                match *arg {
                    goth_ast::types::Type::Tuple(fields) => {
                        assert_eq!(fields.len(), 2);
                    }
                    _ => panic!("Expected tuple type"),
                }
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_tuple_arg_function_declaration() {
        // Function with tuple parameter
        let source = "╭─ pythag : ⟨F, F⟩ → F\n╰─ √(₀.0 × ₀.0 + ₀.1 × ₀.1)";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_mixed_tuple_and_curried() {
        // Mix of tuple and curried: ⟨F, F⟩ → F → F
        let ty = parse_type("⟨F, F⟩ → F → F").unwrap();
        match ty {
            goth_ast::types::Type::Fn(arg, ret) => {
                match *arg {
                    goth_ast::types::Type::Tuple(_) => {
                        // First arg is tuple
                    }
                    _ => panic!("Expected tuple arg"),
                }
                match *ret {
                    goth_ast::types::Type::Fn(_arg2, _ret2) => {
                        // Second arrow present
                    }
                    _ => panic!("Expected second arrow"),
                }
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_multi_arg_with_operators() {
        // Test that new operators work in multi-arg functions
        let source = "╭─ test : F → F → F\n╰─ √₀ + ⌊₁⌋";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_uncertain_in_function_type() {
        // Function returning uncertain value
        let source = "╭─ measure : F → F → (F ± F)\n╰─ ₀ ± ₁";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_enum_decl_simple() {
        // Simple enum with no type parameters
        let source = "enum Color where Red | Green | Blue";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
        match &module.decls[0] {
            goth_ast::decl::Decl::Enum(e) => {
                assert_eq!(e.name.as_ref(), "Color");
                assert!(e.params.is_empty());
                assert_eq!(e.variants.len(), 3);
                assert_eq!(e.variants[0].name.as_ref(), "Red");
                assert!(e.variants[0].payload.is_none());
                assert_eq!(e.variants[1].name.as_ref(), "Green");
                assert!(e.variants[1].payload.is_none());
                assert_eq!(e.variants[2].name.as_ref(), "Blue");
                assert!(e.variants[2].payload.is_none());
            }
            _ => panic!("Expected Enum decl"),
        }
    }

    #[test]
    fn test_enum_decl_with_payload() {
        // Enum with payload types
        let source = "enum Option τ where Some τ | None";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
        match &module.decls[0] {
            goth_ast::decl::Decl::Enum(e) => {
                assert_eq!(e.name.as_ref(), "Option");
                assert_eq!(e.params.len(), 1);
                assert_eq!(e.params[0].name.as_ref(), "τ");
                assert_eq!(e.variants.len(), 2);
                assert_eq!(e.variants[0].name.as_ref(), "Some");
                assert!(e.variants[0].payload.is_some());
                assert_eq!(e.variants[1].name.as_ref(), "None");
                assert!(e.variants[1].payload.is_none());
            }
            _ => panic!("Expected Enum decl"),
        }
    }

    #[test]
    fn test_enum_decl_either() {
        // Either type with two type parameters
        let source = "enum Either a b where Left a | Right b";
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
        match &module.decls[0] {
            goth_ast::decl::Decl::Enum(e) => {
                assert_eq!(e.name.as_ref(), "Either");
                assert_eq!(e.params.len(), 2);
                assert_eq!(e.variants.len(), 2);
                assert_eq!(e.variants[0].name.as_ref(), "Left");
                assert_eq!(e.variants[1].name.as_ref(), "Right");
            }
            _ => panic!("Expected Enum decl"),
        }
    }
}

