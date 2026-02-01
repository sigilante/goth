//! # Goth Eval - Interpreter for the Goth programming language

pub mod value;
pub mod error;
pub mod prim;
pub mod eval;

pub mod prelude {
    pub use crate::value::{Value, Tensor, TensorData, Closure, Env, PrimFn};
    pub use crate::error::{EvalError, EvalResult};
    pub use crate::eval::{Evaluator, eval, eval_trace};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use goth_ast::prelude::*;
    use std::rc::Rc;

    #[test] fn test_int_literal() { assert_eq!(eval(&Expr::int(42)).unwrap(), Value::Int(42)); }
    #[test] fn test_float_literal() { assert_eq!(eval(&Expr::float(3.14)).unwrap(), Value::float(3.14)); }
    #[test] fn test_bool_literals() { assert_eq!(eval(&Expr::bool(true)).unwrap(), Value::Bool(true)); assert_eq!(eval(&Expr::bool(false)).unwrap(), Value::Bool(false)); }
    #[test] fn test_addition() { assert_eq!(eval(&Expr::add(Expr::int(2), Expr::int(3))).unwrap(), Value::Int(5)); }
    #[test] fn test_subtraction() { assert_eq!(eval(&Expr::sub(Expr::int(10), Expr::int(4))).unwrap(), Value::Int(6)); }
    #[test] fn test_multiplication() { assert_eq!(eval(&Expr::mul(Expr::int(6), Expr::int(7))).unwrap(), Value::Int(42)); }
    #[test] fn test_division() { assert_eq!(eval(&Expr::div(Expr::int(20), Expr::int(4))).unwrap(), Value::Int(5)); }
    #[test] fn test_division_by_zero() { assert!(matches!(eval(&Expr::div(Expr::int(1), Expr::int(0))), Err(EvalError::DivisionByZero))); }
    #[test] fn test_nested_arithmetic() { assert_eq!(eval(&Expr::mul(Expr::add(Expr::int(2), Expr::int(3)), Expr::sub(Expr::int(4), Expr::int(1)))).unwrap(), Value::Int(15)); }
    #[test] fn test_float_arithmetic() { assert_eq!(eval(&Expr::add(Expr::float(1.5), Expr::float(2.5))).unwrap(), Value::float(4.0)); }
    #[test] fn test_mixed_arithmetic() { assert_eq!(eval(&Expr::add(Expr::int(1), Expr::float(2.5))).unwrap(), Value::float(3.5)); }
    #[test] fn test_equality() { assert_eq!(eval(&Expr::binop(BinOp::Eq, Expr::int(5), Expr::int(5))).unwrap(), Value::Bool(true)); assert_eq!(eval(&Expr::binop(BinOp::Eq, Expr::int(5), Expr::int(6))).unwrap(), Value::Bool(false)); }
    #[test] fn test_less_than() { assert_eq!(eval(&Expr::binop(BinOp::Lt, Expr::int(3), Expr::int(5))).unwrap(), Value::Bool(true)); }
    #[test] fn test_logical_and() { assert_eq!(eval(&Expr::binop(BinOp::And, Expr::bool(true), Expr::bool(true))).unwrap(), Value::Bool(true)); assert_eq!(eval(&Expr::binop(BinOp::And, Expr::bool(true), Expr::bool(false))).unwrap(), Value::Bool(false)); }
    #[test] fn test_logical_or() { assert_eq!(eval(&Expr::binop(BinOp::Or, Expr::bool(false), Expr::bool(true))).unwrap(), Value::Bool(true)); }
    #[test] fn test_logical_short_circuit() { assert_eq!(eval(&Expr::binop(BinOp::Or, Expr::bool(true), Expr::binop(BinOp::Eq, Expr::div(Expr::int(1), Expr::int(0)), Expr::int(0)))).unwrap(), Value::Bool(true)); }

    #[test] fn test_identity_function() { assert_eq!(eval(&Expr::app(Expr::lam(Expr::idx(0)), Expr::int(42))).unwrap(), Value::Int(42)); }
    #[test] fn test_constant_function() { assert_eq!(eval(&Expr::app(Expr::app(Expr::lam(Expr::lam(Expr::idx(1))), Expr::int(1)), Expr::int(2))).unwrap(), Value::Int(1)); }
    #[test] fn test_add_function() { let add_fn = Expr::lam(Expr::lam(Expr::add(Expr::idx(1), Expr::idx(0)))); assert_eq!(eval(&Expr::app(Expr::app(add_fn, Expr::int(3)), Expr::int(4))).unwrap(), Value::Int(7)); }
    #[test] fn test_closure_capture() { let expr = Expr::let_(Pattern::var("x"), Expr::int(10), Expr::app(Expr::lam(Expr::add(Expr::idx(0), Expr::idx(1))), Expr::int(5))); assert_eq!(eval(&expr).unwrap(), Value::Int(15)); }

    #[test] fn test_simple_let() { let expr = Expr::let_(Pattern::var("x"), Expr::int(5), Expr::add(Expr::idx(0), Expr::int(3))); assert_eq!(eval(&expr).unwrap(), Value::Int(8)); }
    #[test] fn test_nested_let() { let expr = Expr::let_(Pattern::var("x"), Expr::int(5), Expr::let_(Pattern::var("y"), Expr::int(3), Expr::add(Expr::idx(1), Expr::idx(0)))); assert_eq!(eval(&expr).unwrap(), Value::Int(8)); }

    #[test] fn test_if_true() { assert_eq!(eval(&Expr::if_(Expr::bool(true), Expr::int(1), Expr::int(2))).unwrap(), Value::Int(1)); }
    #[test] fn test_if_false() { assert_eq!(eval(&Expr::if_(Expr::bool(false), Expr::int(1), Expr::int(2))).unwrap(), Value::Int(2)); }
    #[test] fn test_if_with_comparison() { assert_eq!(eval(&Expr::if_(Expr::binop(BinOp::Lt, Expr::int(3), Expr::int(5)), Expr::int(1), Expr::int(2))).unwrap(), Value::Int(1)); }

    #[test] fn test_match_literal() { let expr = Expr::match_(Expr::int(1), vec![MatchArm::new(Pattern::lit(1i64), Expr::int(10)), MatchArm::new(Pattern::wildcard(), Expr::int(20))]); assert_eq!(eval(&expr).unwrap(), Value::Int(10)); }
    #[test] fn test_match_wildcard() { let expr = Expr::match_(Expr::int(99), vec![MatchArm::new(Pattern::lit(1i64), Expr::int(10)), MatchArm::new(Pattern::wildcard(), Expr::int(20))]); assert_eq!(eval(&expr).unwrap(), Value::Int(20)); }
    #[test] fn test_match_tuple() { let expr = Expr::match_(Expr::tuple(vec![Expr::int(1), Expr::int(2)]), vec![MatchArm::new(Pattern::tuple(vec![Pattern::var("a"), Pattern::var("b")]), Expr::add(Expr::idx(1), Expr::idx(0)))]); assert_eq!(eval(&expr).unwrap(), Value::Int(3)); }
    #[test] fn test_match_variant() { let expr = Expr::match_(Expr::variant("Some", Some(Expr::int(5))), vec![MatchArm::new(Pattern::variant("None", None), Expr::int(0)), MatchArm::new(Pattern::variant("Some", Some(Pattern::var("x"))), Expr::idx(0))]); assert_eq!(eval(&expr).unwrap(), Value::Int(5)); }

    #[test] fn test_array_literal() { let result = eval(&Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3)])).unwrap(); match result { Value::Tensor(t) => { assert_eq!(t.shape, vec![3]); assert_eq!(t.get_flat(0), Some(Value::Int(1))); } _ => panic!("Expected tensor") } }
    #[test] fn test_array_sum() { assert_eq!(eval(&Expr::sum(Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4)]))).unwrap(), Value::Int(10)); }
    #[test] fn test_array_map() { let expr = Expr::map(Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3)]), Expr::lam(Expr::mul(Expr::idx(0), Expr::int(2)))); match eval(&expr).unwrap() { Value::Tensor(t) => { assert_eq!(t.get_flat(0), Some(Value::Int(2))); assert_eq!(t.get_flat(1), Some(Value::Int(4))); assert_eq!(t.get_flat(2), Some(Value::Int(6))); } _ => panic!("Expected tensor") } }
    #[test] fn test_array_filter() { let expr = Expr::filter(Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4), Expr::int(5)]), Expr::lam(Expr::binop(BinOp::Gt, Expr::idx(0), Expr::int(2)))); match eval(&expr).unwrap() { Value::Tensor(t) => { assert_eq!(t.len(), 3); assert_eq!(t.get_flat(0), Some(Value::Int(3))); } _ => panic!("Expected tensor") } }
    #[test] fn test_tensor_broadcasting() { let expr = Expr::add(Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3)]), Expr::int(10)); match eval(&expr).unwrap() { Value::Tensor(t) => { assert_eq!(t.get_flat(0), Some(Value::Int(11))); } _ => panic!("Expected tensor") } }

    #[test] fn test_tuple_construction() { match eval(&Expr::tuple(vec![Expr::int(1), Expr::bool(true), Expr::float(3.14)])).unwrap() { Value::Tuple(vs) => { assert_eq!(vs.len(), 3); assert_eq!(vs[0], Value::Int(1)); assert_eq!(vs[1], Value::Bool(true)); } _ => panic!("Expected tuple") } }
    #[test] fn test_tuple_field_access() { assert_eq!(eval(&Expr::field_idx(Expr::tuple(vec![Expr::int(1), Expr::int(2), Expr::int(3)]), 1)).unwrap(), Value::Int(2)); }
    #[test] fn test_unit() { assert_eq!(eval(&Expr::tuple(vec![])).unwrap(), Value::Unit); }

    #[test] fn test_factorial() {
        let mut e = Evaluator::new();
        let factorial_body = Expr::match_(Expr::idx(0), vec![MatchArm::new(Pattern::lit(0i64), Expr::int(1)), MatchArm::new(Pattern::var("n"), Expr::mul(Expr::idx(0), Expr::app(Expr::name("factorial"), Expr::sub(Expr::idx(0), Expr::int(1)))))]);
        // Create closure with access to globals so it can call itself recursively
        let env = Env::with_globals(e.globals());
        e.define("factorial", Value::closure(1, factorial_body, env));
        assert_eq!(e.eval(&Expr::app(Expr::name("factorial"), Expr::int(5))).unwrap(), Value::Int(120));
    }

    #[test] fn test_fibonacci() {
        let mut e = Evaluator::new();
        let fib_body = Expr::match_(Expr::idx(0), vec![MatchArm::new(Pattern::lit(0i64), Expr::int(0)), MatchArm::new(Pattern::lit(1i64), Expr::int(1)), MatchArm::new(Pattern::var("n"), Expr::add(Expr::app(Expr::name("fib"), Expr::sub(Expr::idx(0), Expr::int(1))), Expr::app(Expr::name("fib"), Expr::sub(Expr::idx(0), Expr::int(2)))))]);
        // Create closure with access to globals so it can call itself recursively
        let env = Env::with_globals(e.globals());
        e.define("fib", Value::closure(1, fib_body, env));
        assert_eq!(e.eval(&Expr::app(Expr::name("fib"), Expr::int(10))).unwrap(), Value::Int(55));
    }

    #[test] fn test_sqrt_primitive() { let mut e = Evaluator::new(); assert_eq!(e.eval(&Expr::app(Expr::name("sqrt"), Expr::float(16.0))).unwrap(), Value::float(4.0)); }
    #[test] fn test_abs_primitive() { let mut e = Evaluator::new(); assert_eq!(e.eval(&Expr::app(Expr::name("abs"), Expr::int(-5))).unwrap(), Value::Int(5)); }
    #[test] fn test_partial_application() { let mut e = Evaluator::new(); let add5 = Expr::app(Expr::name("add"), Expr::int(5)); assert!(e.eval(&add5).unwrap().is_callable()); assert_eq!(e.eval(&Expr::app(add5, Expr::int(3))).unwrap(), Value::Int(8)); }
    #[test] fn test_function_composition() { let add1 = Expr::lam(Expr::add(Expr::idx(0), Expr::int(1))); let mul2 = Expr::lam(Expr::mul(Expr::idx(0), Expr::int(2))); let composed = Expr::binop(BinOp::Compose, add1, mul2); assert_eq!(eval(&Expr::app(composed, Expr::int(3))).unwrap(), Value::Int(7)); }
    #[test] fn test_map_filter_sum() { let expr = Expr::sum(Expr::map(Expr::filter(Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3), Expr::int(4), Expr::int(5)]), Expr::lam(Expr::binop(BinOp::Eq, Expr::binop(BinOp::Mod, Expr::idx(0), Expr::int(2)), Expr::int(0)))), Expr::lam(Expr::mul(Expr::idx(0), Expr::idx(0))))); assert_eq!(eval(&expr).unwrap(), Value::Int(20)); }
    #[test] fn test_nested_lambdas() { let expr = Expr::app(Expr::app(Expr::app(Expr::lam(Expr::lam(Expr::lam(Expr::add(Expr::add(Expr::idx(2), Expr::idx(1)), Expr::idx(0))))), Expr::int(1)), Expr::int(2)), Expr::int(3)); assert_eq!(eval(&expr).unwrap(), Value::Int(6)); }
    #[test] fn test_type_error() { assert!(eval(&Expr::add(Expr::int(1), Expr::bool(true))).is_err()); }
    #[test] fn test_unbound_variable() { assert!(matches!(eval(&Expr::idx(999)), Err(EvalError::UnboundIndex(999)))); }
    #[test] fn test_undefined_name() { assert!(matches!(eval(&Expr::name("nonexistent")), Err(EvalError::UndefinedName(_)))); }
    #[test] fn test_de_bruijn_simple() { assert_eq!(eval(&Expr::app(Expr::lam(Expr::idx(0)), Expr::int(5))).unwrap(), Value::Int(5)); }
    #[test] fn test_de_bruijn_nested() { assert_eq!(eval(&Expr::app(Expr::app(Expr::lam(Expr::lam(Expr::add(Expr::idx(0), Expr::idx(1)))), Expr::int(3)), Expr::int(4))).unwrap(), Value::Int(7)); }
    #[test] fn test_de_bruijn_capture_in_closure() { let expr = Expr::let_(Pattern::var("x"), Expr::int(5), Expr::let_(Pattern::var("f"), Expr::lam(Expr::add(Expr::idx(1), Expr::idx(0))), Expr::app(Expr::idx(0), Expr::int(3)))); assert_eq!(eval(&expr).unwrap(), Value::Int(8)); }
    #[test] fn test_dot_product() { let mut e = Evaluator::new(); let a = Expr::array(vec![Expr::float(1.0), Expr::float(2.0), Expr::float(3.0)]); let b = Expr::array(vec![Expr::float(4.0), Expr::float(5.0), Expr::float(6.0)]); let expr = Expr::app(Expr::app(Expr::name("dot"), a), b); assert_eq!(e.eval(&expr).unwrap(), Value::float(32.0)); }

    // ============ Multi-Argument Function Tests ============

    #[test]
    fn test_two_arg_closure() {
        // Test that a closure with arity=2 works correctly
        let body = Expr::add(Expr::idx(1), Expr::idx(0));  // ₀ + ₁
        let closure = Value::closure(2, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("add_two", closure);
        
        // Apply both arguments
        let expr = Expr::app(Expr::app(Expr::name("add_two"), Expr::int(3)), Expr::int(4));
        assert_eq!(e.eval(&expr).unwrap(), Value::Int(7));
    }

    #[test]
    fn test_two_arg_partial_application() {
        // Test partial application of a two-argument closure
        let body = Expr::mul(Expr::idx(1), Expr::idx(0));  // ₀ × ₁
        let closure = Value::closure(2, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("mul_two", closure);
        
        // Apply first argument only
        let expr = Expr::app(Expr::name("mul_two"), Expr::int(5));
        let partial = e.eval(&expr).unwrap();
        assert!(partial.is_callable());
        
        // Apply second argument
        let expr2 = Expr::app(Expr::name("mul_two_5"), Expr::int(6));
        e.define("mul_two_5", partial);
        assert_eq!(e.eval(&expr2).unwrap(), Value::Int(30));
    }

    #[test]
    fn test_three_arg_closure() {
        // Test closure with arity=3
        let body = Expr::add(Expr::add(Expr::idx(2), Expr::idx(1)), Expr::idx(0));  // ₀ + ₁ + ₂
        let closure = Value::closure(3, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("add_three", closure);
        
        // Apply all three arguments
        let expr = Expr::app(
            Expr::app(
                Expr::app(Expr::name("add_three"), Expr::int(1)),
                Expr::int(2)
            ),
            Expr::int(3)
        );
        assert_eq!(e.eval(&expr).unwrap(), Value::Int(6));
    }

    #[test]
    fn test_pythag_two_arg() {
        // Test Pythagorean theorem as two-argument function
        // √(₀² + ₁²)
        let body = Expr::UnaryOp(
            UnaryOp::Sqrt,
            Box::new(Expr::add(
                Expr::mul(Expr::idx(1), Expr::idx(1)),
                Expr::mul(Expr::idx(0), Expr::idx(0))
            ))
        );
        let closure = Value::closure(2, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("pythag", closure);
        
        let expr = Expr::app(
            Expr::app(Expr::name("pythag"), Expr::float(3.0)),
            Expr::float(4.0)
        );
        assert_eq!(e.eval(&expr).unwrap(), Value::float(5.0));
    }

    #[test]
    fn test_multi_arg_with_new_operators() {
        // Test multi-arg function using new Unicode operators
        // ⌊₁⌋ + ⌈₀⌉ (first arg floored + second arg ceiled)
        let body = Expr::add(
            Expr::UnaryOp(UnaryOp::Floor, Box::new(Expr::idx(1))),
            Expr::UnaryOp(UnaryOp::Ceil, Box::new(Expr::idx(0)))
        );
        let closure = Value::closure(2, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("floor_ceil_add", closure);
        
        let expr = Expr::app(
            Expr::app(Expr::name("floor_ceil_add"), Expr::float(3.7)),
            Expr::float(2.2)
        );
        // floor and ceil return Int, not Float
        // ⌊3.7⌋ + ⌈2.2⌉ = Int(3) + Int(3) = Int(6)
        assert_eq!(e.eval(&expr).unwrap(), Value::Int(6));
    }

    #[test]
    fn test_uncertain_value_creation() {
        // Test creating uncertain values with ±
        let expr = Expr::binop(BinOp::PlusMinus, Expr::float(10.5), Expr::float(0.3));
        let result = eval(&expr).unwrap();
        
        match result {
            Value::Uncertain { value, uncertainty } => {
                assert_eq!(*value, Value::float(10.5));
                assert_eq!(*uncertainty, Value::float(0.3));
            }
            _ => panic!("Expected uncertain value"),
        }
    }

    #[test]
    fn test_multi_arg_uncertain_function() {
        // Function that creates uncertain value from two args
        // With arity 2: ₁ refers to first arg, ₀ refers to second (last) arg
        // So ₁ ± ₀ means first_arg ± second_arg
        let body = Expr::binop(BinOp::PlusMinus, Expr::idx(1), Expr::idx(0));
        let closure = Value::closure(2, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("make_uncertain", closure);
        
        let expr = Expr::app(
            Expr::app(Expr::name("make_uncertain"), Expr::float(5.0)),
            Expr::float(0.5)
        );
        
        let result = e.eval(&expr).unwrap();
        
        // Match and extract values
        match result {
            Value::Uncertain { ref value, ref uncertainty } => {
                // Check using pattern matching
                match (value.as_ref(), uncertainty.as_ref()) {
                    (Value::Float(v), Value::Float(u)) => {
                        assert_eq!(v.0, 5.0);
                        assert_eq!(u.0, 0.5);
                    }
                    _ => panic!("Expected Float values, got {:?} ± {:?}", value, uncertainty),
                }
            }
            other => panic!("Expected uncertain value, got {:?}", other),
        }
    }

    // ============ Uncertainty Propagation Tests ============

    /// Helper: extract (value, uncertainty) from an Uncertain result
    fn unc_parts(v: &Value) -> (f64, f64) {
        match v {
            Value::Uncertain { value, uncertainty } => {
                (value.coerce_float().unwrap(), uncertainty.coerce_float().unwrap())
            }
            _ => panic!("Expected Uncertain, got {:?}", v),
        }
    }

    /// Helper: make an uncertain expression (a ± da)
    fn unc_expr(a: f64, da: f64) -> Expr {
        Expr::binop(BinOp::PlusMinus, Expr::float(a), Expr::float(da))
    }

    /// Assert approximate equality within epsilon
    fn assert_approx(actual: f64, expected: f64, eps: f64, msg: &str) {
        assert!((actual - expected).abs() < eps,
            "{}: expected {}, got {} (diff {})", msg, expected, actual, (actual - expected).abs());
    }

    #[test]
    fn test_uncertain_add() {
        // (5.0 ± 0.1) + (3.0 ± 0.2) → 8.0 ± √(0.01 + 0.04) = 8.0 ± 0.22360...
        let expr = Expr::binop(BinOp::Add, unc_expr(5.0, 0.1), unc_expr(3.0, 0.2));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 8.0, 1e-10, "add value");
        assert_approx(u, (0.01_f64 + 0.04).sqrt(), 1e-10, "add uncertainty");
    }

    #[test]
    fn test_uncertain_sub() {
        // (10.0 ± 0.3) - (4.0 ± 0.1) → 6.0 ± √(0.09 + 0.01) = 6.0 ± 0.31622...
        let expr = Expr::binop(BinOp::Sub, unc_expr(10.0, 0.3), unc_expr(4.0, 0.1));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 6.0, 1e-10, "sub value");
        assert_approx(u, (0.09_f64 + 0.01).sqrt(), 1e-10, "sub uncertainty");
    }

    #[test]
    fn test_uncertain_mul() {
        // (4.0 ± 0.1) × (3.0 ± 0.2) → 12.0 ± 12 * √((0.1/4)² + (0.2/3)²)
        let expr = Expr::binop(BinOp::Mul, unc_expr(4.0, 0.1), unc_expr(3.0, 0.2));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        let expected_u = 12.0 * ((0.1/4.0_f64).powi(2) + (0.2/3.0_f64).powi(2)).sqrt();
        assert_approx(v, 12.0, 1e-10, "mul value");
        assert_approx(u, expected_u, 1e-10, "mul uncertainty");
    }

    #[test]
    fn test_uncertain_div() {
        // (10.0 ± 0.5) / (2.0 ± 0.1) → 5.0 ± 5 * √((0.5/10)² + (0.1/2)²)
        let expr = Expr::binop(BinOp::Div, unc_expr(10.0, 0.5), unc_expr(2.0, 0.1));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        let expected_u = 5.0 * ((0.5/10.0_f64).powi(2) + (0.1/2.0_f64).powi(2)).sqrt();
        assert_approx(v, 5.0, 1e-10, "div value");
        assert_approx(u, expected_u, 1e-10, "div uncertainty");
    }

    #[test]
    fn test_uncertain_scalar_add() {
        // (5.0 ± 0.1) + 3.0 → 8.0 ± 0.1
        let expr = Expr::binop(BinOp::Add, unc_expr(5.0, 0.1), Expr::float(3.0));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 8.0, 1e-10, "scalar add value");
        assert_approx(u, 0.1, 1e-10, "scalar add uncertainty");
    }

    #[test]
    fn test_uncertain_scalar_mul() {
        // (5.0 ± 0.1) × 3.0 → 15.0 ± 0.3
        let expr = Expr::binop(BinOp::Mul, unc_expr(5.0, 0.1), Expr::float(3.0));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 15.0, 1e-10, "scalar mul value");
        assert_approx(u, 0.3, 1e-10, "scalar mul uncertainty");
    }

    #[test]
    fn test_uncertain_neg() {
        // -(5.0 ± 0.1) → -5.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Neg, Box::new(unc_expr(5.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, -5.0, 1e-10, "neg value");
        assert_approx(u, 0.1, 1e-10, "neg uncertainty");
    }

    #[test]
    fn test_uncertain_sqrt() {
        // √(4.0 ± 0.1) → 2.0 ± 0.1/(2*2) = 2.0 ± 0.025
        let expr = Expr::UnaryOp(UnaryOp::Sqrt, Box::new(unc_expr(4.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 2.0, 1e-10, "sqrt value");
        assert_approx(u, 0.025, 1e-10, "sqrt uncertainty");
    }

    #[test]
    fn test_uncertain_sin() {
        // sin(0.0 ± 0.1) → 0.0 ± |cos(0)| * 0.1 = 0.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Sin, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 0.0, 1e-10, "sin value");
        assert_approx(u, 0.1, 1e-10, "sin uncertainty");
    }

    #[test]
    fn test_uncertain_cos() {
        // cos(0.0 ± 0.1) → 1.0 ± |sin(0)| * 0.1 = 1.0 ± 0.0
        let expr = Expr::UnaryOp(UnaryOp::Cos, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 1.0, 1e-10, "cos value");
        assert_approx(u, 0.0, 1e-10, "cos uncertainty");
    }

    #[test]
    fn test_uncertain_exp() {
        // exp(1.0 ± 0.1) → e ± e * 0.1
        let e_val = std::f64::consts::E;
        let expr = Expr::UnaryOp(UnaryOp::Exp, Box::new(unc_expr(1.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, e_val, 1e-10, "exp value");
        assert_approx(u, e_val * 0.1, 1e-10, "exp uncertainty");
    }

    #[test]
    fn test_uncertain_ln() {
        // ln(1.0 ± 0.1) → 0.0 ± 0.1/|1.0| = 0.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Ln, Box::new(unc_expr(1.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 0.0, 1e-10, "ln value");
        assert_approx(u, 0.1, 1e-10, "ln uncertainty");
    }

    #[test]
    fn test_uncertain_pow() {
        // (2.0 ± 0.1) ^ 3 → 8.0 ± |8 * 3 * 0.1 / 2| = 8.0 ± 1.2
        let expr = Expr::binop(BinOp::Pow, unc_expr(2.0, 0.1), Expr::int(3));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 8.0, 1e-10, "pow value");
        assert_approx(u, 1.2, 1e-10, "pow uncertainty");
    }

    #[test]
    fn test_uncertain_chained() {
        // (2.0 ± 0.1) × (3.0 ± 0.2) + (1.0 ± 0.05)
        // First: 6.0 ± 6*√((0.1/2)² + (0.2/3)²) = 6.0 ± 0.5
        // Then: 7.0 ± √(0.5² + 0.05²)
        let mul_expr = Expr::binop(BinOp::Mul, unc_expr(2.0, 0.1), unc_expr(3.0, 0.2));
        let expr = Expr::binop(BinOp::Add, mul_expr, unc_expr(1.0, 0.05));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 7.0, 1e-10, "chained value");
        // intermediate uncertainty from mul
        let mul_u = 6.0 * ((0.1/2.0_f64).powi(2) + (0.2/3.0_f64).powi(2)).sqrt();
        let expected_u = (mul_u.powi(2) + 0.05_f64.powi(2)).sqrt();
        assert_approx(u, expected_u, 1e-10, "chained uncertainty");
    }

    #[test]
    fn test_uncertain_comparison() {
        // (5.0 ± 0.1) = (5.0 ± 0.2) → true (compares values, ignores uncertainty)
        let expr = Expr::binop(BinOp::Eq, unc_expr(5.0, 0.1), unc_expr(5.0, 0.2));
        assert_eq!(eval(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_uncertain_abs() {
        // abs(-3.0 ± 0.2) → 3.0 ± 0.2
        let expr = Expr::UnaryOp(UnaryOp::Abs, Box::new(unc_expr(-3.0, 0.2)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 3.0, 1e-10, "abs value");
        assert_approx(u, 0.2, 1e-10, "abs uncertainty");
    }

    #[test]
    fn test_uncertain_atan() {
        // atan(1.0 ± 0.1) → π/4 ± 0.1/(1+1²) = π/4 ± 0.05
        let expr = Expr::UnaryOp(UnaryOp::Atan, Box::new(unc_expr(1.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, std::f64::consts::FRAC_PI_4, 1e-10, "atan value");
        assert_approx(u, 0.05, 1e-10, "atan uncertainty");
    }

    #[test]
    fn test_uncertain_log10() {
        // log10(100.0 ± 1.0) → 2.0 ± 1.0 / (100 * ln10)
        let expr = Expr::UnaryOp(UnaryOp::Log10, Box::new(unc_expr(100.0, 1.0)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 2.0, 1e-10, "log10 value");
        assert_approx(u, 1.0 / (100.0 * std::f64::consts::LN_10), 1e-10, "log10 uncertainty");
    }

    #[test]
    fn test_uncertain_tanh() {
        // tanh(0.0 ± 0.1) → 0.0 ± sech²(0) * 0.1 = 0.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Tanh, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 0.0, 1e-10, "tanh value");
        assert_approx(u, 0.1, 1e-10, "tanh uncertainty");  // sech²(0) = 1
    }

    #[test]
    fn test_uncertain_gamma() {
        // Γ(3.0 ± 0.1) → 2.0 ± |Γ(3) * ψ(3) * 0.1|
        // ψ(3) = 1 - γ + 1/1 + 1/2 = 1 + 1/2 - γ_euler ≈ 0.9227...
        // but we just check the value is 2.0 and uncertainty is reasonable
        let expr = Expr::UnaryOp(UnaryOp::Gamma, Box::new(unc_expr(3.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 2.0, 1e-6, "gamma value");
        // ψ(3) ≈ 0.9227..., so uncertainty ≈ 2.0 * 0.9227 * 0.1 ≈ 0.1845
        assert!(u > 0.15 && u < 0.25, "gamma uncertainty {} not in expected range", u);
    }

    #[test]
    fn test_uncertain_log2() {
        // log2(8.0 ± 0.1) → 3.0 ± 0.1 / (8 * ln2)
        let expr = Expr::UnaryOp(UnaryOp::Log2, Box::new(unc_expr(8.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 3.0, 1e-10, "log2 value");
        assert_approx(u, 0.1 / (8.0 * std::f64::consts::LN_2), 1e-10, "log2 uncertainty");
    }

    #[test]
    fn test_uncertain_sinh() {
        // sinh(0.0 ± 0.1) → 0.0 ± cosh(0) * 0.1 = 0.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Sinh, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 0.0, 1e-10, "sinh value");
        assert_approx(u, 0.1, 1e-10, "sinh uncertainty");
    }

    #[test]
    fn test_uncertain_cosh() {
        // cosh(0.0 ± 0.1) → 1.0 ± |sinh(0)| * 0.1 = 1.0 ± 0.0
        let expr = Expr::UnaryOp(UnaryOp::Cosh, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 1.0, 1e-10, "cosh value");
        assert_approx(u, 0.0, 1e-10, "cosh uncertainty");
    }

    #[test]
    fn test_uncertain_asin() {
        // asin(0.0 ± 0.1) → 0.0 ± 0.1/√(1-0) = 0.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Asin, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 0.0, 1e-10, "asin value");
        assert_approx(u, 0.1, 1e-10, "asin uncertainty");
    }

    #[test]
    fn test_uncertain_acos() {
        // acos(0.0 ± 0.1) → π/2 ± 0.1/√(1-0) = π/2 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Acos, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, std::f64::consts::FRAC_PI_2, 1e-10, "acos value");
        assert_approx(u, 0.1, 1e-10, "acos uncertainty");
    }

    #[test]
    fn test_uncertain_tan() {
        // tan(0.0 ± 0.1) → 0.0 ± sec²(0)*0.1 = 0.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Tan, Box::new(unc_expr(0.0, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 0.0, 1e-10, "tan value");
        assert_approx(u, 0.1, 1e-10, "tan uncertainty");
    }

    #[test]
    fn test_uncertain_floor_ceil_round() {
        // floor(3.7 ± 0.1) → 3.0 ± 0.1 (uncertainty passes through)
        let expr = Expr::UnaryOp(UnaryOp::Floor, Box::new(unc_expr(3.7, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 3.0, 1e-10, "floor value");
        assert_approx(u, 0.1, 1e-10, "floor uncertainty");

        // ceil(3.2 ± 0.1) → 4.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Ceil, Box::new(unc_expr(3.2, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 4.0, 1e-10, "ceil value");
        assert_approx(u, 0.1, 1e-10, "ceil uncertainty");

        // round(3.5 ± 0.1) → 4.0 ± 0.1
        let expr = Expr::UnaryOp(UnaryOp::Round, Box::new(unc_expr(3.5, 0.1)));
        let (v, u) = unc_parts(&eval(&expr).unwrap());
        assert_approx(v, 4.0, 1e-10, "round value");
        assert_approx(u, 0.1, 1e-10, "round uncertainty");
    }

    // ============ Math Library Tests (non-uncertain) ============

    #[test]
    fn test_math_exp_ln_roundtrip() {
        // ln(exp(2.0)) ≈ 2.0
        let expr = Expr::UnaryOp(UnaryOp::Ln, Box::new(
            Expr::UnaryOp(UnaryOp::Exp, Box::new(Expr::float(2.0)))));
        let v = eval(&expr).unwrap().coerce_float().unwrap();
        assert_approx(v, 2.0, 1e-10, "exp/ln roundtrip");
    }

    #[test]
    fn test_math_log2_powers() {
        // log2(1024) = 10
        let expr = Expr::UnaryOp(UnaryOp::Log2, Box::new(Expr::float(1024.0)));
        let v = eval(&expr).unwrap().coerce_float().unwrap();
        assert_approx(v, 10.0, 1e-10, "log2(1024)");
    }

    #[test]
    fn test_math_log10_powers() {
        // log10(1000) = 3
        let expr = Expr::UnaryOp(UnaryOp::Log10, Box::new(Expr::float(1000.0)));
        let v = eval(&expr).unwrap().coerce_float().unwrap();
        assert_approx(v, 3.0, 1e-10, "log10(1000)");
    }

    #[test]
    fn test_math_sqrt_perfect() {
        // √144 = 12
        let expr = Expr::UnaryOp(UnaryOp::Sqrt, Box::new(Expr::float(144.0)));
        let v = eval(&expr).unwrap().coerce_float().unwrap();
        assert_approx(v, 12.0, 1e-10, "sqrt(144)");
    }

    #[test]
    fn test_math_trig_identity_sin2_cos2() {
        // sin²(x) + cos²(x) = 1 for x = 1.23
        let x = 1.23;
        let sin_expr = Expr::UnaryOp(UnaryOp::Sin, Box::new(Expr::float(x)));
        let cos_expr = Expr::UnaryOp(UnaryOp::Cos, Box::new(Expr::float(x)));
        let sv = eval(&sin_expr).unwrap().coerce_float().unwrap();
        let cv = eval(&cos_expr).unwrap().coerce_float().unwrap();
        assert_approx(sv * sv + cv * cv, 1.0, 1e-10, "sin²+cos²=1");
    }

    #[test]
    fn test_math_trig_tan_ratio() {
        // tan(x) = sin(x)/cos(x) for x = 0.7
        let x = 0.7;
        let tan_v = eval(&Expr::UnaryOp(UnaryOp::Tan, Box::new(Expr::float(x)))).unwrap().coerce_float().unwrap();
        let sin_v = eval(&Expr::UnaryOp(UnaryOp::Sin, Box::new(Expr::float(x)))).unwrap().coerce_float().unwrap();
        let cos_v = eval(&Expr::UnaryOp(UnaryOp::Cos, Box::new(Expr::float(x)))).unwrap().coerce_float().unwrap();
        assert_approx(tan_v, sin_v / cos_v, 1e-10, "tan=sin/cos");
    }

    #[test]
    fn test_math_inverse_trig() {
        // asin(sin(0.5)) ≈ 0.5, acos(cos(0.5)) ≈ 0.5, atan(tan(0.5)) ≈ 0.5
        let x = 0.5;
        let v = eval(&Expr::UnaryOp(UnaryOp::Asin, Box::new(
            Expr::UnaryOp(UnaryOp::Sin, Box::new(Expr::float(x)))))).unwrap().coerce_float().unwrap();
        assert_approx(v, x, 1e-10, "asin(sin(x))");

        let v = eval(&Expr::UnaryOp(UnaryOp::Acos, Box::new(
            Expr::UnaryOp(UnaryOp::Cos, Box::new(Expr::float(x)))))).unwrap().coerce_float().unwrap();
        assert_approx(v, x, 1e-10, "acos(cos(x))");

        let v = eval(&Expr::UnaryOp(UnaryOp::Atan, Box::new(
            Expr::UnaryOp(UnaryOp::Tan, Box::new(Expr::float(x)))))).unwrap().coerce_float().unwrap();
        assert_approx(v, x, 1e-10, "atan(tan(x))");
    }

    #[test]
    fn test_math_hyperbolic_identity() {
        // cosh²(x) - sinh²(x) = 1
        let x = 1.5;
        let ch = eval(&Expr::UnaryOp(UnaryOp::Cosh, Box::new(Expr::float(x)))).unwrap().coerce_float().unwrap();
        let sh = eval(&Expr::UnaryOp(UnaryOp::Sinh, Box::new(Expr::float(x)))).unwrap().coerce_float().unwrap();
        assert_approx(ch * ch - sh * sh, 1.0, 1e-10, "cosh²-sinh²=1");
    }

    #[test]
    fn test_math_tanh_range() {
        // tanh(x) ∈ (-1, 1) for any finite x; tanh(0) = 0
        let v = eval(&Expr::UnaryOp(UnaryOp::Tanh, Box::new(Expr::float(0.0)))).unwrap().coerce_float().unwrap();
        assert_approx(v, 0.0, 1e-10, "tanh(0)=0");

        let v = eval(&Expr::UnaryOp(UnaryOp::Tanh, Box::new(Expr::float(100.0)))).unwrap().coerce_float().unwrap();
        assert_approx(v, 1.0, 1e-10, "tanh(100)≈1");
    }

    #[test]
    fn test_math_gamma_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24
        for (n, expected) in [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (4.0, 6.0), (5.0, 24.0)] {
            let v = eval(&Expr::UnaryOp(UnaryOp::Gamma, Box::new(Expr::float(n)))).unwrap().coerce_float().unwrap();
            assert_approx(v, expected, 1e-6, &format!("Γ({})", n));
        }
    }

    #[test]
    fn test_math_gamma_half() {
        // Γ(0.5) = √π
        let v = eval(&Expr::UnaryOp(UnaryOp::Gamma, Box::new(Expr::float(0.5)))).unwrap().coerce_float().unwrap();
        assert_approx(v, std::f64::consts::PI.sqrt(), 1e-6, "Γ(0.5)=√π");
    }

    #[test]
    fn test_math_floor_ceil_round() {
        let v = eval(&Expr::UnaryOp(UnaryOp::Floor, Box::new(Expr::float(3.7)))).unwrap();
        assert_eq!(v, Value::Int(3));
        let v = eval(&Expr::UnaryOp(UnaryOp::Ceil, Box::new(Expr::float(3.2)))).unwrap();
        assert_eq!(v, Value::Int(4));
        let v = eval(&Expr::UnaryOp(UnaryOp::Round, Box::new(Expr::float(3.5)))).unwrap();
        assert_eq!(v, Value::Int(4));
        let v = eval(&Expr::UnaryOp(UnaryOp::Round, Box::new(Expr::float(3.4)))).unwrap();
        assert_eq!(v, Value::Int(3));
    }

    #[test]
    fn test_math_abs_sign() {
        let v = eval(&Expr::UnaryOp(UnaryOp::Abs, Box::new(Expr::float(-5.0)))).unwrap().coerce_float().unwrap();
        assert_approx(v, 5.0, 1e-10, "abs(-5)");

        let v = eval(&Expr::UnaryOp(UnaryOp::Sign, Box::new(Expr::float(-3.0)))).unwrap().coerce_float().unwrap();
        assert_approx(v, -1.0, 1e-10, "sign(-3)");

        let v = eval(&Expr::UnaryOp(UnaryOp::Sign, Box::new(Expr::float(0.0)))).unwrap().coerce_float().unwrap();
        assert_approx(v, 0.0, 1e-10, "sign(0)");
    }

    #[test]
    fn test_math_pow_integer() {
        // 2^10 = 1024
        let expr = Expr::binop(BinOp::Pow, Expr::int(2), Expr::int(10));
        let v = eval(&expr).unwrap();
        assert_eq!(v, Value::Int(1024));
    }

    #[test]
    fn test_math_pow_fractional() {
        // 8^(1/3) ≈ 2.0
        let expr = Expr::binop(BinOp::Pow, Expr::float(8.0), Expr::binop(BinOp::Div, Expr::float(1.0), Expr::float(3.0)));
        let v = eval(&expr).unwrap().coerce_float().unwrap();
        assert_approx(v, 2.0, 1e-10, "8^(1/3)");
    }

    #[test]
    fn test_math_modulo() {
        let expr = Expr::binop(BinOp::Mod, Expr::int(17), Expr::int(5));
        assert_eq!(eval(&expr).unwrap(), Value::Int(2));

        let expr = Expr::binop(BinOp::Mod, Expr::float(7.5), Expr::float(2.5));
        let v = eval(&expr).unwrap().coerce_float().unwrap();
        assert_approx(v, 0.0, 1e-10, "7.5 mod 2.5");
    }

    #[test]
    fn test_curried_with_sqrt() {
        // Test curried function with √ operator
        // λ→ λ→ √(₀ × ₁)
        let body = Expr::UnaryOp(
            UnaryOp::Sqrt,
            Box::new(Expr::mul(Expr::idx(1), Expr::idx(0)))
        );
        let closure = Value::closure(2, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("sqrt_product", closure);
        
        let expr = Expr::app(
            Expr::app(Expr::name("sqrt_product"), Expr::float(4.0)),
            Expr::float(9.0)
        );
        assert_eq!(e.eval(&expr).unwrap(), Value::float(6.0)); // √(4 × 9) = √36 = 6
    }

    #[test]
    fn test_four_arg_function() {
        // Test with four arguments
        // ₀ + ₁ + ₂ + ₃
        let body = Expr::add(
            Expr::add(Expr::idx(3), Expr::idx(2)),
            Expr::add(Expr::idx(1), Expr::idx(0))
        );
        let closure = Value::closure(4, body, Env::new());
        
        let mut e = Evaluator::new();
        e.define("add_four", closure);
        
        let expr = Expr::app(
            Expr::app(
                Expr::app(
                    Expr::app(Expr::name("add_four"), Expr::int(1)),
                    Expr::int(2)
                ),
                Expr::int(3)
            ),
            Expr::int(4)
        );
        assert_eq!(e.eval(&expr).unwrap(), Value::Int(10));
    }

    // ============ String and I/O Tests ============

    #[test]
    fn test_string_value() {
        // Create a string value from the Value constructor
        let s = Value::string("hello");
        match s {
            Value::Tensor(t) => {
                assert_eq!(t.to_string_value(), Some("hello".to_string()));
            }
            _ => panic!("Expected string (tensor of chars)"),
        }
    }

    #[test]
    fn test_string_literal() {
        let expr = Expr::Lit(Literal::String("hello world".into()));
        let result = eval(&expr).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.to_string_value(), Some("hello world".to_string()));
            }
            _ => panic!("Expected string tensor"),
        }
    }

    #[test]
    fn test_to_string_int() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("toString"), Expr::int(42));
        let result = e.eval(&expr).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.to_string_value(), Some("42".to_string()));
            }
            _ => panic!("Expected string tensor"),
        }
    }

    #[test]
    fn test_to_string_float() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("toString"), Expr::float(3.14));
        let result = e.eval(&expr).unwrap();
        match result {
            Value::Tensor(t) => {
                let s = t.to_string_value().unwrap();
                assert!(s.starts_with("3.14"));
            }
            _ => panic!("Expected string tensor"),
        }
    }

    #[test]
    fn test_file_io_roundtrip() {
        use std::fs;

        let mut e = Evaluator::new();

        // Create a temp file path
        let temp_path = "/tmp/goth_test_io.txt";
        let content = "Hello, Goth!";

        // Write to file using writeFile primitive
        let write_expr = Expr::app(
            Expr::app(Expr::name("writeFile"), Expr::Lit(Literal::String(temp_path.into()))),
            Expr::Lit(Literal::String(content.into()))
        );
        let result = e.eval(&write_expr);
        assert!(result.is_ok(), "writeFile failed: {:?}", result.err());

        // Read from file using readFile primitive
        let read_expr = Expr::app(Expr::name("readFile"), Expr::Lit(Literal::String(temp_path.into())));
        let result = e.eval(&read_expr).unwrap();

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.to_string_value(), Some(content.to_string()));
            }
            _ => panic!("Expected string tensor"),
        }

        // Clean up
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_read_file_not_found() {
        let mut e = Evaluator::new();
        let expr = Expr::app(
            Expr::name("readFile"),
            Expr::Lit(Literal::String("/nonexistent/path/file.txt".into()))
        );
        let result = e.eval(&expr);
        assert!(result.is_err(), "Expected error but got: {:?}", result);
        match result {
            Err(EvalError::IoError(msg)) => {
                assert!(msg.contains("Failed to read"));
            }
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_str_concat() {
        let mut e = Evaluator::new();
        let expr = Expr::app(
            Expr::app(Expr::name("strConcat"), Expr::Lit(Literal::String("Hello, ".into()))),
            Expr::Lit(Literal::String("World!".into()))
        );
        let result = e.eval(&expr).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.to_string_value(), Some("Hello, World!".to_string()));
            }
            _ => panic!("Expected string tensor"),
        }
    }

    #[test]
    fn test_chars() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("chars"), Expr::Lit(Literal::String("abc".into())));
        let result = e.eval(&expr).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.len(), 3);
                assert_eq!(t.get_flat(0), Some(Value::Char('a')));
                assert_eq!(t.get_flat(1), Some(Value::Char('b')));
                assert_eq!(t.get_flat(2), Some(Value::Char('c')));
            }
            _ => panic!("Expected tensor of chars"),
        }
    }

    #[test]
    fn test_wildcard_binding_shifts_indices() {
        // Wildcards must push to env to maintain De Bruijn index alignment.
        // let x = 1 in let _ = 2 in x
        // After wildcard, x is at index 1 (wildcard at 0).
        let mut e = Evaluator::new();
        let expr = Expr::let_(
            Pattern::var("x"),
            Expr::int(1),
            Expr::let_(
                Pattern::Wildcard,
                Expr::int(2),
                Expr::idx(1), // x is at index 1 after wildcard
            ),
        );
        assert_eq!(e.eval(&expr).unwrap(), Value::Int(1));
    }

    #[test]
    fn test_multiple_wildcards() {
        // let x = 10 in let _ = 20 in let _ = 30 in x
        // After two wildcards, x is at index 2.
        let mut e = Evaluator::new();
        let expr = Expr::let_(
            Pattern::var("x"),
            Expr::int(10),
            Expr::let_(
                Pattern::Wildcard,
                Expr::int(20),
                Expr::let_(
                    Pattern::Wildcard,
                    Expr::int(30),
                    Expr::idx(2), // x is at index 2 after two wildcards
                ),
            ),
        );
        assert_eq!(e.eval(&expr).unwrap(), Value::Int(10));
    }

    // ============ Concat (⊕) Tests ============

    #[test]
    fn test_array_concat() {
        let expr = Expr::binop(BinOp::Concat, Expr::array(vec![Expr::int(1), Expr::int(2)]), Expr::array(vec![Expr::int(3), Expr::int(4)]));
        match eval(&expr).unwrap() { Value::Tensor(t) => { assert_eq!(t.len(), 4); assert_eq!(t.get_flat(0), Some(Value::Int(1))); assert_eq!(t.get_flat(3), Some(Value::Int(4))); } _ => panic!("Expected tensor") }
    }

    #[test]
    fn test_concat_strings() {
        let expr = Expr::binop(BinOp::Concat, Expr::Lit(Literal::String("ab".into())), Expr::Lit(Literal::String("cd".into())));
        match eval(&expr).unwrap() { Value::Tensor(t) => { assert_eq!(t.to_string_value(), Some("abcd".to_string())); } _ => panic!("Expected string tensor") }
    }

    #[test]
    fn test_concat_empty() {
        let expr = Expr::binop(BinOp::Concat, Expr::array(vec![]), Expr::array(vec![Expr::int(1), Expr::int(2)]));
        match eval(&expr).unwrap() { Value::Tensor(t) => { assert_eq!(t.len(), 2); assert_eq!(t.get_flat(0), Some(Value::Int(1))); } _ => panic!("Expected tensor") }
    }

    // ============ ZipWith (⊗) Tests ============

    #[test]
    fn test_zip_with() {
        let expr = Expr::binop(BinOp::ZipWith, Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3)]), Expr::array(vec![Expr::int(4), Expr::int(5), Expr::int(6)]));
        match eval(&expr).unwrap() {
            Value::Tensor(t) => {
                assert_eq!(t.len(), 3);
                assert_eq!(t.get_flat(0), Some(Value::Tuple(vec![Value::Int(1), Value::Int(4)])));
                assert_eq!(t.get_flat(1), Some(Value::Tuple(vec![Value::Int(2), Value::Int(5)])));
                assert_eq!(t.get_flat(2), Some(Value::Tuple(vec![Value::Int(3), Value::Int(6)])));
            }
            _ => panic!("Expected tensor of tuples")
        }
    }

    #[test]
    fn test_zip_dot_product() {
        // Dot product via zip: dot(a, b) = Σ(a ⊗ b mapped to product)
        // Use the dot primitive directly to verify zip works in the pipeline
        let mut e = Evaluator::new();
        let a = Expr::array(vec![Expr::float(1.0), Expr::float(2.0), Expr::float(3.0)]);
        let b = Expr::array(vec![Expr::float(4.0), Expr::float(5.0), Expr::float(6.0)]);
        let expr = Expr::app(Expr::app(Expr::name("dot"), a), b);
        assert_eq!(e.eval(&expr).unwrap(), Value::float(32.0));
    }

    // ============ Bind (⤇) Tests ============

    #[test]
    fn test_bind_flatmap() {
        // [1,2,3] ⤇ (λ→ [₀, ₀×2]) → [1,2,2,4,3,6]
        let expr = Expr::binop(
            BinOp::Bind,
            Expr::array(vec![Expr::int(1), Expr::int(2), Expr::int(3)]),
            Expr::lam(Expr::array(vec![Expr::idx(0), Expr::mul(Expr::idx(0), Expr::int(2))]))
        );
        match eval(&expr).unwrap() {
            Value::Tensor(t) => {
                assert_eq!(t.len(), 6);
                assert_eq!(t.get_flat(0), Some(Value::Int(1)));
                assert_eq!(t.get_flat(1), Some(Value::Int(2)));
                assert_eq!(t.get_flat(2), Some(Value::Int(2)));
                assert_eq!(t.get_flat(3), Some(Value::Int(4)));
                assert_eq!(t.get_flat(4), Some(Value::Int(3)));
                assert_eq!(t.get_flat(5), Some(Value::Int(6)));
            }
            _ => panic!("Expected tensor")
        }
    }

    // ============ StructEq (≡ / ==) Tests ============

    #[test]
    fn test_struct_eq_integers() {
        let expr = Expr::binop(BinOp::StructEq, Expr::int(5), Expr::int(5));
        assert_eq!(eval(&expr).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_struct_eq_arrays() {
        let expr = Expr::binop(
            BinOp::StructEq,
            Expr::array(vec![Expr::int(1), Expr::int(2)]),
            Expr::array(vec![Expr::int(1), Expr::int(2)]),
        );
        assert_eq!(eval(&expr).unwrap(), Value::Bool(true));
    }

    // ============ Mod keyword Tests ============

    #[test]
    fn test_mod_keyword() {
        // 10 mod 3 → 1 (via parser)
        let expr = Expr::binop(BinOp::Mod, Expr::int(10), Expr::int(3));
        assert_eq!(eval(&expr).unwrap(), Value::Int(1));
    }

    // ============ Write (▷) Tests ============

    #[test]
    fn test_write_to_file() {
        use std::fs;
        let temp_path = "/tmp/goth_test_write.txt";
        let mut e = Evaluator::new();
        let expr = Expr::binop(
            BinOp::Write,
            Expr::Lit(Literal::String("goth_write_content".into())),
            Expr::Lit(Literal::String(temp_path.into()))
        );
        let result = e.eval(&expr).unwrap();
        assert_eq!(result, Value::Unit);
        let contents = fs::read_to_string(temp_path).unwrap();
        assert_eq!(contents, "goth_write_content");
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_write_to_stdout() {
        let mut e = Evaluator::new();
        let expr = Expr::binop(BinOp::Write, Expr::Lit(Literal::String("hello".into())), Expr::name("stdout"));
        let result = e.eval(&expr).unwrap();
        assert_eq!(result, Value::Unit);
    }

    #[test]
    fn test_write_to_stderr() {
        let mut e = Evaluator::new();
        let expr = Expr::binop(BinOp::Write, Expr::Lit(Literal::String("error".into())), Expr::name("stderr"));
        let result = e.eval(&expr).unwrap();
        assert_eq!(result, Value::Unit);
    }

    // ============ Read (◁) Tests ============

    #[test]
    fn test_read_from_file() {
        use std::fs;
        let temp_path = "/tmp/goth_test_read.txt";
        fs::write(temp_path, "goth_read_content").unwrap();
        let mut e = Evaluator::new();
        // Read is a BinOp where left is the path; right is unused in eval_read
        let expr = Expr::binop(BinOp::Read, Expr::Lit(Literal::String(temp_path.into())), Expr::tuple(vec![]));
        let result = e.eval(&expr).unwrap();
        match result {
            Value::Tensor(t) => { assert_eq!(t.to_string_value(), Some("goth_read_content".to_string())); }
            _ => panic!("Expected string tensor")
        }
        let _ = fs::remove_file(temp_path);
    }

    // ============ Rc-wrapping invariant tests ============
    // These must pass BEFORE and AFTER the Rc-wrapping refactor.

    #[test]
    fn invariant_tensor_string_roundtrip() {
        let v = Value::string("hello");
        match &v {
            Value::Tensor(t) => assert_eq!(t.to_string_value(), Some("hello".to_string())),
            _ => panic!("Expected Tensor"),
        }
    }

    #[test]
    fn invariant_tensor_clone_independence() {
        let v1 = Value::Tensor(Rc::new(Tensor::from_ints(vec![1, 2, 3])));
        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn invariant_tensor_deep_eq() {
        let a = Value::Tensor(Rc::new(Tensor::from_ints(vec![1, 2, 3])));
        let b = Value::Tensor(Rc::new(Tensor::from_ints(vec![1, 2, 3])));
        assert!(a.deep_eq(&b));
    }

    #[test]
    fn invariant_tensor_as_tensor() {
        let v = Value::Tensor(Rc::new(Tensor::from_floats(vec![1.0, 2.0])));
        let t = v.as_tensor().unwrap();
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn invariant_tensor_display() {
        let v = Value::Tensor(Rc::new(Tensor::from_ints(vec![1, 2, 3])));
        let s = format!("{}", v);
        assert!(s.contains("1") && s.contains("2") && s.contains("3"));
    }

    #[test]
    fn invariant_concat_strings() {
        let mut e = Evaluator::new();
        let expr = Expr::app(
            Expr::app(Expr::name("concat"), Expr::Lit(Literal::String("ab".into()))),
            Expr::Lit(Literal::String("cd".into())),
        );
        let result = e.eval(&expr).unwrap();
        match &result {
            Value::Tensor(t) => assert_eq!(t.to_string_value(), Some("abcd".to_string())),
            _ => panic!("Expected string tensor"),
        }
    }

    #[test]
    fn invariant_concat_int_arrays() {
        let mut e = Evaluator::new();
        let expr = Expr::app(
            Expr::app(
                Expr::name("concat"),
                Expr::array(vec![Expr::int(1), Expr::int(2)]),
            ),
            Expr::array(vec![Expr::int(3), Expr::int(4)]),
        );
        let result = e.eval(&expr).unwrap();
        match &result {
            Value::Tensor(t) => {
                assert_eq!(t.len(), 4);
                assert_eq!(t.get_flat(0), Some(Value::Int(1)));
                assert_eq!(t.get_flat(3), Some(Value::Int(4)));
            }
            _ => panic!("Expected tensor"),
        }
    }

    #[test]
    fn invariant_closure_apply() {
        let result = eval(&Expr::app(
            Expr::lam(Expr::mul(Expr::idx(0), Expr::int(2))),
            Expr::int(21),
        )).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn invariant_closure_partial_application() {
        let add_fn = Expr::lam(Expr::lam(Expr::add(Expr::idx(1), Expr::idx(0))));
        let add5 = Expr::app(add_fn, Expr::int(5));
        let result = eval(&Expr::app(add5, Expr::int(3))).unwrap();
        assert_eq!(result, Value::Int(8));
    }

    #[test]
    fn invariant_closure_captures_env() {
        let expr = Expr::let_(
            Pattern::var("x"), Expr::int(10),
            Expr::app(Expr::lam(Expr::add(Expr::idx(0), Expr::idx(1))), Expr::int(5)),
        );
        assert_eq!(eval(&expr).unwrap(), Value::Int(15));
    }

    #[test]
    fn invariant_closure_eq() {
        let c1 = Value::closure(1, Expr::idx(0), Env::new());
        let c2 = Value::closure(1, Expr::idx(0), Env::new());
        assert_eq!(c1, c2);
    }

    #[test]
    fn invariant_closure_is_callable() {
        let v = Value::closure(1, Expr::idx(0), Env::new());
        assert!(v.is_callable());
    }

    // ============ Rc sharing tests ============

    #[test]
    fn sharing_closure_clone_shares_rc() {
        let v1 = Value::closure(1, Expr::idx(0), Env::new());
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::Closure(a), Value::Closure(b)) => assert!(Rc::ptr_eq(a, b)),
            _ => panic!("Expected closures"),
        }
    }

    #[test]
    fn sharing_tensor_clone_shares_rc() {
        let v1 = Value::Tensor(Rc::new(Tensor::from_ints(vec![1, 2, 3])));
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::Tensor(a), Value::Tensor(b)) => assert!(Rc::ptr_eq(a, b)),
            _ => panic!("Expected tensors"),
        }
    }

    #[test]
    fn sharing_string_value_is_rc_tensor() {
        let v1 = Value::string("hello");
        let v2 = v1.clone();
        match (&v1, &v2) {
            (Value::Tensor(a), Value::Tensor(b)) => assert!(Rc::ptr_eq(a, b)),
            _ => panic!("Expected tensors"),
        }
    }

    // ============ Complex / Quaternion Foundation Tests ============

    #[test]
    fn test_complex_literal_i() {
        let expr = Expr::Lit(Literal::ImagI(3.0));
        assert_eq!(eval(&expr).unwrap(), Value::Complex(0.0, 3.0));
    }

    #[test]
    fn test_complex_literal_j() {
        let expr = Expr::Lit(Literal::ImagJ(2.0));
        assert_eq!(eval(&expr).unwrap(), Value::Quaternion(0.0, 0.0, 2.0, 0.0));
    }

    #[test]
    fn test_complex_literal_k() {
        let expr = Expr::Lit(Literal::ImagK(5.0));
        assert_eq!(eval(&expr).unwrap(), Value::Quaternion(0.0, 0.0, 0.0, 5.0));
    }

    #[test]
    fn test_complex_display() {
        assert_eq!(format!("{}", Value::Complex(3.0, 4.0)), "3 + 4𝕚");
        assert_eq!(format!("{}", Value::Complex(3.0, -4.0)), "3 - 4𝕚");
        assert_eq!(format!("{}", Value::Complex(0.0, 1.0)), "1𝕚");
        assert_eq!(format!("{}", Value::Complex(5.0, 0.0)), "5");
        assert_eq!(format!("{}", Value::Complex(0.0, 0.0)), "0");
    }

    #[test]
    fn test_complex_type_name() {
        assert_eq!(Value::Complex(1.0, 2.0).type_name(), "Complex");
        assert_eq!(Value::Quaternion(1.0, 0.0, 0.0, 0.0).type_name(), "Quaternion");
    }

    #[test]
    fn test_complex_is_numeric() {
        assert!(Value::Complex(1.0, 2.0).is_numeric());
        assert!(Value::Quaternion(1.0, 0.0, 0.0, 0.0).is_numeric());
    }

    #[test]
    fn test_complex_deep_eq() {
        assert!(Value::Complex(1.0, 2.0).deep_eq(&Value::Complex(1.0, 2.0)));
        assert!(!Value::Complex(1.0, 2.0).deep_eq(&Value::Complex(1.0, 3.0)));
        assert!(Value::Quaternion(1.0, 2.0, 3.0, 4.0).deep_eq(&Value::Quaternion(1.0, 2.0, 3.0, 4.0)));
        assert!(!Value::Quaternion(1.0, 2.0, 3.0, 4.0).deep_eq(&Value::Quaternion(1.0, 2.0, 3.0, 5.0)));
    }

    #[test]
    fn test_complex_coerce_float() {
        assert_eq!(Value::Complex(3.0, 0.0).coerce_float(), Some(3.0));
        assert_eq!(Value::Complex(3.0, 1.0).coerce_float(), None);
    }

    #[test]
    fn test_complex_coerce_complex() {
        assert_eq!(Value::Int(5).coerce_complex(), Some((5.0, 0.0)));
        assert_eq!(Value::float(3.14).coerce_complex(), Some((3.14, 0.0)));
        assert_eq!(Value::Complex(1.0, 2.0).coerce_complex(), Some((1.0, 2.0)));
    }

    #[test]
    fn test_quaternion_coerce() {
        assert_eq!(Value::Int(5).coerce_quaternion(), Some((5.0, 0.0, 0.0, 0.0)));
        assert_eq!(Value::Complex(1.0, 2.0).coerce_quaternion(), Some((1.0, 2.0, 0.0, 0.0)));
        assert_eq!(Value::Quaternion(1.0, 2.0, 3.0, 4.0).coerce_quaternion(), Some((1.0, 2.0, 3.0, 4.0)));
    }

    // ── Phase 3: Complex + Quaternion arithmetic ──

    fn assert_complex_approx(result: &Value, re: f64, im: f64, tol: f64, label: &str) {
        match result {
            Value::Complex(r, i) => {
                assert!((*r - re).abs() < tol, "{}: re = {}, expected {}", label, r, re);
                assert!((*i - im).abs() < tol, "{}: im = {}, expected {}", label, i, im);
            }
            other => panic!("{}: expected Complex, got {:?}", label, other),
        }
    }

    fn assert_quat_approx(result: &Value, w: f64, i: f64, j: f64, k: f64, tol: f64, label: &str) {
        match result {
            Value::Quaternion(rw, ri, rj, rk) => {
                assert!((*rw - w).abs() < tol, "{}: w = {}, expected {}", label, rw, w);
                assert!((*ri - i).abs() < tol, "{}: i = {}, expected {}", label, ri, i);
                assert!((*rj - j).abs() < tol, "{}: j = {}, expected {}", label, rj, j);
                assert!((*rk - k).abs() < tol, "{}: k = {}, expected {}", label, rk, k);
            }
            other => panic!("{}: expected Quaternion, got {:?}", label, other),
        }
    }

    #[test]
    fn test_complex_add() {
        // (3+4i) + (1+2i) = 4+6i
        let expr = Expr::add(
            Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))),
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 4.0, 6.0, 1e-10, "complex add");
    }

    #[test]
    fn test_complex_sub() {
        // (3+4i) - (1+2i) = 2+2i
        let expr = Expr::sub(
            Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))),
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 2.0, 2.0, 1e-10, "complex sub");
    }

    #[test]
    fn test_complex_mul() {
        // (3+4i)(1+2i) = 3*1 - 4*2 + (3*2 + 4*1)i = -5+10i
        let expr = Expr::mul(
            Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))),
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), -5.0, 10.0, 1e-10, "complex mul");
    }

    #[test]
    fn test_complex_mul_i_squared() {
        // i * i = -1
        let expr = Expr::mul(
            Expr::lit(Literal::ImagI(1.0)),
            Expr::lit(Literal::ImagI(1.0)),
        );
        let result = eval(&expr).unwrap();
        assert_complex_approx(&result, -1.0, 0.0, 1e-10, "i*i");
    }

    #[test]
    fn test_complex_div() {
        // (3+4i)/(1+2i) = (3+8+4-6i)/(1+4) = (11-2i)/5 = 2.2-0.4i
        let expr = Expr::div(
            Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))),
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 2.2, -0.4, 1e-10, "complex div");
    }

    #[test]
    fn test_complex_abs() {
        // |3+4i| = 5.0
        let expr = Expr::UnaryOp(
            UnaryOp::Abs,
            Box::new(Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0)))),
        );
        assert_eq!(eval(&expr).unwrap(), Value::float(5.0));
    }

    #[test]
    fn test_complex_negate() {
        // -(3+4i) = -3-4i
        let expr = Expr::UnaryOp(
            UnaryOp::Neg,
            Box::new(Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0)))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), -3.0, -4.0, 1e-10, "complex negate");
    }

    #[test]
    fn test_complex_auto_promote() {
        // 5 + 3i = Complex(5, 3)
        let expr = Expr::add(Expr::int(5), Expr::lit(Literal::ImagI(3.0)));
        assert_complex_approx(&eval(&expr).unwrap(), 5.0, 3.0, 1e-10, "auto-promote");
    }

    #[test]
    fn test_complex_exp_euler() {
        // exp(πi) ≈ -1 + 0i
        let pi = std::f64::consts::PI;
        let expr = Expr::UnaryOp(
            UnaryOp::Exp,
            Box::new(Expr::lit(Literal::ImagI(pi))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), -1.0, 0.0, 1e-10, "euler identity");
    }

    #[test]
    fn test_complex_sin() {
        // sin(i) = i·sinh(1)
        let expr = Expr::UnaryOp(
            UnaryOp::Sin,
            Box::new(Expr::lit(Literal::ImagI(1.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 0.0, 1.0_f64.sinh(), 1e-10, "sin(i)");
    }

    #[test]
    fn test_complex_cos() {
        // cos(i) = cosh(1)
        let expr = Expr::UnaryOp(
            UnaryOp::Cos,
            Box::new(Expr::lit(Literal::ImagI(1.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 1.0_f64.cosh(), 0.0, 1e-10, "cos(i)");
    }

    #[test]
    fn test_complex_ln() {
        // ln(i) = πi/2
        let expr = Expr::UnaryOp(
            UnaryOp::Ln,
            Box::new(Expr::lit(Literal::ImagI(1.0))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 0.0, std::f64::consts::FRAC_PI_2, 1e-10, "ln(i)");
    }

    #[test]
    fn test_complex_sqrt_negative() {
        // sqrt(-4) = 2i
        let expr = Expr::UnaryOp(
            UnaryOp::Sqrt,
            Box::new(Expr::UnaryOp(UnaryOp::Neg, Box::new(Expr::int(4)))),
        );
        assert_complex_approx(&eval(&expr).unwrap(), 0.0, 2.0, 1e-10, "sqrt(-4)");
    }

    #[test]
    fn test_quaternion_ij_eq_k() {
        // i × j = k
        let expr = Expr::mul(
            Expr::lit(Literal::ImagI(1.0)),
            Expr::lit(Literal::ImagJ(1.0)),
        );
        assert_quat_approx(&eval(&expr).unwrap(), 0.0, 0.0, 0.0, 1.0, 1e-10, "i*j=k");
    }

    #[test]
    fn test_quaternion_ji_eq_neg_k() {
        // j × i = -k
        let expr = Expr::mul(
            Expr::lit(Literal::ImagJ(1.0)),
            Expr::lit(Literal::ImagI(1.0)),
        );
        assert_quat_approx(&eval(&expr).unwrap(), 0.0, 0.0, 0.0, -1.0, 1e-10, "j*i=-k");
    }

    #[test]
    fn test_quaternion_jk_eq_i() {
        // j × k = i
        let expr = Expr::mul(
            Expr::lit(Literal::ImagJ(1.0)),
            Expr::lit(Literal::ImagK(1.0)),
        );
        assert_quat_approx(&eval(&expr).unwrap(), 0.0, 1.0, 0.0, 0.0, 1e-10, "j*k=i");
    }

    #[test]
    fn test_quaternion_ijk_eq_neg1() {
        // i × j × k = -1
        let expr = Expr::mul(
            Expr::mul(
                Expr::lit(Literal::ImagI(1.0)),
                Expr::lit(Literal::ImagJ(1.0)),
            ),
            Expr::lit(Literal::ImagK(1.0)),
        );
        assert_quat_approx(&eval(&expr).unwrap(), -1.0, 0.0, 0.0, 0.0, 1e-10, "ijk=-1");
    }

    #[test]
    fn test_quaternion_add() {
        // (1+2i+3j+4k) + (5+6i+7j+8k) = 6+8i+10j+12k
        let q1 = Expr::add(
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
            Expr::add(Expr::lit(Literal::ImagJ(3.0)), Expr::lit(Literal::ImagK(4.0))),
        );
        let q2 = Expr::add(
            Expr::add(Expr::int(5), Expr::lit(Literal::ImagI(6.0))),
            Expr::add(Expr::lit(Literal::ImagJ(7.0)), Expr::lit(Literal::ImagK(8.0))),
        );
        let expr = Expr::add(q1, q2);
        assert_quat_approx(&eval(&expr).unwrap(), 6.0, 8.0, 10.0, 12.0, 1e-10, "quat add");
    }

    #[test]
    fn test_quaternion_norm() {
        // |1+2i+3j+4k| = √(1+4+9+16) = √30
        let q = Expr::add(
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
            Expr::add(Expr::lit(Literal::ImagJ(3.0)), Expr::lit(Literal::ImagK(4.0))),
        );
        let expr = Expr::UnaryOp(UnaryOp::Abs, Box::new(q));
        assert_eq!(eval(&expr).unwrap(), Value::float(30.0_f64.sqrt()));
    }

    // ── Phase 4: re, im, conj, arg primitives ──

    #[test]
    fn test_re_complex() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("re"), Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))));
        assert_eq!(e.eval(&expr).unwrap(), Value::float(3.0));
    }

    #[test]
    fn test_im_complex() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("im"), Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))));
        assert_eq!(e.eval(&expr).unwrap(), Value::float(4.0));
    }

    #[test]
    fn test_conj_complex() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("conj"), Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0))));
        assert_complex_approx(&e.eval(&expr).unwrap(), 3.0, -4.0, 1e-10, "conj(3+4i)");
    }

    #[test]
    fn test_arg_complex() {
        let mut e = Evaluator::new();
        // arg(i) = π/2
        let expr = Expr::app(Expr::name("arg"), Expr::lit(Literal::ImagI(1.0)));
        assert_approx(
            match e.eval(&expr).unwrap() { Value::Float(f) => f.0, v => panic!("expected Float, got {:?}", v) },
            std::f64::consts::FRAC_PI_2, 1e-10, "arg(i)"
        );
    }

    #[test]
    fn test_conj_quaternion() {
        let mut e = Evaluator::new();
        let q = Expr::add(
            Expr::add(Expr::int(1), Expr::lit(Literal::ImagI(2.0))),
            Expr::add(Expr::lit(Literal::ImagJ(3.0)), Expr::lit(Literal::ImagK(4.0))),
        );
        let expr = Expr::app(Expr::name("conj"), q);
        assert_quat_approx(&e.eval(&expr).unwrap(), 1.0, -2.0, -3.0, -4.0, 1e-10, "conj(quat)");
    }

    #[test]
    fn test_re_of_real() {
        let mut e = Evaluator::new();
        assert_eq!(e.eval(&Expr::app(Expr::name("re"), Expr::float(5.0))).unwrap(), Value::float(5.0));
    }

    #[test]
    fn test_conj_of_real() {
        let mut e = Evaluator::new();
        assert_eq!(e.eval(&Expr::app(Expr::name("conj"), Expr::float(5.0))).unwrap(), Value::float(5.0));
    }

    #[test]
    fn test_z_times_conj_z() {
        // z × conj(z) = |z|² (real)
        let mut e = Evaluator::new();
        let z = Expr::add(Expr::int(3), Expr::lit(Literal::ImagI(4.0)));
        let conj_z = Expr::app(Expr::name("conj"), z.clone());
        let expr = Expr::mul(z, conj_z);
        // 3² + 4² = 25, should be Complex(25, 0)
        assert_complex_approx(&e.eval(&expr).unwrap(), 25.0, 0.0, 1e-10, "z*conj(z)");
    }

    // ── Matrix utility tests ──

    fn mat2x2(a: f64, b: f64, c: f64, d: f64) -> Expr {
        Expr::array(vec![
            Expr::array(vec![Expr::float(a), Expr::float(b)]),
            Expr::array(vec![Expr::float(c), Expr::float(d)]),
        ])
    }

    fn mat3x3(vals: [f64; 9]) -> Expr {
        Expr::array(vec![
            Expr::array(vec![Expr::float(vals[0]), Expr::float(vals[1]), Expr::float(vals[2])]),
            Expr::array(vec![Expr::float(vals[3]), Expr::float(vals[4]), Expr::float(vals[5])]),
            Expr::array(vec![Expr::float(vals[6]), Expr::float(vals[7]), Expr::float(vals[8])]),
        ])
    }

    fn vec_expr(vals: &[f64]) -> Expr {
        Expr::array(vals.iter().map(|&v| Expr::float(v)).collect())
    }

    fn assert_tensor_float(result: &Value, idx: &[usize], expected: f64, tol: f64, label: &str) {
        match result {
            Value::Tensor(t) => {
                let v = t.get(idx).unwrap_or_else(|| panic!("{}: index {:?} out of bounds", label, idx));
                let f = v.coerce_float().unwrap_or_else(|| panic!("{}: not numeric at {:?}", label, idx));
                assert!((f - expected).abs() < tol, "{}: at {:?} got {}, expected {}", label, idx, f, expected);
            }
            other => panic!("{}: expected Tensor, got {:?}", label, other),
        }
    }

    // Phase 1: trace + eye

    #[test]
    fn test_trace_general() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("trace"), mat2x2(1.0, 2.0, 3.0, 4.0));
        assert_eq!(e.eval(&expr).unwrap(), Value::float(5.0));
    }

    #[test]
    fn test_trace_identity_3x3() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("trace"), Expr::app(Expr::name("eye"), Expr::int(3)));
        assert_eq!(e.eval(&expr).unwrap(), Value::float(3.0));
    }

    #[test]
    fn test_trace_non_square_error() {
        let mut e = Evaluator::new();
        let mat = Expr::array(vec![
            Expr::array(vec![Expr::float(1.0), Expr::float(2.0), Expr::float(3.0)]),
            Expr::array(vec![Expr::float(4.0), Expr::float(5.0), Expr::float(6.0)]),
        ]);
        assert!(e.eval(&Expr::app(Expr::name("trace"), mat)).is_err());
    }

    #[test]
    fn test_eye_3() {
        let mut e = Evaluator::new();
        let result = e.eval(&Expr::app(Expr::name("eye"), Expr::int(3))).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.shape, vec![3, 3]);
            for i in 0..3 { for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.get(&[i, j]).unwrap().coerce_float().unwrap(), expected);
            }}
        } else { panic!("Expected tensor"); }
    }

    #[test]
    fn test_eye_1() {
        let mut e = Evaluator::new();
        let result = e.eval(&Expr::app(Expr::name("eye"), Expr::int(1))).unwrap();
        assert_tensor_float(&result, &[0, 0], 1.0, 1e-15, "eye(1)");
    }

    // Phase 2: diag

    #[test]
    fn test_diag_vec_to_matrix() {
        let mut e = Evaluator::new();
        let result = e.eval(&Expr::app(Expr::name("diag"), vec_expr(&[1.0, 2.0, 3.0]))).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.shape, vec![3, 3]);
        } else { panic!("Expected tensor"); }
        assert_tensor_float(&result, &[0, 0], 1.0, 1e-15, "diag[0,0]");
        assert_tensor_float(&result, &[1, 1], 2.0, 1e-15, "diag[1,1]");
        assert_tensor_float(&result, &[2, 2], 3.0, 1e-15, "diag[2,2]");
        assert_tensor_float(&result, &[0, 1], 0.0, 1e-15, "diag[0,1]");
    }

    #[test]
    fn test_diag_matrix_to_vec() {
        let mut e = Evaluator::new();
        let result = e.eval(&Expr::app(Expr::name("diag"), mat2x2(1.0, 2.0, 3.0, 4.0))).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.shape, vec![2]);
            assert_eq!(t.get(&[0]).unwrap().coerce_float().unwrap(), 1.0);
            assert_eq!(t.get(&[1]).unwrap().coerce_float().unwrap(), 4.0);
        } else { panic!("Expected tensor"); }
    }

    // Phase 3: det

    #[test]
    fn test_det_2x2() {
        let mut e = Evaluator::new();
        let f = e.eval(&Expr::app(Expr::name("det"), mat2x2(1.0, 2.0, 3.0, 4.0))).unwrap()
            .coerce_float().unwrap();
        assert!((f - (-2.0)).abs() < 1e-10, "det = {}, expected -2", f);
    }

    #[test]
    fn test_det_3x3() {
        let mut e = Evaluator::new();
        let f = e.eval(&Expr::app(Expr::name("det"), mat3x3([6.0,1.0,1.0, 4.0,-2.0,5.0, 2.0,8.0,7.0]))).unwrap()
            .coerce_float().unwrap();
        assert!((f - (-306.0)).abs() < 1e-8, "det = {}, expected -306", f);
    }

    #[test]
    fn test_det_identity() {
        let mut e = Evaluator::new();
        let f = e.eval(&Expr::app(Expr::name("det"), Expr::app(Expr::name("eye"), Expr::int(3)))).unwrap()
            .coerce_float().unwrap();
        assert!((f - 1.0).abs() < 1e-12, "det(I) = {}, expected 1", f);
    }

    #[test]
    fn test_det_singular() {
        let mut e = Evaluator::new();
        let f = e.eval(&Expr::app(Expr::name("det"), mat2x2(1.0, 2.0, 2.0, 4.0))).unwrap()
            .coerce_float().unwrap();
        assert!(f.abs() < 1e-10, "det(singular) = {}, expected 0", f);
    }

    // Phase 4: inv

    #[test]
    fn test_inv_2x2() {
        let mut e = Evaluator::new();
        let result = e.eval(&Expr::app(Expr::name("inv"), mat2x2(1.0, 2.0, 3.0, 4.0))).unwrap();
        let tol = 1e-10;
        assert_tensor_float(&result, &[0, 0], -2.0, tol, "inv[0,0]");
        assert_tensor_float(&result, &[0, 1], 1.0, tol, "inv[0,1]");
        assert_tensor_float(&result, &[1, 0], 1.5, tol, "inv[1,0]");
        assert_tensor_float(&result, &[1, 1], -0.5, tol, "inv[1,1]");
    }

    #[test]
    fn test_inv_identity() {
        let mut e = Evaluator::new();
        let result = e.eval(&Expr::app(Expr::name("inv"), Expr::app(Expr::name("eye"), Expr::int(3)))).unwrap();
        for i in 0..3 { for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_tensor_float(&result, &[i, j], expected, 1e-10, &format!("inv(I)[{},{}]", i, j));
        }}
    }

    #[test]
    fn test_inv_singular_error() {
        let mut e = Evaluator::new();
        assert!(e.eval(&Expr::app(Expr::name("inv"), mat2x2(1.0, 2.0, 2.0, 4.0))).is_err());
    }

    #[test]
    fn test_inv_roundtrip() {
        let mut e = Evaluator::new();
        let a = mat2x2(2.0, 1.0, 5.0, 3.0);
        let inv_a = Expr::app(Expr::name("inv"), a.clone());
        let product = Expr::app(Expr::app(Expr::name("matmul"), a), inv_a);
        let result = e.eval(&product).unwrap();
        for i in 0..2 { for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_tensor_float(&result, &[i, j], expected, 1e-10, &format!("A*inv(A)[{},{}]", i, j));
        }}
    }

    // Phase 5: solve

    #[test]
    fn test_solve_2x2() {
        let mut e = Evaluator::new();
        let a = mat2x2(2.0, 1.0, 5.0, 3.0);
        let b = vec_expr(&[4.0, 7.0]);
        let result = e.eval(&Expr::app(Expr::app(Expr::name("solve"), a), b)).unwrap();
        assert_tensor_float(&result, &[0], 5.0, 1e-10, "x[0]");
        assert_tensor_float(&result, &[1], -6.0, 1e-10, "x[1]");
    }

    #[test]
    fn test_solve_3x3() {
        let mut e = Evaluator::new();
        let a = mat3x3([1.0,1.0,1.0, 0.0,2.0,5.0, 2.0,5.0,-1.0]);
        let b = vec_expr(&[6.0, -4.0, 27.0]);
        let result = e.eval(&Expr::app(Expr::app(Expr::name("solve"), a), b)).unwrap();
        assert_tensor_float(&result, &[0], 5.0, 1e-10, "x[0]");
        assert_tensor_float(&result, &[1], 3.0, 1e-10, "x[1]");
        assert_tensor_float(&result, &[2], -2.0, 1e-10, "x[2]");
    }

    #[test]
    fn test_solve_singular_error() {
        let mut e = Evaluator::new();
        let a = mat2x2(1.0, 2.0, 2.0, 4.0);
        let b = vec_expr(&[1.0, 2.0]);
        assert!(e.eval(&Expr::app(Expr::app(Expr::name("solve"), a), b)).is_err());
    }

    #[test]
    fn test_solve_dimension_mismatch() {
        let mut e = Evaluator::new();
        let a = mat2x2(1.0, 2.0, 3.0, 4.0);
        let b = vec_expr(&[1.0, 2.0, 3.0]);
        assert!(e.eval(&Expr::app(Expr::app(Expr::name("solve"), a), b)).is_err());
    }

    // Phase 6: solveWith + QR

    #[test]
    fn test_solve_with_lu_explicit() {
        let mut e = Evaluator::new();
        let a = mat2x2(2.0, 1.0, 5.0, 3.0);
        let b = vec_expr(&[4.0, 7.0]);
        let method = Expr::Lit(Literal::string("lu"));
        let expr = Expr::app(Expr::app(Expr::app(Expr::name("solveWith"), a), b), method);
        let result = e.eval(&expr).unwrap();
        assert_tensor_float(&result, &[0], 5.0, 1e-10, "lu x[0]");
        assert_tensor_float(&result, &[1], -6.0, 1e-10, "lu x[1]");
    }

    #[test]
    fn test_solve_with_qr() {
        let mut e = Evaluator::new();
        let a = mat2x2(2.0, 1.0, 5.0, 3.0);
        let b = vec_expr(&[4.0, 7.0]);
        let method = Expr::Lit(Literal::string("qr"));
        let expr = Expr::app(Expr::app(Expr::app(Expr::name("solveWith"), a), b), method);
        let result = e.eval(&expr).unwrap();
        assert_tensor_float(&result, &[0], 5.0, 1e-8, "qr x[0]");
        assert_tensor_float(&result, &[1], -6.0, 1e-8, "qr x[1]");
    }

    #[test]
    fn test_solve_with_qr_overdetermined() {
        let mut e = Evaluator::new();
        let a = Expr::array(vec![
            Expr::array(vec![Expr::float(1.0), Expr::float(1.0)]),
            Expr::array(vec![Expr::float(1.0), Expr::float(2.0)]),
            Expr::array(vec![Expr::float(1.0), Expr::float(3.0)]),
        ]);
        let b = vec_expr(&[1.0, 2.0, 2.0]);
        let method = Expr::Lit(Literal::string("qr"));
        let expr = Expr::app(Expr::app(Expr::app(Expr::name("solveWith"), a), b), method);
        let result = e.eval(&expr).unwrap();
        assert_tensor_float(&result, &[0], 2.0 / 3.0, 1e-8, "lstsq x[0]");
        assert_tensor_float(&result, &[1], 0.5, 1e-8, "lstsq x[1]");
    }

    #[test]
    fn test_solve_with_unknown_method() {
        let mut e = Evaluator::new();
        let a = mat2x2(1.0, 0.0, 0.0, 1.0);
        let b = vec_expr(&[1.0, 2.0]);
        let method = Expr::Lit(Literal::string("nonsense"));
        let expr = Expr::app(Expr::app(Expr::app(Expr::name("solveWith"), a), b), method);
        assert!(e.eval(&expr).is_err());
    }

    // ── Eigenvalue tests ──

    fn assert_eigenvalue_approx(val: &Value, idx: usize, expected_re: f64, expected_im: f64, tol: f64, label: &str) {
        match val {
            Value::Tensor(t) => {
                let elem = t.get_flat(idx).expect(&format!("{}: index {} out of bounds", label, idx));
                match elem {
                    Value::Float(f) => {
                        assert!(expected_im.abs() < tol, "{}: expected complex but got Float({})", label, f.0);
                        assert!((f.0 - expected_re).abs() < tol, "{}: expected re={}, got {} (diff {})", label, expected_re, f.0, (f.0 - expected_re).abs());
                    }
                    Value::Complex(re, im) => {
                        assert!((re - expected_re).abs() < tol, "{}: expected re={}, got {} (diff {})", label, expected_re, re, (re - expected_re).abs());
                        assert!((im - expected_im).abs() < tol, "{}: expected im={}, got {} (diff {})", label, expected_im, im, (im - expected_im).abs());
                    }
                    other => panic!("{}: expected Float or Complex, got {:?}", label, other),
                }
            }
            _ => panic!("{}: expected Tensor, got {:?}", label, val),
        }
    }

    #[test]
    fn test_eig_identity() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("eig"), Expr::app(Expr::name("eye"), Expr::int(3)));
        let result = e.eval(&expr).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.shape, vec![3]);
            // All eigenvalues should be 1.0
            for i in 0..3 {
                assert_eigenvalue_approx(&result, i, 1.0, 0.0, 1e-10, "eig(eye(3))");
            }
            // Should be Float tensor (all real)
            assert!(matches!(t.data, TensorData::Float(_)), "identity eigenvalues should be Float tensor");
        } else {
            panic!("expected Tensor");
        }
    }

    #[test]
    fn test_eig_1x1() {
        let mut e = Evaluator::new();
        let mat = Expr::array(vec![Expr::array(vec![Expr::float(7.0)])]);
        let expr = Expr::app(Expr::name("eig"), mat);
        let result = e.eval(&expr).unwrap();
        assert_eigenvalue_approx(&result, 0, 7.0, 0.0, 1e-10, "eig([[7]])");
    }

    #[test]
    fn test_eig_symmetric_2x2() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("eig"), mat2x2(2.0, 1.0, 1.0, 2.0));
        let result = e.eval(&expr).unwrap();
        // Eigenvalues should be 3 and 1 (sorted descending)
        assert_eigenvalue_approx(&result, 0, 3.0, 0.0, 1e-10, "eig sym 2x2 [0]");
        assert_eigenvalue_approx(&result, 1, 1.0, 0.0, 1e-10, "eig sym 2x2 [1]");
    }

    #[test]
    fn test_eig_diagonal() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("eig"),
            Expr::app(Expr::name("diag"), Expr::array(vec![Expr::float(1.0), Expr::float(2.0), Expr::float(3.0)])));
        let result = e.eval(&expr).unwrap();
        // Sorted descending: 3, 2, 1
        assert_eigenvalue_approx(&result, 0, 3.0, 0.0, 1e-10, "eig diag [0]");
        assert_eigenvalue_approx(&result, 1, 2.0, 0.0, 1e-10, "eig diag [1]");
        assert_eigenvalue_approx(&result, 2, 1.0, 0.0, 1e-10, "eig diag [2]");
    }

    #[test]
    fn test_eig_rotation_complex() {
        let mut e = Evaluator::new();
        // [[0, -1], [1, 0]] has eigenvalues i and -i
        let expr = Expr::app(Expr::name("eig"), mat2x2(0.0, -1.0, 1.0, 0.0));
        let result = e.eval(&expr).unwrap();
        if let Value::Tensor(t) = &result {
            assert!(matches!(t.data, TensorData::Generic(_)), "rotation eigenvalues should be Generic (complex)");
        }
        // Eigenvalues are ±i, sorted by real part then imaginary descending
        assert_eigenvalue_approx(&result, 0, 0.0, 1.0, 1e-10, "eig rot [0]");
        assert_eigenvalue_approx(&result, 1, 0.0, -1.0, 1e-10, "eig rot [1]");
    }

    #[test]
    fn test_eig_trace_invariant() {
        // Sum of eigenvalues = trace(A)
        let mut e = Evaluator::new();
        let a = mat3x3([2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0]);
        let trace_expr = Expr::app(Expr::name("trace"), a.clone());
        let trace_val = e.eval(&trace_expr).unwrap().coerce_float().unwrap();

        let eig_expr = Expr::app(Expr::name("eig"), a);
        let eig_result = e.eval(&eig_expr).unwrap();
        let mut eig_sum = 0.0;
        if let Value::Tensor(t) = &eig_result {
            for i in 0..t.shape[0] {
                let v = t.get_flat(i).unwrap();
                eig_sum += v.coerce_float().unwrap();
            }
        }
        assert_approx(eig_sum, trace_val, 1e-10, "sum(eig) == trace");
    }

    #[test]
    fn test_eig_det_invariant() {
        // Product of eigenvalues = det(A)
        let mut e = Evaluator::new();
        let a = mat2x2(4.0, 2.0, 1.0, 3.0);
        let det_expr = Expr::app(Expr::name("det"), a.clone());
        let det_val = e.eval(&det_expr).unwrap().coerce_float().unwrap();

        let eig_expr = Expr::app(Expr::name("eig"), a);
        let eig_result = e.eval(&eig_expr).unwrap();
        let mut eig_prod = 1.0;
        if let Value::Tensor(t) = &eig_result {
            for i in 0..t.shape[0] {
                let v = t.get_flat(i).unwrap();
                eig_prod *= v.coerce_float().unwrap();
            }
        }
        assert_approx(eig_prod, det_val, 1e-10, "prod(eig) == det");
    }

    #[test]
    fn test_eig_non_square_error() {
        let mut e = Evaluator::new();
        let mat = Expr::array(vec![
            Expr::array(vec![Expr::float(1.0), Expr::float(2.0), Expr::float(3.0)]),
            Expr::array(vec![Expr::float(4.0), Expr::float(5.0), Expr::float(6.0)]),
        ]);
        assert!(e.eval(&Expr::app(Expr::name("eig"), mat)).is_err());
    }

    // ── Eigenvector tests ──

    #[test]
    fn test_eigvecs_returns_tuple() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("eigvecs"), Expr::app(Expr::name("eye"), Expr::int(2)));
        let result = e.eval(&expr).unwrap();
        if let Value::Tuple(vs) = &result {
            assert_eq!(vs.len(), 2, "eigvecs should return a 2-tuple");
        } else {
            panic!("expected Tuple, got {:?}", result);
        }
    }

    #[test]
    fn test_eigvecs_identity() {
        let mut e = Evaluator::new();
        let expr = Expr::app(Expr::name("eigvecs"), Expr::app(Expr::name("eye"), Expr::int(3)));
        let result = e.eval(&expr).unwrap();
        if let Value::Tuple(vs) = &result {
            // All eigenvalues should be 1.0
            for i in 0..3 {
                assert_eigenvalue_approx(&vs[0], i, 1.0, 0.0, 1e-10, "eigvecs(eye(3)) eval");
            }
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn test_eigvecs_av_equals_lambda_v() {
        // Fundamental property: A*v = λ*v for each eigenvalue/eigenvector pair
        let mut e = Evaluator::new();
        let a_data = [4.0, 1.0, 2.0, 3.0];
        let a_expr = mat2x2(a_data[0], a_data[1], a_data[2], a_data[3]);
        let expr = Expr::app(Expr::name("eigvecs"), a_expr);
        let result = e.eval(&expr).unwrap();
        if let Value::Tuple(vs) = &result {
            let evals = &vs[0];
            let evecs = &vs[1];
            if let (Value::Tensor(eval_t), Value::Tensor(evec_t)) = (evals, evecs) {
                let n = 2;
                for col in 0..n {
                    let lambda = eval_t.get_flat(col).unwrap().coerce_float().unwrap();
                    // Extract eigenvector column
                    let mut v = vec![0.0; n];
                    for row in 0..n { v[row] = evec_t.get(&[row, col]).unwrap().coerce_float().unwrap(); }
                    // Compute A*v
                    let mut av = vec![0.0; n];
                    for i in 0..n {
                        for j in 0..n {
                            av[i] += a_data[i * n + j] * v[j];
                        }
                    }
                    // Check A*v = λ*v
                    for i in 0..n {
                        assert_approx(av[i], lambda * v[i], 1e-8, &format!("A*v = λ*v, col={}, row={}", col, i));
                    }
                }
            }
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn test_eigvecs_symmetric_orthogonal() {
        // Eigenvectors of symmetric matrix should be orthogonal
        let mut e = Evaluator::new();
        let a_expr = mat2x2(2.0, 1.0, 1.0, 2.0);
        let expr = Expr::app(Expr::name("eigvecs"), a_expr);
        let result = e.eval(&expr).unwrap();
        if let Value::Tuple(vs) = &result {
            if let Value::Tensor(evec_t) = &vs[1] {
                let n = 2;
                // Get columns
                let mut v0 = vec![0.0; n];
                let mut v1 = vec![0.0; n];
                for i in 0..n {
                    v0[i] = evec_t.get(&[i, 0]).unwrap().coerce_float().unwrap();
                    v1[i] = evec_t.get(&[i, 1]).unwrap().coerce_float().unwrap();
                }
                let dot: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-8, "eigenvectors should be orthogonal, dot = {}", dot);
            }
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn test_eigvecs_diagonal() {
        let mut e = Evaluator::new();
        let a_expr = Expr::app(Expr::name("diag"), Expr::array(vec![Expr::float(5.0), Expr::float(3.0)]));
        let expr = Expr::app(Expr::name("eigvecs"), a_expr);
        let result = e.eval(&expr).unwrap();
        if let Value::Tuple(vs) = &result {
            // Eigenvalues should be 5 and 3 (sorted descending)
            assert_eigenvalue_approx(&vs[0], 0, 5.0, 0.0, 1e-10, "eigvecs diag eval[0]");
            assert_eigenvalue_approx(&vs[0], 1, 3.0, 0.0, 1e-10, "eigvecs diag eval[1]");
            // Eigenvectors should be unit vectors
            if let Value::Tensor(evec_t) = &vs[1] {
                for col in 0..2 {
                    let mut norm_sq = 0.0;
                    for row in 0..2 {
                        let v = evec_t.get(&[row, col]).unwrap().coerce_float().unwrap();
                        norm_sq += v * v;
                    }
                    assert_approx(norm_sq, 1.0, 1e-10, &format!("eigvec col {} should be unit", col));
                }
            }
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn test_eigvecs_non_square_error() {
        let mut e = Evaluator::new();
        let mat = Expr::array(vec![
            Expr::array(vec![Expr::float(1.0), Expr::float(2.0), Expr::float(3.0)]),
            Expr::array(vec![Expr::float(4.0), Expr::float(5.0), Expr::float(6.0)]),
        ]);
        assert!(e.eval(&Expr::app(Expr::name("eigvecs"), mat)).is_err());
    }
}
