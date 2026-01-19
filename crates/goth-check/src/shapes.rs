//! Shape checking and constraint solving for Goth
//!
//! This module implements tensor shape checking, the key differentiating feature
//! of the Goth type system. It provides:
//!
//! - Shape constraint representation and collection
//! - Constraint-based shape solving (similar to HM type inference)
//! - Dimension simplification and normalization
//! - Rich error messages for shape mismatches
//!
//! # Example
//!
//! ```goth
//! ╭─ matmul : [m n]F64 → [n p]F64 → [m p]F64
//! ╰─ ₀ @ ₁
//! ```
//!
//! The shape checker ensures that the inner dimensions match (`n` in both
//! arguments) and the result has the correct outer dimensions (`[m p]`).

use std::collections::{HashMap, HashSet};
use goth_ast::shape::{Shape, Dim, DimOp};
use crate::error::{TypeError, TypeResult};

// ============================================================================
// Shape Constraints
// ============================================================================

/// A constraint between two dimensions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeConstraint {
    /// Two dimensions must be equal: d1 = d2
    DimEq(Dim, Dim),
    /// Two shapes must be equal (same rank, corresponding dims equal)
    ShapeEq(Shape, Shape),
    /// A dimension must be positive: d > 0
    Positive(Dim),
    /// A dimension must be at least n: d >= n
    AtLeast(Dim, u64),
}

impl std::fmt::Display for ShapeConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeConstraint::DimEq(d1, d2) => write!(f, "{} = {}", d1, d2),
            ShapeConstraint::ShapeEq(s1, s2) => write!(f, "{} = {}", s1, s2),
            ShapeConstraint::Positive(d) => write!(f, "{} > 0", d),
            ShapeConstraint::AtLeast(d, n) => write!(f, "{} ≥ {}", d, n),
        }
    }
}

// ============================================================================
// Shape Substitution
// ============================================================================

/// Substitution mapping shape variables to dimensions
#[derive(Debug, Clone, Default)]
pub struct ShapeSubst {
    bindings: HashMap<String, Dim>,
}

impl ShapeSubst {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create substitution with single binding
    pub fn singleton(var: impl Into<String>, dim: Dim) -> Self {
        let mut s = Self::new();
        s.bindings.insert(var.into(), dim);
        s
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Get binding for variable
    pub fn get(&self, var: &str) -> Option<&Dim> {
        self.bindings.get(var)
    }

    /// Insert binding
    pub fn insert(&mut self, var: impl Into<String>, dim: Dim) {
        self.bindings.insert(var.into(), dim);
    }

    /// Compose two substitutions: apply s2 after s1
    /// (s2 ∘ s1)(x) = s2(s1(x))
    pub fn compose(s1: ShapeSubst, s2: &ShapeSubst) -> ShapeSubst {
        let mut result = ShapeSubst::new();

        // Apply s2 to all bindings in s1
        for (var, dim) in s1.bindings {
            result.bindings.insert(var, apply_subst(s2, &dim));
        }

        // Add all bindings from s2 (overwriting if present)
        for (var, dim) in &s2.bindings {
            result.bindings.insert(var.clone(), dim.clone());
        }

        result
    }

    /// Get all bound variables
    pub fn vars(&self) -> impl Iterator<Item = &str> {
        self.bindings.keys().map(|s| s.as_str())
    }
}

/// Apply substitution to a dimension
pub fn apply_subst(subst: &ShapeSubst, dim: &Dim) -> Dim {
    match dim {
        Dim::Var(v) => {
            subst.bindings.get(v.as_ref()).cloned().unwrap_or_else(|| dim.clone())
        }
        Dim::Const(_) => dim.clone(),
        Dim::BinOp(l, op, r) => {
            let l_sub = apply_subst(subst, l);
            let r_sub = apply_subst(subst, r);
            simplify_dim(&Dim::BinOp(Box::new(l_sub), *op, Box::new(r_sub)))
        }
    }
}

/// Apply substitution to a shape
pub fn apply_subst_shape(subst: &ShapeSubst, shape: &Shape) -> Shape {
    Shape(shape.0.iter().map(|d| apply_subst(subst, d)).collect())
}

// ============================================================================
// Dimension Simplification
// ============================================================================

/// Simplify a dimension expression
///
/// Applies algebraic simplifications like:
/// - n + 0 → n
/// - n × 1 → n
/// - 0 × n → 0
/// - 3 + 4 → 7 (constant folding)
pub fn simplify_dim(dim: &Dim) -> Dim {
    match dim {
        Dim::Const(_) | Dim::Var(_) => dim.clone(),

        Dim::BinOp(l, op, r) => {
            let l = simplify_dim(l);
            let r = simplify_dim(r);

            // Constant folding
            if let (Dim::Const(n1), Dim::Const(n2)) = (&l, &r) {
                return match op {
                    DimOp::Add => Dim::Const(n1 + n2),
                    DimOp::Sub => Dim::Const(n1.saturating_sub(*n2)),
                    DimOp::Mul => Dim::Const(n1 * n2),
                    DimOp::Div if *n2 != 0 => Dim::Const(n1 / n2),
                    DimOp::Div => dim.clone(), // Keep as-is if div by zero
                };
            }

            // Algebraic simplifications
            match op {
                DimOp::Add => {
                    // n + 0 = n
                    if let Dim::Const(0) = r { return l; }
                    // 0 + n = n
                    if let Dim::Const(0) = l { return r; }
                }
                DimOp::Sub => {
                    // n - 0 = n
                    if let Dim::Const(0) = r { return l; }
                    // n - n = 0 (if same variable)
                    if l == r { return Dim::Const(0); }
                }
                DimOp::Mul => {
                    // n × 1 = n
                    if let Dim::Const(1) = r { return l; }
                    // 1 × n = n
                    if let Dim::Const(1) = l { return r; }
                    // n × 0 = 0
                    if let Dim::Const(0) = r { return Dim::Const(0); }
                    // 0 × n = 0
                    if let Dim::Const(0) = l { return Dim::Const(0); }
                }
                DimOp::Div => {
                    // n / 1 = n
                    if let Dim::Const(1) = r { return l; }
                    // n / n = 1 (if same variable and not zero)
                    if l == r {
                        if let Dim::Const(0) = l {
                            // 0/0 is undefined, keep as-is
                        } else {
                            return Dim::Const(1);
                        }
                    }
                }
            }

            Dim::BinOp(Box::new(l), *op, Box::new(r))
        }
    }
}

// ============================================================================
// Occurs Check
// ============================================================================

/// Check if a shape variable occurs in a dimension
pub fn dim_occurs_in(var: &str, dim: &Dim) -> bool {
    match dim {
        Dim::Var(v) => var == v.as_ref(),
        Dim::Const(_) => false,
        Dim::BinOp(l, _, r) => dim_occurs_in(var, l) || dim_occurs_in(var, r),
    }
}

/// Collect all shape variables in a dimension
pub fn collect_dim_vars(dim: &Dim) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_dim_vars_into(dim, &mut vars);
    vars
}

fn collect_dim_vars_into(dim: &Dim, vars: &mut HashSet<String>) {
    match dim {
        Dim::Var(v) => { vars.insert(v.to_string()); }
        Dim::Const(_) => {}
        Dim::BinOp(l, _, r) => {
            collect_dim_vars_into(l, vars);
            collect_dim_vars_into(r, vars);
        }
    }
}

/// Collect all shape variables in a shape
pub fn collect_shape_vars(shape: &Shape) -> HashSet<String> {
    let mut vars = HashSet::new();
    for dim in &shape.0 {
        collect_dim_vars_into(dim, &mut vars);
    }
    vars
}

// ============================================================================
// Shape Unification
// ============================================================================

/// Unify two shapes, returning a substitution that makes them equal
pub fn unify_shapes(sh1: &Shape, sh2: &Shape) -> TypeResult<ShapeSubst> {
    // Check rank first
    if sh1.rank() != sh2.rank() {
        return Err(TypeError::RankMismatch {
            expected: sh1.rank(),
            found: sh2.rank(),
        });
    }

    // Unify dimension by dimension
    let mut subst = ShapeSubst::new();
    for (i, (d1, d2)) in sh1.0.iter().zip(&sh2.0).enumerate() {
        let d1_sub = apply_subst(&subst, d1);
        let d2_sub = apply_subst(&subst, d2);
        let s = unify_dims(&d1_sub, &d2_sub, i)?;
        subst = ShapeSubst::compose(subst, &s);
    }

    Ok(subst)
}

/// Unify two dimensions
pub fn unify_dims(d1: &Dim, d2: &Dim, position: usize) -> TypeResult<ShapeSubst> {
    // Simplify first
    let d1 = simplify_dim(d1);
    let d2 = simplify_dim(d2);

    match (&d1, &d2) {
        // Same constant: OK
        (Dim::Const(n1), Dim::Const(n2)) if n1 == n2 => {
            Ok(ShapeSubst::new())
        }

        // Different constants: error
        (Dim::Const(n1), Dim::Const(n2)) => {
            Err(TypeError::DimMismatch {
                position,
                expected: n1.to_string(),
                found: n2.to_string(),
            })
        }

        // Variable on left: bind if occurs check passes
        (Dim::Var(v), dim) => {
            // Check if same variable
            if let Dim::Var(v2) = dim {
                if v.as_ref() == v2.as_ref() {
                    return Ok(ShapeSubst::new());
                }
            }

            // Occurs check
            if dim_occurs_in(v, dim) {
                return Err(TypeError::DimMismatch {
                    position,
                    expected: format!("{}", d1),
                    found: format!("{} (infinite dimension: {} occurs in {})", d2, v, dim),
                });
            }

            Ok(ShapeSubst::singleton(v.as_ref(), dim.clone()))
        }

        // Variable on right: symmetric case
        (dim, Dim::Var(v)) => {
            if dim_occurs_in(v, dim) {
                return Err(TypeError::DimMismatch {
                    position,
                    expected: format!("{} (infinite dimension: {} occurs in {})", d1, v, dim),
                    found: format!("{}", d2),
                });
            }

            Ok(ShapeSubst::singleton(v.as_ref(), dim.clone()))
        }

        // BinOp: try structural unification
        (Dim::BinOp(l1, op1, r1), Dim::BinOp(l2, op2, r2)) if op1 == op2 => {
            let s1 = unify_dims(l1, l2, position)?;
            let r1_sub = apply_subst(&s1, r1);
            let r2_sub = apply_subst(&s1, r2);
            let s2 = unify_dims(&r1_sub, &r2_sub, position)?;
            Ok(ShapeSubst::compose(s1, &s2))
        }

        // Try to solve simple equations
        // n = n + m  →  m = 0
        // n = m + k where n is const and m is var  →  m = n - k (if k is const)
        (Dim::Const(n), Dim::BinOp(l, DimOp::Add, r)) |
        (Dim::BinOp(l, DimOp::Add, r), Dim::Const(n)) => {
            // If one side is a var and other is const, we can solve
            match (l.as_ref(), r.as_ref()) {
                (Dim::Var(v), Dim::Const(k)) if *n >= *k => {
                    Ok(ShapeSubst::singleton(v.as_ref(), Dim::Const(n - k)))
                }
                (Dim::Const(k), Dim::Var(v)) if *n >= *k => {
                    Ok(ShapeSubst::singleton(v.as_ref(), Dim::Const(n - k)))
                }
                _ => Err(TypeError::DimMismatch {
                    position,
                    expected: format!("{}", d1),
                    found: format!("{}", d2),
                })
            }
        }

        // Fallback: cannot unify
        _ => Err(TypeError::DimMismatch {
            position,
            expected: format!("{}", d1),
            found: format!("{}", d2),
        })
    }
}

// ============================================================================
// Shape Checker
// ============================================================================

/// Shape checker that collects and solves shape constraints
#[derive(Debug, Clone, Default)]
pub struct ShapeChecker {
    /// Collected constraints
    constraints: Vec<ShapeConstraint>,
    /// Current substitution (incrementally built)
    subst: ShapeSubst,
    /// Fresh variable counter
    fresh_counter: u32,
    /// Shape variables in scope
    scope: HashSet<String>,
}

impl ShapeChecker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a fresh shape variable
    pub fn fresh_var(&mut self) -> Dim {
        let name = format!("_s{}", self.fresh_counter);
        self.fresh_counter += 1;
        Dim::Var(name.into())
    }

    /// Generate a fresh shape with given rank
    pub fn fresh_shape(&mut self, rank: usize) -> Shape {
        Shape((0..rank).map(|_| self.fresh_var()).collect())
    }

    /// Add a shape variable to scope
    pub fn add_to_scope(&mut self, var: impl Into<String>) {
        self.scope.insert(var.into());
    }

    /// Add multiple shape variables to scope
    pub fn add_vars_to_scope(&mut self, vars: impl IntoIterator<Item = impl Into<String>>) {
        for v in vars {
            self.scope.insert(v.into());
        }
    }

    /// Check if a shape variable is in scope
    pub fn in_scope(&self, var: &str) -> bool {
        self.scope.contains(var)
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: ShapeConstraint) {
        self.constraints.push(constraint);
    }

    /// Require two dimensions to be equal
    pub fn require_dim_eq(&mut self, d1: Dim, d2: Dim) -> TypeResult<()> {
        // Try to solve immediately
        let d1_sub = apply_subst(&self.subst, &d1);
        let d2_sub = apply_subst(&self.subst, &d2);

        let new_subst = unify_dims(&d1_sub, &d2_sub, 0)?;
        self.subst = ShapeSubst::compose(std::mem::take(&mut self.subst), &new_subst);

        Ok(())
    }

    /// Require two shapes to be equal
    pub fn require_shape_eq(&mut self, sh1: &Shape, sh2: &Shape) -> TypeResult<()> {
        let sh1_sub = apply_subst_shape(&self.subst, sh1);
        let sh2_sub = apply_subst_shape(&self.subst, sh2);

        let new_subst = unify_shapes(&sh1_sub, &sh2_sub)?;
        self.subst = ShapeSubst::compose(std::mem::take(&mut self.subst), &new_subst);

        Ok(())
    }

    /// Get the current substitution
    pub fn substitution(&self) -> &ShapeSubst {
        &self.subst
    }

    /// Apply current substitution to a dimension
    pub fn resolve_dim(&self, dim: &Dim) -> Dim {
        simplify_dim(&apply_subst(&self.subst, dim))
    }

    /// Apply current substitution to a shape
    pub fn resolve_shape(&self, shape: &Shape) -> Shape {
        Shape(shape.0.iter().map(|d| self.resolve_dim(d)).collect())
    }

    /// Check that a shape is fully resolved (no variables)
    pub fn is_concrete(&self, shape: &Shape) -> bool {
        self.resolve_shape(shape).is_concrete()
    }
}

// ============================================================================
// Rich Error Messages
// ============================================================================

/// Generate a rich error message for a shape mismatch
pub fn shape_mismatch_error(
    context: &str,
    expected: &Shape,
    found: &Shape,
    position: Option<usize>,
) -> TypeError {
    let mut msg = format!("Shape mismatch in {}\n", context);
    msg.push_str(&format!("  Expected: {}\n", expected));
    msg.push_str(&format!("  Found:    {}\n", found));

    if expected.rank() != found.rank() {
        msg.push_str(&format!(
            "  Rank mismatch: expected {} dimensions, found {}\n",
            expected.rank(),
            found.rank()
        ));
        return TypeError::RankMismatch {
            expected: expected.rank(),
            found: found.rank(),
        };
    }

    // Find first mismatching dimension
    for (i, (d1, d2)) in expected.0.iter().zip(&found.0).enumerate() {
        if d1 != d2 {
            let pos = position.unwrap_or(i);
            return TypeError::DimMismatch {
                position: pos,
                expected: format!("{}", d1),
                found: format!("{}", d2),
            };
        }
    }

    TypeError::ShapeMismatch {
        expected: expected.clone(),
        found: found.clone(),
    }
}

/// Generate error message for a matrix multiplication shape error
pub fn matmul_shape_error(
    left_shape: &Shape,
    right_shape: &Shape,
) -> TypeError {
    if left_shape.rank() != 2 || right_shape.rank() != 2 {
        return TypeError::ShapeMismatch {
            expected: Shape::symbolic(&["m", "n"]),
            found: if left_shape.rank() != 2 { left_shape.clone() } else { right_shape.clone() },
        };
    }

    // Check inner dimensions
    let left_cols = &left_shape.0[1];
    let right_rows = &right_shape.0[0];

    TypeError::DimMismatch {
        position: 1,
        expected: format!("{} (columns of left matrix)", left_cols),
        found: format!("{} (rows of right matrix)", right_rows),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_add_zero() {
        let d = Dim::BinOp(
            Box::new(Dim::Var("n".into())),
            DimOp::Add,
            Box::new(Dim::Const(0)),
        );
        assert_eq!(simplify_dim(&d), Dim::Var("n".into()));
    }

    #[test]
    fn test_simplify_mul_one() {
        let d = Dim::BinOp(
            Box::new(Dim::Var("n".into())),
            DimOp::Mul,
            Box::new(Dim::Const(1)),
        );
        assert_eq!(simplify_dim(&d), Dim::Var("n".into()));
    }

    #[test]
    fn test_simplify_mul_zero() {
        let d = Dim::BinOp(
            Box::new(Dim::Var("n".into())),
            DimOp::Mul,
            Box::new(Dim::Const(0)),
        );
        assert_eq!(simplify_dim(&d), Dim::Const(0));
    }

    #[test]
    fn test_simplify_const_fold() {
        let d = Dim::BinOp(
            Box::new(Dim::Const(3)),
            DimOp::Add,
            Box::new(Dim::Const(4)),
        );
        assert_eq!(simplify_dim(&d), Dim::Const(7));
    }

    #[test]
    fn test_unify_same_const() {
        let s = unify_dims(&Dim::Const(5), &Dim::Const(5), 0).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn test_unify_diff_const() {
        let result = unify_dims(&Dim::Const(3), &Dim::Const(5), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_var_const() {
        let s = unify_dims(&Dim::Var("n".into()), &Dim::Const(10), 0).unwrap();
        assert_eq!(s.get("n"), Some(&Dim::Const(10)));
    }

    #[test]
    fn test_unify_const_var() {
        let s = unify_dims(&Dim::Const(10), &Dim::Var("n".into()), 0).unwrap();
        assert_eq!(s.get("n"), Some(&Dim::Const(10)));
    }

    #[test]
    fn test_unify_same_var() {
        let s = unify_dims(&Dim::Var("n".into()), &Dim::Var("n".into()), 0).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn test_unify_diff_var() {
        let s = unify_dims(&Dim::Var("n".into()), &Dim::Var("m".into()), 0).unwrap();
        // n should be bound to m (or vice versa)
        assert!(s.get("n").is_some() || s.get("m").is_some());
    }

    #[test]
    fn test_unify_shapes_same_rank() {
        let sh1 = Shape(vec![Dim::Var("n".into()), Dim::Const(3)]);
        let sh2 = Shape(vec![Dim::Const(10), Dim::Const(3)]);
        let s = unify_shapes(&sh1, &sh2).unwrap();
        assert_eq!(s.get("n"), Some(&Dim::Const(10)));
    }

    #[test]
    fn test_unify_shapes_rank_mismatch() {
        let sh1 = Shape(vec![Dim::Const(3)]);
        let sh2 = Shape(vec![Dim::Const(3), Dim::Const(4)]);
        let result = unify_shapes(&sh1, &sh2);
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_shapes_matmul() {
        // [m n] × [n p] → should unify n's
        let sh1 = Shape(vec![Dim::Var("m".into()), Dim::Var("n".into())]);
        let sh2 = Shape(vec![Dim::Var("n".into()), Dim::Var("p".into())]);

        // Extract inner dimensions
        let inner1 = &sh1.0[1]; // n from first
        let inner2 = &sh2.0[0]; // n from second

        let s = unify_dims(inner1, inner2, 0).unwrap();
        // Should unify (both are same var)
        assert!(s.is_empty()); // Same var, no substitution needed
    }

    #[test]
    fn test_unify_shapes_matmul_concrete() {
        // [2 3] × [3 4] → inner dims match
        let sh1 = Shape(vec![Dim::Const(2), Dim::Const(3)]);
        let sh2 = Shape(vec![Dim::Const(3), Dim::Const(4)]);

        let inner1 = &sh1.0[1];
        let inner2 = &sh2.0[0];

        let s = unify_dims(inner1, inner2, 0).unwrap();
        assert!(s.is_empty()); // Both are 3, no subst needed
    }

    #[test]
    fn test_unify_shapes_matmul_mismatch() {
        // [2 3] × [4 5] → inner dims don't match (3 ≠ 4)
        let sh1 = Shape(vec![Dim::Const(2), Dim::Const(3)]);
        let sh2 = Shape(vec![Dim::Const(4), Dim::Const(5)]);

        let inner1 = &sh1.0[1];
        let inner2 = &sh2.0[0];

        let result = unify_dims(inner1, inner2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_occurs_check() {
        // n cannot unify with n + 1 (infinite dimension)
        let d1 = Dim::Var("n".into());
        let d2 = Dim::BinOp(
            Box::new(Dim::Var("n".into())),
            DimOp::Add,
            Box::new(Dim::Const(1)),
        );
        let result = unify_dims(&d1, &d2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_checker_basic() {
        let mut checker = ShapeChecker::new();

        // Add constraint: n = 10
        checker.require_dim_eq(Dim::Var("n".into()), Dim::Const(10)).unwrap();

        // Resolve n
        assert_eq!(checker.resolve_dim(&Dim::Var("n".into())), Dim::Const(10));
    }

    #[test]
    fn test_shape_checker_chain() {
        let mut checker = ShapeChecker::new();

        // n = m, m = 5 → n = 5
        checker.require_dim_eq(Dim::Var("n".into()), Dim::Var("m".into())).unwrap();
        checker.require_dim_eq(Dim::Var("m".into()), Dim::Const(5)).unwrap();

        assert_eq!(checker.resolve_dim(&Dim::Var("n".into())), Dim::Const(5));
    }

    #[test]
    fn test_shape_checker_shapes() {
        let mut checker = ShapeChecker::new();

        let sh1 = Shape(vec![Dim::Var("n".into()), Dim::Const(3)]);
        let sh2 = Shape(vec![Dim::Const(10), Dim::Const(3)]);

        checker.require_shape_eq(&sh1, &sh2).unwrap();

        let resolved = checker.resolve_shape(&sh1);
        assert_eq!(resolved, Shape(vec![Dim::Const(10), Dim::Const(3)]));
    }

    #[test]
    fn test_shape_checker_mismatch() {
        let mut checker = ShapeChecker::new();

        let sh1 = Shape(vec![Dim::Const(3)]);
        let sh2 = Shape(vec![Dim::Const(5)]);

        let result = checker.require_shape_eq(&sh1, &sh2);
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_shape_vars() {
        let shape = Shape(vec![
            Dim::Var("m".into()),
            Dim::BinOp(
                Box::new(Dim::Var("n".into())),
                DimOp::Add,
                Box::new(Dim::Var("k".into())),
            ),
        ]);
        let vars = collect_shape_vars(&shape);
        assert!(vars.contains("m"));
        assert!(vars.contains("n"));
        assert!(vars.contains("k"));
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_solve_n_plus_k_eq_const() {
        // n + 3 = 10 → n = 7
        let d1 = Dim::BinOp(
            Box::new(Dim::Var("n".into())),
            DimOp::Add,
            Box::new(Dim::Const(3)),
        );
        let d2 = Dim::Const(10);

        let s = unify_dims(&d1, &d2, 0).unwrap();
        assert_eq!(s.get("n"), Some(&Dim::Const(7)));
    }

    #[test]
    fn test_identity_function_shape() {
        // id : [n]F64 → [n]F64
        // When called with [3]F64, should unify n = 3
        let mut checker = ShapeChecker::new();

        let param_shape = Shape(vec![Dim::Var("n".into())]);
        let arg_shape = Shape(vec![Dim::Const(3)]);

        checker.require_shape_eq(&param_shape, &arg_shape).unwrap();

        // Return shape should also resolve to [3]
        let return_shape = Shape(vec![Dim::Var("n".into())]);
        let resolved = checker.resolve_shape(&return_shape);
        assert_eq!(resolved, Shape(vec![Dim::Const(3)]));
    }
}
