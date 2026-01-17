//! Name Resolution for Goth
//!
//! Converts named variable references to de Bruijn indices.
//! 
//! After parsing, expressions contain `Name("x")` for variable references.
//! This pass resolves local bindings to `Idx(n)` where n is the de Bruijn index,
//! while keeping global references as `Name("x")`.
//!
//! ## Example
//!
//! ```text
//! Input:  let x = 5 in x * x
//! Parsed: Let { pattern: Var("x"), value: 5, body: Name("x") * Name("x") }
//! Resolved: Let { pattern: Var("x"), value: 5, body: Idx(0) * Idx(0) }
//! ```

use goth_ast::expr::{Expr, MatchArm, DoOp};
use goth_ast::pattern::Pattern;
use goth_ast::decl::{Module, Decl, FnDecl, LetDecl};

/// Resolve names to de Bruijn indices in an expression
pub fn resolve_expr(expr: Expr) -> Expr {
    let mut ctx = ResolveCtx::new();
    ctx.resolve_expr(expr)
}

/// Resolve names in a module
pub fn resolve_module(module: Module) -> Module {
    let mut ctx = ResolveCtx::new();
    ctx.resolve_module(module)
}

/// Resolution context tracking the current scope
struct ResolveCtx {
    /// Stack of bound names (innermost last)
    /// Each entry is a list of names bound at that level (for patterns with multiple bindings)
    scopes: Vec<Vec<String>>,
}

impl ResolveCtx {
    fn new() -> Self {
        ResolveCtx { scopes: Vec::new() }
    }

    /// Push a new scope with a single binding
    fn push(&mut self, name: String) {
        self.scopes.push(vec![name]);
    }

    /// Push a new scope with multiple bindings (from a pattern)
    fn push_many(&mut self, names: Vec<String>) {
        self.scopes.push(names);
    }

    /// Pop the innermost scope
    fn pop(&mut self) {
        self.scopes.pop();
    }

    /// Look up a name and return its de Bruijn index if bound locally
    fn lookup(&self, name: &str) -> Option<u32> {
        let mut idx = 0u32;
        // Iterate from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            // Check each name in this scope (in reverse for correct indexing)
            for bound_name in scope.iter().rev() {
                if bound_name == name {
                    return Some(idx);
                }
                idx += 1;
            }
        }
        None
    }

    /// Extract binding names from a pattern
    fn pattern_names(&self, pattern: &Pattern) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_pattern_names(pattern, &mut names);
        names
    }

    fn collect_pattern_names(&self, pattern: &Pattern, names: &mut Vec<String>) {
        match pattern {
            Pattern::Var(Some(name)) => names.push(name.to_string()),
            Pattern::Var(None) => names.push("_".to_string()),
            Pattern::Tuple(pats) => {
                for p in pats {
                    self.collect_pattern_names(p, names);
                }
            }
            Pattern::Array(pats) => {
                for p in pats {
                    self.collect_pattern_names(p, names);
                }
            }
            Pattern::ArraySplit { head, tail } => {
                for p in head {
                    self.collect_pattern_names(p, names);
                }
                self.collect_pattern_names(tail, names);
            }
            Pattern::Variant { payload: Some(p), .. } => {
                self.collect_pattern_names(p, names);
            }
            Pattern::Typed(inner, _) => {
                self.collect_pattern_names(inner, names);
            }
            Pattern::Or(p1, _p2) => {
                // Or patterns must bind the same names in both branches
                // For simplicity, just use the first branch
                self.collect_pattern_names(p1, names);
            }
            Pattern::Guard(inner, _) => {
                self.collect_pattern_names(inner, names);
            }
            // Wildcard, Lit don't bind names
            _ => {}
        }
    }

    /// Resolve an expression
    fn resolve_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            // Variable reference - look up in scope
            Expr::Name(name) => {
                if let Some(idx) = self.lookup(&name) {
                    Expr::Idx(idx)
                } else {
                    // Keep as global reference
                    Expr::Name(name)
                }
            }

            // Index - already resolved
            Expr::Idx(i) => Expr::Idx(i),

            // Literals - no resolution needed
            Expr::Lit(lit) => Expr::Lit(lit),
            Expr::Prim(name) => Expr::Prim(name),

            // Lambda - introduces one binding
            Expr::Lam(body) => {
                self.push("_".to_string());
                let body = self.resolve_expr(*body);
                self.pop();
                Expr::Lam(Box::new(body))
            }

            // Multi-arg lambda
            Expr::LamN(n, body) => {
                // Push n anonymous bindings
                let names: Vec<String> = (0..n).map(|i| format!("_{}", i)).collect();
                self.push_many(names);
                let body = self.resolve_expr(*body);
                self.pop();
                Expr::LamN(n, Box::new(body))
            }

            // Let - resolve value, then body with pattern bindings in scope
            Expr::Let { pattern, value, body } => {
                let value = self.resolve_expr(*value);
                let names = self.pattern_names(&pattern);
                self.push_many(names);
                let body = self.resolve_expr(*body);
                self.pop();
                Expr::Let {
                    pattern,
                    value: Box::new(value),
                    body: Box::new(body),
                }
            }

            // LetRec - all bindings are in scope for all values and body
            Expr::LetRec { bindings, body } => {
                // Collect all binding names first
                let names: Vec<String> = bindings.iter()
                    .flat_map(|(pat, _)| self.pattern_names(pat))
                    .collect();
                
                // Push all names into scope
                self.push_many(names);
                
                // Resolve all values and body
                let bindings = bindings.into_iter()
                    .map(|(pat, val)| (pat, self.resolve_expr(val)))
                    .collect();
                let body = self.resolve_expr(*body);
                
                self.pop();
                Expr::LetRec {
                    bindings,
                    body: Box::new(body),
                }
            }

            // Application
            Expr::App(func, arg) => {
                let func = self.resolve_expr(*func);
                let arg = self.resolve_expr(*arg);
                Expr::App(Box::new(func), Box::new(arg))
            }

            // Binary operation
            Expr::BinOp(op, left, right) => {
                let left = self.resolve_expr(*left);
                let right = self.resolve_expr(*right);
                Expr::BinOp(op, Box::new(left), Box::new(right))
            }

            // Unary operation
            Expr::UnaryOp(op, operand) => {
                let operand = self.resolve_expr(*operand);
                Expr::UnaryOp(op, Box::new(operand))
            }

            // Norm
            Expr::Norm(inner) => {
                let inner = self.resolve_expr(*inner);
                Expr::Norm(Box::new(inner))
            }

            // If
            Expr::If { cond, then_, else_ } => {
                let cond = self.resolve_expr(*cond);
                let then_ = self.resolve_expr(*then_);
                let else_ = self.resolve_expr(*else_);
                Expr::If {
                    cond: Box::new(cond),
                    then_: Box::new(then_),
                    else_: Box::new(else_),
                }
            }

            // Match - scrutinee in outer scope, arms have pattern bindings
            Expr::Match { scrutinee, arms } => {
                let scrutinee = self.resolve_expr(*scrutinee);
                let arms = arms.into_iter()
                    .map(|arm| self.resolve_match_arm(arm))
                    .collect();
                Expr::Match {
                    scrutinee: Box::new(scrutinee),
                    arms,
                }
            }

            // Tuple
            Expr::Tuple(exprs) => {
                let exprs = exprs.into_iter()
                    .map(|e| self.resolve_expr(e))
                    .collect();
                Expr::Tuple(exprs)
            }

            // Record
            Expr::Record(fields) => {
                let fields = fields.into_iter()
                    .map(|(name, e)| (name, self.resolve_expr(e)))
                    .collect();
                Expr::Record(fields)
            }

            // Array
            Expr::Array(exprs) => {
                let exprs = exprs.into_iter()
                    .map(|e| self.resolve_expr(e))
                    .collect();
                Expr::Array(exprs)
            }

            // ArrayFill
            Expr::ArrayFill { shape, value } => {
                let shape = shape.into_iter()
                    .map(|e| self.resolve_expr(e))
                    .collect();
                let value = self.resolve_expr(*value);
                Expr::ArrayFill {
                    shape,
                    value: Box::new(value),
                }
            }

            // Variant
            Expr::Variant { constructor, payload } => {
                let payload = payload.map(|e| Box::new(self.resolve_expr(*e)));
                Expr::Variant { constructor, payload }
            }

            // Field access
            Expr::Field(base, access) => {
                let base = self.resolve_expr(*base);
                Expr::Field(Box::new(base), access)
            }

            // Index
            Expr::Index(base, indices) => {
                let base = self.resolve_expr(*base);
                let indices = indices.into_iter()
                    .map(|e| self.resolve_expr(e))
                    .collect();
                Expr::Index(Box::new(base), indices)
            }

            // Slice
            Expr::Slice { array, start, end } => {
                let array = self.resolve_expr(*array);
                let start = start.map(|e| Box::new(self.resolve_expr(*e)));
                let end = end.map(|e| Box::new(self.resolve_expr(*e)));
                Expr::Slice {
                    array: Box::new(array),
                    start,
                    end,
                }
            }

            // Annotation
            Expr::Annot(inner, ty) => {
                let inner = self.resolve_expr(*inner);
                Expr::Annot(Box::new(inner), ty)
            }

            // Cast
            Expr::Cast { expr, target, kind } => {
                let expr = self.resolve_expr(*expr);
                Expr::Cast {
                    expr: Box::new(expr),
                    target,
                    kind,
                }
            }

            // Update
            Expr::Update { base, fields } => {
                let base = self.resolve_expr(*base);
                let fields = fields.into_iter()
                    .map(|(name, e)| (name, self.resolve_expr(e)))
                    .collect();
                Expr::Update {
                    base: Box::new(base),
                    fields,
                }
            }

            // Do notation
            Expr::Do { init, ops } => {
                let init = self.resolve_expr(*init);
                let ops = ops.into_iter()
                    .map(|op| self.resolve_do_op(op))
                    .collect();
                Expr::Do {
                    init: Box::new(init),
                    ops,
                }
            }

            // Disabled
            Expr::Disabled(inner) => {
                let inner = self.resolve_expr(*inner);
                Expr::Disabled(Box::new(inner))
            }

            // Hole - no resolution
            Expr::Hole => Expr::Hole,

            // Quote/Unquote - resolve inside
            Expr::Quote(inner) => {
                let inner = self.resolve_expr(*inner);
                Expr::Quote(Box::new(inner))
            }
            Expr::Unquote(inner) => {
                let inner = self.resolve_expr(*inner);
                Expr::Unquote(Box::new(inner))
            }
        }
    }

    /// Resolve a match arm
    fn resolve_match_arm(&mut self, arm: MatchArm) -> MatchArm {
        let names = self.pattern_names(&arm.pattern);
        self.push_many(names);
        
        let guard = arm.guard.map(|e| self.resolve_expr(e));
        let body = self.resolve_expr(arm.body);
        
        self.pop();
        
        MatchArm {
            pattern: arm.pattern,
            guard,
            body,
        }
    }

    /// Resolve a do operation
    fn resolve_do_op(&mut self, op: DoOp) -> DoOp {
        match op {
            DoOp::Map(e) => DoOp::Map(self.resolve_expr(e)),
            DoOp::Filter(e) => DoOp::Filter(self.resolve_expr(e)),
            DoOp::Bind(e) => DoOp::Bind(self.resolve_expr(e)),
            DoOp::Op(binop, e) => DoOp::Op(binop, self.resolve_expr(e)),
            DoOp::Let(pat, e) => {
                // Let in do binds for subsequent ops
                // For now, just resolve the expression
                let e = self.resolve_expr(e);
                let names = self.pattern_names(&pat);
                self.push_many(names);
                DoOp::Let(pat, e)
            }
        }
    }

    /// Resolve a module
    fn resolve_module(&mut self, module: Module) -> Module {
        let decls = module.decls.into_iter()
            .map(|d| self.resolve_decl(d))
            .collect();
        Module {
            name: module.name,
            decls,
        }
    }

    /// Resolve a declaration
    fn resolve_decl(&mut self, decl: Decl) -> Decl {
        match decl {
            Decl::Fn(fn_decl) => {
                // Function body has one implicit argument
                self.push("_".to_string());
                let body = self.resolve_expr(fn_decl.body);
                self.pop();
                
                let preconditions = fn_decl.preconditions.into_iter()
                    .map(|e| self.resolve_expr(e))
                    .collect();
                let postconditions = fn_decl.postconditions.into_iter()
                    .map(|e| self.resolve_expr(e))
                    .collect();
                
                Decl::Fn(FnDecl {
                    name: fn_decl.name,
                    type_params: fn_decl.type_params,
                    signature: fn_decl.signature,
                    constraints: fn_decl.constraints,
                    preconditions,
                    postconditions,
                    body,
                })
            }
            Decl::Let(let_decl) => {
                let value = self.resolve_expr(let_decl.value);
                Decl::Let(LetDecl {
                    name: let_decl.name,
                    type_: let_decl.type_,
                    value,
                })
            }
            // Type, Class, Impl, Op don't need expression resolution
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_expr;

    #[test]
    fn test_resolve_let() {
        let expr = parse_expr("let x = 5 in x").unwrap();
        let resolved = resolve_expr(expr);
        
        match resolved {
            Expr::Let { body, .. } => {
                assert!(matches!(*body, Expr::Idx(0)));
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_resolve_let_arithmetic() {
        let expr = parse_expr("let x = 5 in x * x").unwrap();
        let resolved = resolve_expr(expr);
        
        match resolved {
            Expr::Let { body, .. } => {
                match *body {
                    Expr::BinOp(_, left, right) => {
                        assert!(matches!(*left, Expr::Idx(0)));
                        assert!(matches!(*right, Expr::Idx(0)));
                    }
                    _ => panic!("Expected BinOp"),
                }
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_resolve_nested_let() {
        let expr = parse_expr("let x = 1 in let y = 2 in x + y").unwrap();
        let resolved = resolve_expr(expr);
        
        match resolved {
            Expr::Let { body: outer_body, .. } => {
                match *outer_body {
                    Expr::Let { body: inner_body, .. } => {
                        match *inner_body {
                            Expr::BinOp(_, left, right) => {
                                // x is at index 1 (outer), y is at index 0 (inner)
                                assert!(matches!(*left, Expr::Idx(1)));
                                assert!(matches!(*right, Expr::Idx(0)));
                            }
                            _ => panic!("Expected BinOp"),
                        }
                    }
                    _ => panic!("Expected inner Let"),
                }
            }
            _ => panic!("Expected outer Let"),
        }
    }

    #[test]
    fn test_resolve_lambda() {
        let expr = parse_expr("let f = λ→ x in f").unwrap();
        let resolved = resolve_expr(expr);
        
        // The 'x' inside the lambda should stay as Name (it's a free variable)
        // The 'f' at the end should become Idx(0)
        match resolved {
            Expr::Let { value, body, .. } => {
                match *value {
                    Expr::Lam(lam_body) => {
                        // x is not bound, stays as Name
                        assert!(matches!(*lam_body, Expr::Name(_)));
                    }
                    _ => panic!("Expected Lam"),
                }
                // f is bound by let
                assert!(matches!(*body, Expr::Idx(0)));
            }
            _ => panic!("Expected Let"),
        }
    }

    #[test]
    fn test_resolve_global_stays_name() {
        let expr = parse_expr("sqrt").unwrap();
        let resolved = resolve_expr(expr);
        
        // sqrt is not bound locally, should stay as Name
        assert!(matches!(resolved, Expr::Name(_)));
    }

    #[test]
    fn test_resolve_match() {
        let expr = parse_expr("match x { y → y }").unwrap();
        let resolved = resolve_expr(expr);
        
        match resolved {
            Expr::Match { arms, .. } => {
                assert_eq!(arms.len(), 1);
                // y in body should be Idx(0)
                assert!(matches!(arms[0].body, Expr::Idx(0)));
            }
            _ => panic!("Expected Match"),
        }
    }

    #[test]
    fn test_resolve_shadowing() {
        let expr = parse_expr("let x = 1 in let x = 2 in x").unwrap();
        let resolved = resolve_expr(expr);
        
        match resolved {
            Expr::Let { body: outer_body, .. } => {
                match *outer_body {
                    Expr::Let { body: inner_body, .. } => {
                        // Inner x shadows outer, so x refers to index 0
                        assert!(matches!(*inner_body, Expr::Idx(0)));
                    }
                    _ => panic!("Expected inner Let"),
                }
            }
            _ => panic!("Expected outer Let"),
        }
    }
}
