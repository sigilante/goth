//! Evaluator for Goth

use crate::value::{Value, Tensor, Closure, Env, PrimFn, StreamKind};
use crate::error::{EvalError, EvalResult, OptionExt};
use crate::prim;
use goth_ast::expr::{Expr, MatchArm, FieldAccess, CastKind, DoOp};
use goth_ast::literal::Literal;
use goth_ast::pattern::Pattern;
use goth_ast::op::BinOp;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

/// Represents either a final value or a tail call that needs to be evaluated.
/// Used for tail call optimization (TCO) to avoid stack overflow on deep recursion.
enum TcoResult {
    /// Evaluation is complete with this value
    Done(Value),
    /// A tail call that needs to be trampolined (closure body + environment)
    TailCall { body: Expr, env: Env },
}

pub struct Evaluator {
    globals: Rc<RefCell<HashMap<String, Value>>>,
    max_depth: usize,
    depth: usize,
    trace: bool,
}

impl Evaluator {
    pub fn new() -> Self {
        let globals = Rc::new(RefCell::new(HashMap::new()));
        let mut eval = Evaluator { globals, max_depth: 10000, depth: 0, trace: false };
        eval.register_primitives();
        eval
    }

    pub fn with_trace(mut self, trace: bool) -> Self { self.trace = trace; self }
    pub fn with_max_depth(mut self, depth: usize) -> Self { self.max_depth = depth; self }

    fn register_primitives(&mut self) {
        let prims: &[(&str, PrimFn)] = &[
            ("add", PrimFn::Add), ("sub", PrimFn::Sub), ("mul", PrimFn::Mul), ("div", PrimFn::Div), ("mod", PrimFn::Mod), ("neg", PrimFn::Neg), ("abs", PrimFn::Abs),
            ("exp", PrimFn::Exp), ("ln", PrimFn::Ln), ("sqrt", PrimFn::Sqrt), ("sin", PrimFn::Sin), ("cos", PrimFn::Cos), ("tan", PrimFn::Tan), ("pow", PrimFn::Pow), ("floor", PrimFn::Floor), ("ceil", PrimFn::Ceil), ("round", PrimFn::Round),
            ("eq", PrimFn::Eq), ("neq", PrimFn::Neq), ("lt", PrimFn::Lt), ("gt", PrimFn::Gt), ("leq", PrimFn::Leq), ("geq", PrimFn::Geq),
            ("and", PrimFn::And), ("or", PrimFn::Or), ("not", PrimFn::Not),
            ("sum", PrimFn::Sum), ("prod", PrimFn::Prod),
            ("len", PrimFn::Len),
            ("shape", PrimFn::Shape), ("ρ", PrimFn::Shape),  // APL rho
            ("reverse", PrimFn::Reverse), ("⌽", PrimFn::Reverse),  // APL reverse
            ("concat", PrimFn::Concat),
            ("iota", PrimFn::Iota), ("ι", PrimFn::Iota), ("⍳", PrimFn::Iota),  // APL-style
            ("range", PrimFn::Range), ("…", PrimFn::Range),
            ("dot", PrimFn::Dot), ("·", PrimFn::Dot),  // middle dot
            ("norm", PrimFn::Norm), ("matmul", PrimFn::MatMul),
            ("print", PrimFn::Print), ("println", PrimFn::Print),
            ("write", PrimFn::Write),  // Print without newline (for TUI)
            ("flush", PrimFn::Flush),  // Flush stdout
            ("readLine", PrimFn::ReadLine), ("read_line", PrimFn::ReadLine),
            ("readKey", PrimFn::ReadKey), ("read_key", PrimFn::ReadKey),  // Read single key
            ("readFile", PrimFn::ReadFile), ("writeFile", PrimFn::WriteFile),
            ("readBytes", PrimFn::ReadBytes), ("⧏", PrimFn::ReadBytes),
            ("writeBytes", PrimFn::WriteBytes), ("⧐", PrimFn::WriteBytes),
            ("rawModeEnter", PrimFn::RawModeEnter), ("rawModeExit", PrimFn::RawModeExit),  // Terminal raw mode
            ("sleep", PrimFn::Sleep),  // Sleep for milliseconds
            ("toInt", PrimFn::ToInt), ("toFloat", PrimFn::ToFloat), ("toBool", PrimFn::ToBool), ("toChar", PrimFn::ToChar),
            ("parseInt", PrimFn::ParseInt), ("parseFloat", PrimFn::ParseFloat),
            ("toString", PrimFn::ToString), ("str", PrimFn::ToString),
            ("chars", PrimFn::Chars), ("fromChars", PrimFn::FromChars),
            ("strConcat", PrimFn::StrConcat), ("⧺", PrimFn::StrConcat),  // double plus
            ("filter", PrimFn::Filter), ("map", PrimFn::Map), ("fold", PrimFn::Fold), ("⌿", PrimFn::Fold),
            // Bitwise operations
            ("bitand", PrimFn::BitAnd), ("bitor", PrimFn::BitOr),
            ("bitxor", PrimFn::BitXor), ("⊻", PrimFn::BitXor),
            ("shl", PrimFn::Shl), ("shr", PrimFn::Shr),
            ("index", PrimFn::Index),
            ("take", PrimFn::Take), ("↑", PrimFn::Take),  // APL take
            ("drop", PrimFn::Drop), ("↓", PrimFn::Drop),  // APL drop
            ("zip", PrimFn::Zip),
            ("transpose", PrimFn::Transpose), ("⍉", PrimFn::Transpose),  // APL transpose
            // String splitting (for wc-like operations)
            ("lines", PrimFn::Lines), ("words", PrimFn::Words), ("bytes", PrimFn::Bytes),
            // String comparison
            ("strEq", PrimFn::StrEq), ("startsWith", PrimFn::StartsWith),
            ("endsWith", PrimFn::EndsWith), ("contains", PrimFn::Contains),
            // Complex/quaternion decomposition
            ("re", PrimFn::Re), ("im", PrimFn::Im), ("conj", PrimFn::Conj), ("arg", PrimFn::Arg),
            // Matrix utilities
            ("trace", PrimFn::Trace), ("tr", PrimFn::Trace),
            ("det", PrimFn::Det), ("inv", PrimFn::Inv),
            ("diag", PrimFn::Diag), ("eye", PrimFn::Eye),
            ("solve", PrimFn::Solve), ("solveWith", PrimFn::SolveWith),
            ("eig", PrimFn::Eig), ("eigvecs", PrimFn::EigVecs),
        ];
        for (name, prim) in prims { self.globals.borrow_mut().insert(name.to_string(), Value::Primitive(*prim)); }
        // Register stream constants
        self.globals.borrow_mut().insert("stdout".to_string(), Value::Stream(StreamKind::Stdout));
        self.globals.borrow_mut().insert("stderr".to_string(), Value::Stream(StreamKind::Stderr));
    }

    pub fn define(&mut self, name: impl Into<String>, value: Value) { self.globals.borrow_mut().insert(name.into(), value); }
    
    pub fn globals(&self) -> Rc<RefCell<HashMap<String, Value>>> { Rc::clone(&self.globals) }

    pub fn eval(&mut self, expr: &Expr) -> EvalResult<Value> {
        let env = Env::with_globals(Rc::clone(&self.globals));
        self.eval_with_env(expr, &env)
    }

    pub fn eval_with_env(&mut self, expr: &Expr, env: &Env) -> EvalResult<Value> {
        self.depth += 1;
        if self.depth > self.max_depth { self.depth -= 1; return Err(EvalError::Internal("Recursion limit exceeded".into())); }
        if self.trace { eprintln!("{}eval: {}", "  ".repeat(self.depth), expr); }
        let result = self.eval_inner(expr, env);
        if self.trace { match &result { Ok(v) => eprintln!("{}=> {}", "  ".repeat(self.depth), v), Err(e) => eprintln!("{}=> ERROR: {}", "  ".repeat(self.depth), e), } }
        self.depth -= 1;
        result
    }

    fn eval_inner(&mut self, expr: &Expr, env: &Env) -> EvalResult<Value> {
        match expr {
            Expr::Idx(i) => env.get(*i).cloned().ok_or_unbound(*i),
            Expr::Name(name) => env.get_global(name).ok_or_undefined(name),
            Expr::Lit(lit) => Ok(self.eval_literal(lit)),
            Expr::Prim(name) => env.get_global(name).ok_or_else(|| EvalError::not_implemented(format!("primitive: {}", name))),
            Expr::App(func, arg) => { let func_val = self.eval_with_env(func, env)?; let arg_val = self.eval_with_env(arg, env)?; self.apply(func_val, arg_val) }
            Expr::Lam(body) => Ok(Value::Closure(Rc::new(Closure { arity: 1, body: (**body).clone(), env: env.capture(), preconditions: vec![], postconditions: vec![] }))),
            Expr::LamN(n, body) => Ok(Value::Closure(Rc::new(Closure { arity: *n, body: (**body).clone(), env: env.capture(), preconditions: vec![], postconditions: vec![] }))),
            Expr::Let { pattern, type_: _, value, body } => { let val = self.eval_with_env(value, env)?; let mut new_env = env.clone(); self.bind_pattern(pattern, val, &mut new_env)?; self.eval_with_env(body, &new_env) }
            Expr::LetRec { bindings, body } => {
                let mut new_env = env.clone();
                for _ in bindings { new_env.push(Value::Error("uninitialized letrec".into())); }
                let values: Vec<Value> = bindings.iter().map(|(_, expr)| self.eval_with_env(expr, &new_env)).collect::<Result<_, _>>()?;
                let depth = new_env.depth();
                for (i, val) in values.into_iter().enumerate() { let idx = depth - bindings.len() + i; if let Some(slot) = new_env.values.get_mut(idx) { *slot = val; } }
                self.eval_with_env(body, &new_env)
            }
            Expr::Match { scrutinee, arms } => { let val = self.eval_with_env(scrutinee, env)?; self.eval_match(val, arms, env) }
            Expr::If { cond, then_, else_ } => { let cond_val = self.eval_with_env(cond, env)?; match cond_val { Value::Bool(true) => self.eval_with_env(then_, env), Value::Bool(false) => self.eval_with_env(else_, env), _ => Err(EvalError::type_error("Bool", &cond_val)) } }
            Expr::BinOp(op, left, right) => self.eval_binop(op, left, right, env),
            Expr::UnaryOp(op, operand) => { let val = self.eval_with_env(operand, env)?; prim::apply_unaryop(op, val) }
            Expr::Norm(inner) => { let val = self.eval_with_env(inner, env)?; prim::apply_prim(PrimFn::Norm, vec![val]) }
            Expr::Tuple(exprs) => { let values: Vec<Value> = exprs.iter().map(|e| self.eval_with_env(e, env)).collect::<Result<_, _>>()?; Ok(Value::tuple(values)) }
            Expr::Record(fields) => { let map: HashMap<String, Value> = fields.iter().map(|(name, expr)| { let val = self.eval_with_env(expr, env)?; Ok((name.to_string(), val)) }).collect::<EvalResult<_>>()?; Ok(Value::Record(Rc::new(map))) }
            Expr::Array(exprs) => { let values: Vec<Value> = exprs.iter().map(|e| self.eval_with_env(e, env)).collect::<Result<_, _>>()?; Ok(self.values_to_tensor(values)) }
            Expr::ArrayFill { shape, value } => { let shape_vals: Vec<usize> = shape.iter().map(|e| { let v = self.eval_with_env(e, env)?; v.as_int().map(|n| n as usize).ok_or_else(|| EvalError::type_error("Int", &v)) }).collect::<Result<_, _>>()?; let fill_val = self.eval_with_env(value, env)?; let size: usize = shape_vals.iter().product(); let data = vec![fill_val; size]; Ok(Value::Tensor(Rc::new(Tensor::from_values(shape_vals, data)))) }
            Expr::Variant { constructor, payload } => { let payload_val = match payload { Some(p) => Some(self.eval_with_env(p, env)?), None => None }; Ok(Value::variant(constructor.to_string(), payload_val)) }
            Expr::Field(base, access) => { let val = self.eval_with_env(base, env)?; self.access_field(val, access) }
            Expr::Index(base, indices) => { let arr = self.eval_with_env(base, env)?; let idx_vals: Vec<usize> = indices.iter().map(|e| { let v = self.eval_with_env(e, env)?; v.as_index().ok_or_else(|| EvalError::type_error("Int", &v)) }).collect::<Result<_, _>>()?; self.index_value(arr, &idx_vals) }
            Expr::Slice { array, start, end } => { let arr = self.eval_with_env(array, env)?; let start_idx = match start { Some(e) => { let v = self.eval_with_env(e, env)?; v.as_int().map(|n| n as usize).unwrap_or(0) } None => 0 }; let end_idx = match end { Some(e) => { let v = self.eval_with_env(e, env)?; v.as_int().map(|n| n as usize) } None => None }; self.slice_value(arr, start_idx, end_idx) }
            Expr::Annot(inner, _ty) => self.eval_with_env(inner, env),
            Expr::Cast { expr, target: _, kind } => { let val = self.eval_with_env(expr, env)?; match kind { CastKind::Static => Ok(val), CastKind::Try => Ok(Value::variant("Some", Some(val))), CastKind::Force => Ok(val) } }
            Expr::Update { base, fields } => { let base_val = self.eval_with_env(base, env)?; match base_val { Value::Record(map) => { let mut new_map = (*map).clone(); for (name, expr) in fields { let val = self.eval_with_env(expr, env)?; new_map.insert(name.to_string(), val); } Ok(Value::Record(Rc::new(new_map))) } _ => Err(EvalError::type_error("Record", &base_val)) } }
            Expr::Do { init, ops } => self.eval_do(init, ops, env),
            Expr::Disabled(_) => Ok(Value::Unit),
            Expr::Hole => Err(EvalError::not_implemented("hole evaluation")),
            Expr::Quote(_) => Err(EvalError::not_implemented("quote evaluation")),
            Expr::Unquote(_) => Err(EvalError::Internal("unquote outside of quote".into())),
        }
    }

    fn eval_literal(&self, lit: &Literal) -> Value {
        match lit { Literal::Int(n) => Value::Int(*n), Literal::Float(f) => Value::float(*f), Literal::Char(c) => Value::Char(*c), Literal::String(s) => Value::string(s), Literal::True => Value::Bool(true), Literal::False => Value::Bool(false), Literal::Unit => Value::Unit, Literal::ImagI(f) => Value::Complex(0.0, *f), Literal::ImagJ(f) => Value::Quaternion(0.0, 0.0, *f, 0.0), Literal::ImagK(f) => Value::Quaternion(0.0, 0.0, 0.0, *f) }
    }

    fn eval_binop(&mut self, op: &BinOp, left: &Expr, right: &Expr, env: &Env) -> EvalResult<Value> {
        match op {
            BinOp::Map => { let arr = self.eval_with_env(left, env)?; let func = self.eval_with_env(right, env)?; self.eval_map(arr, func) }
            BinOp::Filter => { let arr = self.eval_with_env(left, env)?; let pred = self.eval_with_env(right, env)?; self.eval_filter(arr, pred) }
            BinOp::Bind => { let arr = self.eval_with_env(left, env)?; let func = self.eval_with_env(right, env)?; self.eval_bind(arr, func) }
            BinOp::Compose => { let f = self.eval_with_env(left, env)?; let g = self.eval_with_env(right, env)?; self.eval_compose(f, g) }
            BinOp::And => { let left_val = self.eval_with_env(left, env)?; match left_val { Value::Bool(false) => Ok(Value::Bool(false)), Value::Bool(true) => self.eval_with_env(right, env), _ => Err(EvalError::type_error("Bool", &left_val)) } }
            BinOp::Or => { let left_val = self.eval_with_env(left, env)?; match left_val { Value::Bool(true) => Ok(Value::Bool(true)), Value::Bool(false) => self.eval_with_env(right, env), _ => Err(EvalError::type_error("Bool", &left_val)) } }
            BinOp::Write => { let content = self.eval_with_env(left, env)?; let path = self.eval_with_env(right, env)?; self.eval_write(content, path) }
            BinOp::Read => { let path = self.eval_with_env(left, env)?; self.eval_read(path) }
            _ => { let left_val = self.eval_with_env(left, env)?; let right_val = self.eval_with_env(right, env)?; prim::apply_binop(op, left_val, right_val) }
        }
    }

    /// Main apply function with trampoline for tail call optimization.
    /// This prevents stack overflow on deeply recursive tail calls.
    fn apply(&mut self, func: Value, arg: Value) -> EvalResult<Value> {
        // Use trampoline: loop instead of recurse for tail calls
        let mut tco_result = self.apply_once(func, arg)?;

        loop {
            match tco_result {
                TcoResult::Done(value) => return Ok(value),
                TcoResult::TailCall { body, env } => {
                    // Check depth limit (logical recursion depth)
                    self.depth += 1;
                    if self.depth > self.max_depth {
                        self.depth -= 1;
                        return Err(EvalError::Internal("Recursion limit exceeded".into()));
                    }

                    // Evaluate in tail position - may return another TailCall
                    tco_result = self.eval_tail(&body, &env)?;
                    self.depth -= 1;
                }
            }
        }
    }

    /// Single step of application that may return a tail call for trampolining.
    fn apply_once(&mut self, func: Value, arg: Value) -> EvalResult<TcoResult> {
        match func {
            Value::Closure(rc_closure) => {
                if rc_closure.arity == 1 {
                    let closure = Rc::unwrap_or_clone(rc_closure);
                    let mut new_env = closure.env.clone();
                    new_env.push(arg.clone());

                    // Check preconditions (argument bound as ₀)
                    self.check_preconditions(&closure.preconditions, &new_env)?;

                    // Return tail call for trampoline (postconditions checked after body eval)
                    if closure.postconditions.is_empty() {
                        Ok(TcoResult::TailCall { body: closure.body, env: new_env })
                    } else {
                        // Has postconditions - evaluate now and check them
                        let result = self.eval_with_env(&closure.body, &new_env)?;
                        self.check_postconditions(&closure.postconditions, &new_env, &result)?;
                        Ok(TcoResult::Done(result))
                    }
                } else {
                    let remaining = (rc_closure.arity - 1) as usize;
                    Ok(TcoResult::Done(Value::Partial { func: Box::new(Value::Closure(rc_closure)), args: vec![arg], remaining }))
                }
            }
            Value::Partial { func, mut args, remaining } => {
                args.push(arg);
                if remaining == 1 {
                    match *func {
                        Value::Closure(rc_closure) => {
                            let closure = Rc::unwrap_or_clone(rc_closure);
                            let mut new_env = closure.env.clone();
                            for a in &args {
                                new_env.push(a.clone());
                            }

                            // Check preconditions
                            self.check_preconditions(&closure.preconditions, &new_env)?;

                            // Return tail call for trampoline (postconditions checked after)
                            if closure.postconditions.is_empty() {
                                Ok(TcoResult::TailCall { body: closure.body, env: new_env })
                            } else {
                                let result = self.eval_with_env(&closure.body, &new_env)?;
                                self.check_postconditions(&closure.postconditions, &new_env, &result)?;
                                Ok(TcoResult::Done(result))
                            }
                        }
                        Value::Primitive(prim) => {
                            if prim == PrimFn::Fold && args.len() == 3 {
                                let func = args.remove(0);
                                let acc = args.remove(0);
                                let arr = args.remove(0);
                                Ok(TcoResult::Done(self.eval_fold(func, acc, arr)?))
                            } else {
                                Ok(TcoResult::Done(prim::apply_prim(prim, args)?))
                            }
                        }
                        _ => Err(EvalError::type_error("function", &func)),
                    }
                } else {
                    Ok(TcoResult::Done(Value::Partial { func, args, remaining: remaining - 1 }))
                }
            }
            Value::Primitive(prim) => {
                let arity = prim_arity(prim);
                if arity == 1 {
                    Ok(TcoResult::Done(prim::apply_prim(prim, vec![arg])?))
                } else {
                    Ok(TcoResult::Done(Value::Partial { func: Box::new(Value::Primitive(prim)), args: vec![arg], remaining: arity - 1 }))
                }
            }
            _ => Err(EvalError::type_error("function", &func)),
        }
    }

    /// Check preconditions for a closure
    fn check_preconditions(&mut self, preconditions: &[Expr], env: &Env) -> EvalResult<()> {
        for (i, pre) in preconditions.iter().enumerate() {
            let pre_result = self.eval_with_env(pre, env)?;
            match pre_result {
                Value::Bool(true) => {},
                Value::Bool(false) => {
                    return Err(EvalError::PreconditionViolated(format!("precondition #{} failed", i + 1)));
                }
                _ => {
                    return Err(EvalError::type_error("Bool", &pre_result));
                }
            }
        }
        Ok(())
    }

    /// Check postconditions for a closure
    fn check_postconditions(&mut self, postconditions: &[Expr], env: &Env, result: &Value) -> EvalResult<()> {
        for (i, post) in postconditions.iter().enumerate() {
            let mut post_env = env.clone();
            post_env.push(result.clone());
            let post_result = self.eval_with_env(post, &post_env)?;
            match post_result {
                Value::Bool(true) => {},
                Value::Bool(false) => {
                    return Err(EvalError::PostconditionViolated(format!("postcondition #{} failed", i + 1)));
                }
                _ => {
                    return Err(EvalError::type_error("Bool", &post_result));
                }
            }
        }
        Ok(())
    }

    /// Evaluate an expression in tail position, returning TcoResult.
    /// This enables proper tail call optimization for recursive functions.
    fn eval_tail(&mut self, expr: &Expr, env: &Env) -> EvalResult<TcoResult> {
        match expr {
            // If: both branches are in tail position
            Expr::If { cond, then_, else_ } => {
                let cond_val = self.eval_with_env(cond, env)?;
                match cond_val {
                    Value::Bool(true) => self.eval_tail(then_, env),
                    Value::Bool(false) => self.eval_tail(else_, env),
                    _ => Err(EvalError::type_error("Bool", &cond_val))
                }
            }
            // Let: body is in tail position
            Expr::Let { pattern, value, body, type_: _ } => {
                let val = self.eval_with_env(value, env)?;
                let mut new_env = env.clone();
                self.bind_pattern(pattern, val, &mut new_env)?;
                self.eval_tail(body, &new_env)
            }
            // LetRec: body is in tail position
            Expr::LetRec { bindings, body } => {
                let mut new_env = env.clone();
                for _ in bindings { new_env.push(Value::Error("uninitialized letrec".into())); }
                let values: Vec<Value> = bindings.iter()
                    .map(|(_, expr)| self.eval_with_env(expr, &new_env))
                    .collect::<Result<_, _>>()?;
                let depth = new_env.depth();
                for (i, val) in values.into_iter().enumerate() {
                    let idx = depth - bindings.len() + i;
                    if let Some(slot) = new_env.values.get_mut(idx) { *slot = val; }
                }
                self.eval_tail(body, &new_env)
            }
            // Match: each arm body is in tail position
            Expr::Match { scrutinee, arms } => {
                let val = self.eval_with_env(scrutinee, env)?;
                self.eval_match_tail(val, arms, env)
            }
            // Application: this IS the tail call - return for trampolining
            Expr::App(func, arg) => {
                let func_val = self.eval_with_env(func, env)?;
                let arg_val = self.eval_with_env(arg, env)?;
                self.apply_once(func_val, arg_val)
            }
            // Everything else: not a tail call, evaluate normally and wrap
            _ => Ok(TcoResult::Done(self.eval_with_env(expr, env)?))
        }
    }

    /// Pattern match with tail call optimization for arm bodies
    fn eval_match_tail(&mut self, val: Value, arms: &[MatchArm], env: &Env) -> EvalResult<TcoResult> {
        for arm in arms {
            let mut new_env = env.clone();
            if self.match_pattern(&arm.pattern, &val, &mut new_env)? {
                if let Some(guard) = &arm.guard {
                    let guard_val = self.eval_with_env(guard, &new_env)?;
                    match guard_val {
                        Value::Bool(true) => {}
                        Value::Bool(false) => continue,
                        _ => return Err(EvalError::type_error("Bool", &guard_val))
                    }
                }
                return self.eval_tail(&arm.body, &new_env);
            }
        }
        Err(EvalError::NonExhaustiveMatch)
    }

    fn eval_match(&mut self, val: Value, arms: &[MatchArm], env: &Env) -> EvalResult<Value> {
        for arm in arms {
            let mut new_env = env.clone();
            if self.match_pattern(&arm.pattern, &val, &mut new_env)? {
                if let Some(guard) = &arm.guard { let guard_val = self.eval_with_env(guard, &new_env)?; match guard_val { Value::Bool(true) => {} Value::Bool(false) => continue, _ => return Err(EvalError::type_error("Bool", &guard_val)) } }
                return self.eval_with_env(&arm.body, &new_env);
            }
        }
        Err(EvalError::NonExhaustiveMatch)
    }

    fn match_pattern(&self, pattern: &Pattern, val: &Value, env: &mut Env) -> EvalResult<bool> {
        match pattern {
            // Wildcards must push to env to maintain De Bruijn index alignment with resolver
            Pattern::Wildcard => { env.push(val.clone()); Ok(true) }
            Pattern::Var(_) => { env.push(val.clone()); Ok(true) }
            Pattern::Lit(lit) => { let lit_val = self.eval_literal(lit); Ok(val.deep_eq(&lit_val)) }
            Pattern::Array(pats) => { match val { Value::Tensor(t) => { if t.rank() != 1 || t.len() != pats.len() { return Ok(false); } for (i, pat) in pats.iter().enumerate() { let elem = t.get_flat(i).unwrap(); if !self.match_pattern(pat, &elem, env)? { return Ok(false); } } Ok(true) } _ => Ok(false) } }
            Pattern::ArraySplit { head, tail } => { match val { Value::Tensor(t) => { if t.rank() != 1 || t.len() < head.len() { return Ok(false); } for (i, pat) in head.iter().enumerate() { let elem = t.get_flat(i).unwrap(); if !self.match_pattern(pat, &elem, env)? { return Ok(false); } } let tail_data: Vec<Value> = (head.len()..t.len()).map(|i| t.get_flat(i).unwrap()).collect(); let tail_tensor = Rc::new(Tensor::from_values(vec![tail_data.len()], tail_data)); self.match_pattern(tail, &Value::Tensor(tail_tensor), env) } _ => Ok(false) } }
            Pattern::Tuple(pats) => { match val { Value::Tuple(vals) => { if vals.len() != pats.len() { return Ok(false); } for (pat, v) in pats.iter().zip(vals) { if !self.match_pattern(pat, v, env)? { return Ok(false); } } Ok(true) } Value::Unit if pats.is_empty() => Ok(true), _ => Ok(false) } }
            Pattern::Variant { constructor, payload } => { match val { Value::Variant { tag, payload: val_payload } => { if tag.as_str() != constructor.as_ref() { return Ok(false); } match (payload, val_payload) { (None, None) => Ok(true), (Some(pat), Some(v)) => self.match_pattern(pat, v, env), _ => Ok(false) } } _ => Ok(false) } }
            Pattern::Typed(inner, _ty) => self.match_pattern(inner, val, env),
            Pattern::Or(p1, p2) => { let mut env1 = env.clone(); if self.match_pattern(p1, val, &mut env1)? { *env = env1; return Ok(true); } self.match_pattern(p2, val, env) }
            Pattern::Guard(inner, _) => self.match_pattern(inner, val, env),
        }
    }

    fn bind_pattern(&self, pattern: &Pattern, val: Value, env: &mut Env) -> EvalResult<()> { if self.match_pattern(pattern, &val, env)? { Ok(()) } else { Err(EvalError::MatchFailed) } }

    fn values_to_tensor(&self, values: Vec<Value>) -> Value {
        self.values_to_tensor_shaped(vec![values.len()], values)
    }

    fn values_to_tensor_shaped(&self, shape: Vec<usize>, values: Vec<Value>) -> Value {
        if values.is_empty() { return Value::Tensor(Rc::new(Tensor::from_ints(vec![]))); }
        let all_int = values.iter().all(|v| matches!(v, Value::Int(_)));
        let all_float = values.iter().all(|v| matches!(v, Value::Float(_) | Value::Int(_)));
        let all_bool = values.iter().all(|v| matches!(v, Value::Bool(_)));
        let all_char = values.iter().all(|v| matches!(v, Value::Char(_)));
        if all_int { Value::Tensor(Rc::new(Tensor { shape, data: crate::value::TensorData::Int(values.iter().map(|v| v.as_int().unwrap()).collect()) })) }
        else if all_float { Value::Tensor(Rc::new(Tensor { shape, data: crate::value::TensorData::Float(values.iter().map(|v| ordered_float::OrderedFloat(v.coerce_float().unwrap())).collect()) })) }
        else if all_bool { Value::Tensor(Rc::new(Tensor { shape, data: crate::value::TensorData::Bool(values.iter().map(|v| v.as_bool().unwrap()).collect()) })) }
        else if all_char { Value::Tensor(Rc::new(Tensor { shape, data: crate::value::TensorData::Char(values.iter().map(|v| v.as_char().unwrap()).collect()) })) }
        else {
            // Auto-flatten: if all values are tensors with the same shape and compatible data,
            // combine into a higher-dimensional tensor (e.g., [[1,2],[3,4]] → shape [2,2])
            let all_tensor_same_shape = if let Some(Value::Tensor(first)) = values.first() {
                let s = &first.shape;
                values.iter().all(|v| matches!(v, Value::Tensor(t) if &t.shape == s))
            } else { false };
            if all_tensor_same_shape {
                let sub_tensors: Vec<&Tensor> = values.iter().map(|v| match v { Value::Tensor(t) => t.as_ref(), _ => unreachable!() }).collect();
                let sub_shape = &sub_tensors[0].shape;
                let mut new_shape = shape.clone();
                new_shape.extend_from_slice(sub_shape);
                // Try to flatten into typed tensor
                let all_int = sub_tensors.iter().all(|t| matches!(t.data, crate::value::TensorData::Int(_)));
                let all_float = sub_tensors.iter().all(|t| matches!(t.data, crate::value::TensorData::Int(_) | crate::value::TensorData::Float(_)));
                let all_bool = sub_tensors.iter().all(|t| matches!(t.data, crate::value::TensorData::Bool(_)));
                if all_int {
                    let data: Vec<i128> = sub_tensors.iter().flat_map(|t| match &t.data { crate::value::TensorData::Int(v) => v.iter().copied(), _ => unreachable!() }).collect();
                    Value::Tensor(Rc::new(Tensor { shape: new_shape, data: crate::value::TensorData::Int(data) }))
                } else if all_float {
                    let data: Vec<ordered_float::OrderedFloat<f64>> = sub_tensors.iter().flat_map(|t| match &t.data {
                        crate::value::TensorData::Float(v) => v.clone(),
                        crate::value::TensorData::Int(v) => v.iter().map(|&i| ordered_float::OrderedFloat(i as f64)).collect(),
                        _ => unreachable!()
                    }).collect();
                    Value::Tensor(Rc::new(Tensor { shape: new_shape, data: crate::value::TensorData::Float(data) }))
                } else if all_bool {
                    let data: Vec<bool> = sub_tensors.iter().flat_map(|t| match &t.data { crate::value::TensorData::Bool(v) => v.iter().copied(), _ => unreachable!() }).collect();
                    Value::Tensor(Rc::new(Tensor { shape: new_shape, data: crate::value::TensorData::Bool(data) }))
                } else {
                    Value::Tensor(Rc::new(Tensor::from_values(shape, values)))
                }
            } else {
                Value::Tensor(Rc::new(Tensor::from_values(shape, values)))
            }
        }
    }

    fn access_field(&self, val: Value, access: &FieldAccess) -> EvalResult<Value> {
        match access {
            FieldAccess::Index(i) => { match val { Value::Tuple(vs) => vs.get(*i as usize).cloned().ok_or_else(|| EvalError::IndexOutOfBounds { index: *i as usize, size: vs.len() }), Value::Tensor(t) => t.get_flat(*i as usize).ok_or_else(|| EvalError::IndexOutOfBounds { index: *i as usize, size: t.len() }), _ => Err(EvalError::type_error("Tuple or Tensor", &val)) } }
            FieldAccess::Named(name) => { match val { Value::Record(map) => map.get(name.as_ref()).cloned().ok_or_else(|| EvalError::UndefinedName(name.to_string())), _ => Err(EvalError::type_error("Record", &val)) } }
        }
    }

    fn index_value(&self, val: Value, indices: &[usize]) -> EvalResult<Value> {
        match val {
            Value::Tensor(t) => t.get(indices).ok_or_else(|| EvalError::IndexOutOfBounds { index: indices[0], size: t.shape.get(0).copied().unwrap_or(0) }),
            Value::Tuple(vs) => { if indices.len() != 1 { return Err(EvalError::type_error_msg("Tuple indexing requires single index")); } vs.get(indices[0]).cloned().ok_or_else(|| EvalError::IndexOutOfBounds { index: indices[0], size: vs.len() }) }
            _ => Err(EvalError::type_error("Tensor or Tuple", &val)),
        }
    }

    fn slice_value(&self, val: Value, start: usize, end: Option<usize>) -> EvalResult<Value> {
        match val {
            Value::Tensor(t) => { if t.rank() != 1 { return Err(EvalError::not_implemented("slicing rank > 1")); } let end = end.unwrap_or(t.len()); if start > end || end > t.len() { return Err(EvalError::IndexOutOfBounds { index: end, size: t.len() }); } let data: Vec<Value> = (start..end).map(|i| t.get_flat(i).unwrap()).collect(); Ok(self.values_to_tensor_shaped(vec![data.len()], data)) }
            _ => Err(EvalError::type_error("Tensor", &val)),
        }
    }

    fn eval_map(&mut self, arr: Value, func: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => { let shape = t.shape.clone(); let results: Vec<Value> = t.iter().map(|elem| self.apply(func.clone(), elem)).collect::<Result<_, _>>()?; Ok(self.values_to_tensor_shaped(shape, results)) }
            Value::Tuple(vs) => { let results: Vec<Value> = vs.into_iter().map(|elem| self.apply(func.clone(), elem)).collect::<Result<_, _>>()?; Ok(Value::Tuple(results)) }
            _ => Err(EvalError::type_error("Tensor or Tuple", &arr)),
        }
    }

    fn eval_filter(&mut self, arr: Value, pred: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => {
                let mut results = Vec::new();
                for elem in t.iter() {
                    let keep = self.apply(pred.clone(), elem.clone())?;
                    if keep == Value::Bool(true) {
                        results.push(elem);
                    }
                }
                Ok(self.values_to_tensor_shaped(vec![results.len()], results))
            }
            Value::Tuple(vs) => {
                let mut results = Vec::new();
                for elem in vs {
                    let keep = self.apply(pred.clone(), elem.clone())?;
                    if keep == Value::Bool(true) {
                        results.push(elem);
                    }
                }
                Ok(Value::Tuple(results))
            }
            _ => Err(EvalError::type_error("Tensor or Tuple", &arr)),
        }
    }

    fn eval_fold(&mut self, func: Value, init: Value, arr: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => {
                let mut acc = init;
                for elem in t.iter() {
                    let partial = self.apply(func.clone(), acc)?;
                    acc = self.apply(partial, elem)?;
                }
                Ok(acc)
            }
            Value::Tuple(vs) => {
                let mut acc = init;
                for elem in vs {
                    let partial = self.apply(func.clone(), acc)?;
                    acc = self.apply(partial, elem)?;
                }
                Ok(acc)
            }
            _ => Err(EvalError::type_error("Tensor or Tuple", &arr)),
        }
    }

    fn eval_bind(&mut self, arr: Value, func: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => { let mut results = Vec::new(); for elem in t.iter() { let mapped = self.apply(func.clone(), elem)?; match mapped { Value::Tensor(inner) => results.extend(inner.iter()), Value::Tuple(inner) => results.extend(inner), other => results.push(other) } } Ok(self.values_to_tensor_shaped(vec![results.len()], results)) }
            _ => Err(EvalError::type_error("Tensor", &arr)),
        }
    }

    fn eval_compose(&mut self, f: Value, g: Value) -> EvalResult<Value> {
        if !f.is_callable() { return Err(EvalError::type_error("function", &f)); }
        if !g.is_callable() { return Err(EvalError::type_error("function", &g)); }
        // (f ∘ g)(x) = f(g(x))
        // Idx(0) = x, Idx(1) = g, Idx(2) = f
        let body = Expr::App(Box::new(Expr::Idx(2)), Box::new(Expr::App(Box::new(Expr::Idx(1)), Box::new(Expr::Idx(0)))));
        let mut env = Env::with_globals(Rc::clone(&self.globals));
        env.push(f); env.push(g);  // Push f first, then g, so g is at Idx(1) and f is at Idx(2)
        Ok(Value::Closure(Rc::new(Closure { arity: 1, body, env, preconditions: vec![], postconditions: vec![] })))
    }

    fn eval_write(&mut self, content: Value, target: Value) -> EvalResult<Value> {
        // If target is a stream, write to stdout/stderr
        if let Value::Stream(kind) = &target {
            let content_str = match &content {
                Value::Tensor(t) => tensor_to_string(t).ok_or_else(|| EvalError::type_error("String", &content))?,
                other => format!("{}", other),
            };
            match kind {
                StreamKind::Stdout => {
                    use std::io::Write;
                    print!("{}", content_str);
                    std::io::stdout().flush().map_err(|e| EvalError::io_error(format!("flush stdout: {}", e)))?;
                }
                StreamKind::Stderr => {
                    use std::io::Write;
                    eprint!("{}", content_str);
                    std::io::stderr().flush().map_err(|e| EvalError::io_error(format!("flush stderr: {}", e)))?;
                }
            }
            return Ok(Value::Unit);
        }
        // Extract path as string
        let path_str = match &target {
            Value::Tensor(t) => tensor_to_string(t).ok_or_else(|| EvalError::type_error("String or Stream", &target))?,
            _ => return Err(EvalError::type_error("String or Stream", &target)),
        };
        // Extract content as string
        let content_str = match &content {
            Value::Tensor(t) => tensor_to_string(t).ok_or_else(|| EvalError::type_error("String", &content))?,
            _ => return Err(EvalError::type_error("String", &content)),
        };
        // Write to file
        std::fs::write(&path_str, &content_str)
            .map_err(|e| EvalError::io_error(format!("Failed to write '{}': {}", path_str, e)))?;
        Ok(Value::Unit)
    }

    fn eval_read(&mut self, path: Value) -> EvalResult<Value> {
        // Extract path as string
        let path_str = match &path {
            Value::Tensor(t) => tensor_to_string(t).ok_or_else(|| EvalError::type_error("String", &path))?,
            _ => return Err(EvalError::type_error("String", &path)),
        };
        // Read from file
        let contents = std::fs::read_to_string(&path_str)
            .map_err(|e| EvalError::io_error(format!("Failed to read '{}': {}", path_str, e)))?;
        Ok(Value::string(&contents))
    }

    fn eval_do(&mut self, init: &Expr, ops: &[DoOp], env: &Env) -> EvalResult<Value> {
        let mut current = self.eval_with_env(init, env)?;
        for op in ops {
            current = match op {
                DoOp::Map(f) => { let func = self.eval_with_env(f, env)?; self.eval_map(current, func)? }
                DoOp::Filter(p) => { let pred = self.eval_with_env(p, env)?; self.eval_filter(current, pred)? }
                DoOp::Bind(f) => { let func = self.eval_with_env(f, env)?; self.eval_bind(current, func)? }
                DoOp::Op(binop, e) => { let right = self.eval_with_env(e, env)?; prim::apply_binop(binop, current, right)? }
                DoOp::Let(_, e) => { self.eval_with_env(e, env)? }
            };
        }
        Ok(current)
    }
}

impl Default for Evaluator { fn default() -> Self { Self::new() } }

/// Helper to convert a tensor to a string.
/// Handles both native Char tensors and Generic tensors containing Char values.
fn tensor_to_string(t: &Tensor) -> Option<String> {
    // First try native char tensor
    if let Some(s) = t.to_string_value() {
        return Some(s);
    }
    // Then try generic tensor of chars
    let chars: Option<Vec<char>> = t.iter().map(|v| match v {
        Value::Char(c) => Some(c),
        _ => None,
    }).collect();
    chars.map(|cs| cs.into_iter().collect())
}

fn prim_arity(prim: PrimFn) -> usize {
    match prim {
        PrimFn::Neg | PrimFn::Abs | PrimFn::Not | PrimFn::Exp | PrimFn::Ln | PrimFn::Sqrt | PrimFn::Sin | PrimFn::Cos | PrimFn::Tan | PrimFn::Floor | PrimFn::Ceil | PrimFn::Round | PrimFn::Sum | PrimFn::Prod | PrimFn::Len | PrimFn::Shape | PrimFn::Reverse | PrimFn::Transpose | PrimFn::Norm | PrimFn::ToInt | PrimFn::ToFloat | PrimFn::ToBool | PrimFn::ToChar | PrimFn::ParseInt | PrimFn::ParseFloat | PrimFn::Iota | PrimFn::ToString | PrimFn::Chars => 1,
        PrimFn::Print | PrimFn::Write | PrimFn::ReadLine | PrimFn::ReadKey | PrimFn::ReadFile | PrimFn::Sleep => 1,
        PrimFn::Flush | PrimFn::RawModeEnter | PrimFn::RawModeExit => 1,  // Terminal control (take unit)
        PrimFn::FromChars => 1,  // [n]Char → String
        PrimFn::Lines | PrimFn::Words | PrimFn::Bytes => 1,  // String splitting (unary)
        PrimFn::Re | PrimFn::Im | PrimFn::Conj | PrimFn::Arg => 1,  // Complex decomposition
        PrimFn::Trace | PrimFn::Det | PrimFn::Inv | PrimFn::Diag | PrimFn::Eye | PrimFn::Eig | PrimFn::EigVecs => 1,  // Matrix utilities
        PrimFn::Solve => 2,  // Linear solve (default LU)
        PrimFn::SolveWith => 3,  // Linear solve with method string
        PrimFn::WriteFile | PrimFn::ReadBytes | PrimFn::WriteBytes => 2,  // Binary I/O takes 2 args
        PrimFn::Fold => 3,  // fold f acc arr
        PrimFn::StrEq | PrimFn::StartsWith | PrimFn::EndsWith | PrimFn::Contains => 2,  // String comparison (binary)
        _ => 2,  // Range, StrConcat, Take, Drop, Index etc take 2 args
    }
}

pub fn eval(expr: &Expr) -> EvalResult<Value> { let mut evaluator = Evaluator::new(); evaluator.eval(expr) }
pub fn eval_trace(expr: &Expr) -> EvalResult<Value> { let mut evaluator = Evaluator::new().with_trace(true); evaluator.eval(expr) }
