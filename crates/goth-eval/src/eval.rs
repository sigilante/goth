//! Evaluator for Goth

use crate::value::{Value, Tensor, Closure, Env, PrimFn};
use crate::error::{EvalError, EvalResult, OptionExt};
use crate::prim;
use goth_ast::expr::{Expr, MatchArm, FieldAccess, CastKind, DoOp};
use goth_ast::literal::Literal;
use goth_ast::pattern::Pattern;
use goth_ast::op::BinOp;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

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
            ("sum", PrimFn::Sum), ("prod", PrimFn::Prod), ("len", PrimFn::Len), ("shape", PrimFn::Shape), ("reverse", PrimFn::Reverse), ("concat", PrimFn::Concat),
            ("iota", PrimFn::Iota), ("range", PrimFn::Range),
            ("dot", PrimFn::Dot), ("norm", PrimFn::Norm), ("matmul", PrimFn::MatMul), ("transpose", PrimFn::Transpose),
            ("print", PrimFn::Print), ("read_line", PrimFn::ReadLine),
            ("toInt", PrimFn::ToInt), ("toFloat", PrimFn::ToFloat), ("toBool", PrimFn::ToBool), ("toChar", PrimFn::ToChar),
        ];
        for (name, prim) in prims { self.globals.borrow_mut().insert(name.to_string(), Value::Primitive(*prim)); }
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
            Expr::Lam(body) => Ok(Value::Closure(Closure { arity: 1, body: (**body).clone(), env: env.capture(), preconditions: vec![], postconditions: vec![] })),
            Expr::LamN(n, body) => Ok(Value::Closure(Closure { arity: *n, body: (**body).clone(), env: env.capture(), preconditions: vec![], postconditions: vec![] })),
            Expr::Let { pattern, value, body } => { let val = self.eval_with_env(value, env)?; let mut new_env = env.clone(); self.bind_pattern(pattern, val, &mut new_env)?; self.eval_with_env(body, &new_env) }
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
            Expr::ArrayFill { shape, value } => { let shape_vals: Vec<usize> = shape.iter().map(|e| { let v = self.eval_with_env(e, env)?; v.as_int().map(|n| n as usize).ok_or_else(|| EvalError::type_error("Int", &v)) }).collect::<Result<_, _>>()?; let fill_val = self.eval_with_env(value, env)?; let size: usize = shape_vals.iter().product(); let data = vec![fill_val; size]; Ok(Value::Tensor(Tensor::from_values(shape_vals, data))) }
            Expr::Variant { constructor, payload } => { let payload_val = match payload { Some(p) => Some(self.eval_with_env(p, env)?), None => None }; Ok(Value::variant(constructor.to_string(), payload_val)) }
            Expr::Field(base, access) => { let val = self.eval_with_env(base, env)?; self.access_field(val, access) }
            Expr::Index(base, indices) => { let arr = self.eval_with_env(base, env)?; let idx_vals: Vec<usize> = indices.iter().map(|e| { let v = self.eval_with_env(e, env)?; v.as_int().map(|n| n as usize).ok_or_else(|| EvalError::type_error("Int", &v)) }).collect::<Result<_, _>>()?; self.index_value(arr, &idx_vals) }
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
        match lit { Literal::Int(n) => Value::Int(*n), Literal::Float(f) => Value::float(*f), Literal::Char(c) => Value::Char(*c), Literal::String(s) => Value::string(s), Literal::True => Value::Bool(true), Literal::False => Value::Bool(false), Literal::Unit => Value::Unit }
    }

    fn eval_binop(&mut self, op: &BinOp, left: &Expr, right: &Expr, env: &Env) -> EvalResult<Value> {
        match op {
            BinOp::Map => { let arr = self.eval_with_env(left, env)?; let func = self.eval_with_env(right, env)?; self.eval_map(arr, func) }
            BinOp::Filter => { let arr = self.eval_with_env(left, env)?; let pred = self.eval_with_env(right, env)?; self.eval_filter(arr, pred) }
            BinOp::Bind => { let arr = self.eval_with_env(left, env)?; let func = self.eval_with_env(right, env)?; self.eval_bind(arr, func) }
            BinOp::Compose => { let f = self.eval_with_env(left, env)?; let g = self.eval_with_env(right, env)?; self.eval_compose(f, g) }
            BinOp::And => { let left_val = self.eval_with_env(left, env)?; match left_val { Value::Bool(false) => Ok(Value::Bool(false)), Value::Bool(true) => self.eval_with_env(right, env), _ => Err(EvalError::type_error("Bool", &left_val)) } }
            BinOp::Or => { let left_val = self.eval_with_env(left, env)?; match left_val { Value::Bool(true) => Ok(Value::Bool(true)), Value::Bool(false) => self.eval_with_env(right, env), _ => Err(EvalError::type_error("Bool", &left_val)) } }
            _ => { let left_val = self.eval_with_env(left, env)?; let right_val = self.eval_with_env(right, env)?; prim::apply_binop(op, left_val, right_val) }
        }
    }

    fn apply(&mut self, func: Value, arg: Value) -> EvalResult<Value> {
        match func {
            Value::Closure(closure) => {
                if closure.arity == 1 {
                    let mut new_env = closure.env.clone();
                    new_env.push(arg.clone());
                    
                    // Check preconditions (argument bound as ₀)
                    for (i, pre) in closure.preconditions.iter().enumerate() {
                        let pre_result = self.eval_with_env(pre, &new_env)?;
                        match pre_result {
                            Value::Bool(true) => {},
                            Value::Bool(false) => {
                                return Err(EvalError::PreconditionViolated(
                                    format!("precondition #{} failed", i + 1)
                                ));
                            }
                            _ => {
                                return Err(EvalError::type_error("Bool", &pre_result));
                            }
                        }
                    }
                    
                    // Evaluate body
                    let result = self.eval_with_env(&closure.body, &new_env)?;
                    
                    // Check postconditions (result bound as ₀, argument shifts to ₁)
                    for (i, post) in closure.postconditions.iter().enumerate() {
                        let mut post_env = new_env.clone();
                        post_env.push(result.clone());
                        let post_result = self.eval_with_env(post, &post_env)?;
                        match post_result {
                            Value::Bool(true) => {},
                            Value::Bool(false) => {
                                return Err(EvalError::PostconditionViolated(
                                    format!("postcondition #{} failed", i + 1)
                                ));
                            }
                            _ => {
                                return Err(EvalError::type_error("Bool", &post_result));
                            }
                        }
                    }
                    
                    Ok(result)
                } else {
                    let remaining = (closure.arity - 1) as usize;
                    Ok(Value::Partial { func: Box::new(Value::Closure(closure)), args: vec![arg], remaining })
                }
            }
            Value::Partial { func, mut args, remaining } => {
                args.push(arg);
                if remaining == 1 {
                    match *func {
                        Value::Closure(closure) => {
                            let mut new_env = closure.env.clone();
                            for a in args.iter().rev() {
                                new_env.push(a.clone());
                            }
                            
                            // Check preconditions (all arguments bound as ₀, ₁, ...)
                            for (i, pre) in closure.preconditions.iter().enumerate() {
                                let pre_result = self.eval_with_env(pre, &new_env)?;
                                match pre_result {
                                    Value::Bool(true) => {},
                                    Value::Bool(false) => {
                                        return Err(EvalError::PreconditionViolated(
                                            format!("precondition #{} failed", i + 1)
                                        ));
                                    }
                                    _ => {
                                        return Err(EvalError::type_error("Bool", &pre_result));
                                    }
                                }
                            }
                            
                            // Evaluate body
                            let result = self.eval_with_env(&closure.body, &new_env)?;
                            
                            // Check postconditions (result as ₀, args shift)
                            for (i, post) in closure.postconditions.iter().enumerate() {
                                let mut post_env = new_env.clone();
                                post_env.push(result.clone());
                                let post_result = self.eval_with_env(post, &post_env)?;
                                match post_result {
                                    Value::Bool(true) => {},
                                    Value::Bool(false) => {
                                        return Err(EvalError::PostconditionViolated(
                                            format!("postcondition #{} failed", i + 1)
                                        ));
                                    }
                                    _ => {
                                        return Err(EvalError::type_error("Bool", &post_result));
                                    }
                                }
                            }
                            
                            Ok(result)
                        }
                        Value::Primitive(prim) => prim::apply_prim(prim, args),
                        _ => Err(EvalError::type_error("function", &func)),
                    }
                } else {
                    Ok(Value::Partial { func, args, remaining: remaining - 1 })
                }
            }
            Value::Primitive(prim) => {
                let arity = prim_arity(prim);
                if arity == 1 {
                    prim::apply_prim(prim, vec![arg])
                } else {
                    Ok(Value::Partial { func: Box::new(Value::Primitive(prim)), args: vec![arg], remaining: arity - 1 })
                }
            }
            _ => Err(EvalError::type_error("function", &func)),
        }
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
            Pattern::Wildcard => Ok(true),
            Pattern::Var(_) => { env.push(val.clone()); Ok(true) }
            Pattern::Lit(lit) => { let lit_val = self.eval_literal(lit); Ok(val.deep_eq(&lit_val)) }
            Pattern::Array(pats) => { match val { Value::Tensor(t) => { if t.rank() != 1 || t.len() != pats.len() { return Ok(false); } for (i, pat) in pats.iter().enumerate() { let elem = t.get_flat(i).unwrap(); if !self.match_pattern(pat, &elem, env)? { return Ok(false); } } Ok(true) } _ => Ok(false) } }
            Pattern::ArraySplit { head, tail } => { match val { Value::Tensor(t) => { if t.rank() != 1 || t.len() < head.len() { return Ok(false); } for (i, pat) in head.iter().enumerate() { let elem = t.get_flat(i).unwrap(); if !self.match_pattern(pat, &elem, env)? { return Ok(false); } } let tail_data: Vec<Value> = (head.len()..t.len()).map(|i| t.get_flat(i).unwrap()).collect(); let tail_tensor = Tensor::from_values(vec![tail_data.len()], tail_data); self.match_pattern(tail, &Value::Tensor(tail_tensor), env) } _ => Ok(false) } }
            Pattern::Tuple(pats) => { match val { Value::Tuple(vals) => { if vals.len() != pats.len() { return Ok(false); } for (pat, v) in pats.iter().zip(vals) { if !self.match_pattern(pat, v, env)? { return Ok(false); } } Ok(true) } Value::Unit if pats.is_empty() => Ok(true), _ => Ok(false) } }
            Pattern::Variant { constructor, payload } => { match val { Value::Variant { tag, payload: val_payload } => { if tag.as_str() != constructor.as_ref() { return Ok(false); } match (payload, val_payload) { (None, None) => Ok(true), (Some(pat), Some(v)) => self.match_pattern(pat, v, env), _ => Ok(false) } } _ => Ok(false) } }
            Pattern::Typed(inner, _ty) => self.match_pattern(inner, val, env),
            Pattern::Or(p1, p2) => { let mut env1 = env.clone(); if self.match_pattern(p1, val, &mut env1)? { *env = env1; return Ok(true); } self.match_pattern(p2, val, env) }
            Pattern::Guard(inner, _) => self.match_pattern(inner, val, env),
        }
    }

    fn bind_pattern(&self, pattern: &Pattern, val: Value, env: &mut Env) -> EvalResult<()> { if self.match_pattern(pattern, &val, env)? { Ok(()) } else { Err(EvalError::MatchFailed) } }

    fn values_to_tensor(&self, values: Vec<Value>) -> Value {
        if values.is_empty() { return Value::Tensor(Tensor::from_ints(vec![])); }
        let all_int = values.iter().all(|v| matches!(v, Value::Int(_)));
        let all_float = values.iter().all(|v| matches!(v, Value::Float(_) | Value::Int(_)));
        let all_bool = values.iter().all(|v| matches!(v, Value::Bool(_)));
        let all_char = values.iter().all(|v| matches!(v, Value::Char(_)));
        if all_int { Value::Tensor(Tensor::from_ints(values.iter().map(|v| v.as_int().unwrap()).collect())) }
        else if all_float { Value::Tensor(Tensor::from_floats(values.iter().map(|v| v.coerce_float().unwrap()).collect())) }
        else if all_bool { Value::Tensor(Tensor::from_bools(values.iter().map(|v| v.as_bool().unwrap()).collect())) }
        else if all_char { let chars: Vec<char> = values.iter().map(|v| v.as_char().unwrap()).collect(); let len = chars.len(); Value::Tensor(Tensor { shape: vec![len], data: crate::value::TensorData::Char(chars) }) }
        else { Value::Tensor(Tensor::from_values(vec![values.len()], values)) }
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
            Value::Tensor(t) => { if t.rank() != 1 { return Err(EvalError::not_implemented("slicing rank > 1")); } let end = end.unwrap_or(t.len()); if start > end || end > t.len() { return Err(EvalError::IndexOutOfBounds { index: end, size: t.len() }); } let data: Vec<Value> = (start..end).map(|i| t.get_flat(i).unwrap()).collect(); Ok(Value::Tensor(Tensor::from_values(vec![data.len()], data))) }
            _ => Err(EvalError::type_error("Tensor", &val)),
        }
    }

    fn eval_map(&mut self, arr: Value, func: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => { let results: Vec<Value> = t.iter().map(|elem| self.apply(func.clone(), elem)).collect::<Result<_, _>>()?; Ok(Value::Tensor(Tensor::from_values(t.shape.clone(), results))) }
            Value::Tuple(vs) => { let results: Vec<Value> = vs.into_iter().map(|elem| self.apply(func.clone(), elem)).collect::<Result<_, _>>()?; Ok(Value::Tuple(results)) }
            _ => Err(EvalError::type_error("Tensor or Tuple", &arr)),
        }
    }

    fn eval_filter(&mut self, arr: Value, pred: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => { let results: Vec<Value> = t.iter().filter_map(|elem| { let keep = self.apply(pred.clone(), elem.clone()).ok()?; match keep { Value::Bool(true) => Some(elem), _ => None } }).collect(); Ok(Value::Tensor(Tensor::from_values(vec![results.len()], results))) }
            Value::Tuple(vs) => { let results: Vec<Value> = vs.into_iter().filter_map(|elem| { let keep = self.apply(pred.clone(), elem.clone()).ok()?; match keep { Value::Bool(true) => Some(elem), _ => None } }).collect(); Ok(Value::Tuple(results)) }
            _ => Err(EvalError::type_error("Tensor or Tuple", &arr)),
        }
    }

    fn eval_bind(&mut self, arr: Value, func: Value) -> EvalResult<Value> {
        match arr {
            Value::Tensor(t) => { let mut results = Vec::new(); for elem in t.iter() { let mapped = self.apply(func.clone(), elem)?; match mapped { Value::Tensor(inner) => results.extend(inner.iter()), Value::Tuple(inner) => results.extend(inner), other => results.push(other) } } Ok(Value::Tensor(Tensor::from_values(vec![results.len()], results))) }
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
        Ok(Value::Closure(Closure { arity: 1, body, env, preconditions: vec![], postconditions: vec![] }))
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

fn prim_arity(prim: PrimFn) -> usize {
    match prim {
        PrimFn::Neg | PrimFn::Abs | PrimFn::Not | PrimFn::Exp | PrimFn::Ln | PrimFn::Sqrt | PrimFn::Sin | PrimFn::Cos | PrimFn::Tan | PrimFn::Floor | PrimFn::Ceil | PrimFn::Round | PrimFn::Sum | PrimFn::Prod | PrimFn::Len | PrimFn::Shape | PrimFn::Reverse | PrimFn::Transpose | PrimFn::Norm | PrimFn::ToInt | PrimFn::ToFloat | PrimFn::ToBool | PrimFn::ToChar | PrimFn::Iota => 1,
        PrimFn::Print | PrimFn::ReadLine => 1,  // ReadLine takes unit, returns string
        _ => 2,  // Range takes 2 args (start, end)
    }
}

pub fn eval(expr: &Expr) -> EvalResult<Value> { let mut evaluator = Evaluator::new(); evaluator.eval(expr) }
pub fn eval_trace(expr: &Expr) -> EvalResult<Value> { let mut evaluator = Evaluator::new().with_trace(true); evaluator.eval(expr) }
