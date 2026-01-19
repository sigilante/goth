//! Primitive operations for Goth

use crate::value::{Value, Tensor, PrimFn};
use crate::error::{EvalError, EvalResult};
use ordered_float::OrderedFloat;

pub fn apply_binop(op: &goth_ast::op::BinOp, left: Value, right: Value) -> EvalResult<Value> {
    use goth_ast::op::BinOp::*;
    match op {
        Add => add(left, right), Sub => sub(left, right), Mul => mul(left, right),
        Div => div(left, right), Pow => pow(left, right), Mod => modulo(left, right),
        PlusMinus => Ok(Value::Uncertain { value: Box::new(left), uncertainty: Box::new(right) }),
        Eq => Ok(Value::Bool(left.deep_eq(&right))), Neq => Ok(Value::Bool(!left.deep_eq(&right))),
        Lt => compare_lt(left, right), Gt => compare_gt(left, right),
        Leq => compare_leq(left, right), Geq => compare_geq(left, right),
        And => logical_and(left, right), Or => logical_or(left, right),
        Compose | Map | Filter | Bind => Err(EvalError::internal("should be handled by evaluator")),
        ZipWith => zip_with(left, right), Concat => concat(left, right),
        Custom(name) => Err(EvalError::not_implemented(format!("custom operator: {}", name))),
    }
}

pub fn apply_unaryop(op: &goth_ast::op::UnaryOp, value: Value) -> EvalResult<Value> {
    use goth_ast::op::UnaryOp::*;
    match op { 
        Neg => negate(value), 
        Not => logical_not(value), 
        Sum => sum(value), 
        Prod => product(value), 
        Scan => scan(value),
        Sqrt => sqrt(value),
        Floor => floor(value),
        Ceil => ceil(value),
    }
}

pub fn apply_prim(prim: PrimFn, args: Vec<Value>) -> EvalResult<Value> {
    match prim {
        PrimFn::Add => binary_args(&args, add), PrimFn::Sub => binary_args(&args, sub),
        PrimFn::Mul => binary_args(&args, mul), PrimFn::Div => binary_args(&args, div),
        PrimFn::Mod => binary_args(&args, modulo), PrimFn::Neg => unary_args(&args, negate),
        PrimFn::Abs => unary_args(&args, abs),
        PrimFn::Exp => unary_args(&args, exp), PrimFn::Ln => unary_args(&args, ln),
        PrimFn::Sqrt => unary_args(&args, sqrt), PrimFn::Sin => unary_args(&args, sin),
        PrimFn::Cos => unary_args(&args, cos), PrimFn::Tan => unary_args(&args, tan),
        PrimFn::Pow => binary_args(&args, pow), PrimFn::Floor => unary_args(&args, floor),
        PrimFn::Ceil => unary_args(&args, ceil), PrimFn::Round => unary_args(&args, round),
        PrimFn::Eq => binary_args(&args, |a, b| Ok(Value::Bool(a.deep_eq(&b)))),
        PrimFn::Neq => binary_args(&args, |a, b| Ok(Value::Bool(!a.deep_eq(&b)))),
        PrimFn::Lt => binary_args(&args, compare_lt), PrimFn::Gt => binary_args(&args, compare_gt),
        PrimFn::Leq => binary_args(&args, compare_leq), PrimFn::Geq => binary_args(&args, compare_geq),
        PrimFn::And => binary_args(&args, logical_and), PrimFn::Or => binary_args(&args, logical_or),
        PrimFn::Not => unary_args(&args, logical_not),
        PrimFn::Sum => unary_args(&args, sum), PrimFn::Prod => unary_args(&args, product),
        PrimFn::Len => unary_args(&args, len), PrimFn::Shape => unary_args(&args, shape),
        PrimFn::Reverse => unary_args(&args, reverse), PrimFn::Concat => binary_args(&args, concat),
        PrimFn::Dot => binary_args(&args, dot), PrimFn::Norm => unary_args(&args, norm),
        PrimFn::MatMul => binary_args(&args, matmul), PrimFn::Transpose => unary_args(&args, transpose),
        PrimFn::ToInt => unary_args(&args, to_int), PrimFn::ToFloat => unary_args(&args, to_float),
        PrimFn::ToBool => unary_args(&args, to_bool), PrimFn::ToChar => unary_args(&args, to_char),
        PrimFn::Print => { for arg in &args { println!("{}", arg); } Ok(Value::Unit) }
        PrimFn::ReadLine => {
            use std::io::{self, BufRead};
            let mut line = String::new();
            io::stdin().lock().read_line(&mut line).map_err(|e| EvalError::IoError(e.to_string()))?;
            // Remove trailing newline
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            Ok(Value::string(&line))
        }
        PrimFn::Iota => unary_args(&args, iota),
        PrimFn::Range => binary_args(&args, range),
        _ => Err(EvalError::not_implemented(format!("primitive: {:?}", prim))),
    }
}

fn unary_args<F>(args: &[Value], f: F) -> EvalResult<Value> where F: FnOnce(Value) -> EvalResult<Value> {
    if args.len() != 1 { return Err(EvalError::ArityMismatch { expected: 1, got: args.len() }); }
    f(args[0].clone())
}

fn binary_args<F>(args: &[Value], f: F) -> EvalResult<Value> where F: FnOnce(Value, Value) -> EvalResult<Value> {
    if args.len() != 2 { return Err(EvalError::ArityMismatch { expected: 2, got: args.len() }); }
    f(args[0].clone(), args[1].clone())
}

fn add(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0 + b.0))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(*a as f64 + b.0))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0 + *b as f64))),
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch(format!("Cannot add tensors with shapes {:?} and {:?}", a.shape, b.shape))); }
            let result = a.zip_with(b, |x, y| add(x, y).unwrap_or(Value::Error("add failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(Value::Tensor(result))
        }
        (Value::Tensor(t), scalar) if scalar.is_numeric() => Ok(Value::Tensor(t.map(|x| add(x, scalar.clone()).unwrap_or(Value::Error("add failed".into()))))),
        (scalar, Value::Tensor(t)) if scalar.is_numeric() => Ok(Value::Tensor(t.map(|x| add(scalar.clone(), x).unwrap_or(Value::Error("add failed".into()))))),
        _ => Err(EvalError::type_error_msg(format!("Cannot add {} and {}", left.type_name(), right.type_name()))),
    }
}

fn sub(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0 - b.0))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(*a as f64 - b.0))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0 - *b as f64))),
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch("Tensor shapes must match for subtraction")); }
            let result = a.zip_with(b, |x, y| sub(x, y).unwrap_or(Value::Error("sub failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(Value::Tensor(result))
        }
        _ => Err(EvalError::type_error_msg(format!("Cannot subtract {} and {}", left.type_name(), right.type_name()))),
    }
}

fn mul(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0 * b.0))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(*a as f64 * b.0))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0 * *b as f64))),
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch("Tensor shapes must match for multiplication")); }
            let result = a.zip_with(b, |x, y| mul(x, y).unwrap_or(Value::Error("mul failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(Value::Tensor(result))
        }
        (Value::Tensor(t), scalar) if scalar.is_numeric() => Ok(Value::Tensor(t.map(|x| mul(x, scalar.clone()).unwrap_or(Value::Error("mul failed".into()))))),
        (scalar, Value::Tensor(t)) if scalar.is_numeric() => Ok(Value::Tensor(t.map(|x| mul(scalar.clone(), x).unwrap_or(Value::Error("mul failed".into()))))),
        _ => Err(EvalError::type_error_msg(format!("Cannot multiply {} and {}", left.type_name(), right.type_name()))),
    }
}

fn div(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => if *b == 0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Int(a / b)) },
        (Value::Float(a), Value::Float(b)) => if b.0 == 0.0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(a.0 / b.0))) },
        (Value::Int(a), Value::Float(b)) => if b.0 == 0.0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(*a as f64 / b.0))) },
        (Value::Float(a), Value::Int(b)) => if *b == 0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(a.0 / *b as f64))) },
        (Value::Tensor(t), scalar) if scalar.is_numeric() => Ok(Value::Tensor(t.map(|x| div(x, scalar.clone()).unwrap_or(Value::Error("div failed".into()))))),
        _ => Err(EvalError::type_error_msg(format!("Cannot divide {} by {}", left.type_name(), right.type_name()))),
    }
}

fn modulo(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => if *b == 0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Int(a % b)) },
        (Value::Float(a), Value::Float(b)) => if b.0 == 0.0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(a.0 % b.0))) },
        _ => Err(EvalError::type_error_msg(format!("Cannot compute modulo of {} and {}", left.type_name(), right.type_name()))),
    }
}

fn pow(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => if *b >= 0 { Ok(Value::Int(a.pow(*b as u32))) } else { Ok(Value::Float(OrderedFloat((*a as f64).powi(*b as i32)))) },
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0.powf(b.0)))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0.powi(*b as i32)))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat((*a as f64).powf(b.0)))),
        _ => Err(EvalError::type_error_msg(format!("Cannot raise {} to power {}", left.type_name(), right.type_name()))),
    }
}

fn negate(value: Value) -> EvalResult<Value> {
    match value {
        Value::Int(n) => Ok(Value::Int(-n)),
        Value::Float(f) => Ok(Value::Float(OrderedFloat(-f.0))),
        Value::Tensor(t) => Ok(Value::Tensor(t.map(|x| negate(x).unwrap_or(Value::Error("negate failed".into()))))),
        _ => Err(EvalError::type_error("numeric", &value)),
    }
}

fn abs(value: Value) -> EvalResult<Value> {
    match value { Value::Int(n) => Ok(Value::Int(n.abs())), Value::Float(f) => Ok(Value::Float(OrderedFloat(f.0.abs()))), _ => Err(EvalError::type_error("numeric", &value)) }
}

fn exp(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.exp()))) }
fn ln(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f <= 0.0 { Err(EvalError::type_error_msg("ln requires positive argument")) } else { Ok(Value::Float(OrderedFloat(f.ln()))) } }
fn sqrt(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f < 0.0 { Err(EvalError::type_error_msg("sqrt requires non-negative argument")) } else { Ok(Value::Float(OrderedFloat(f.sqrt()))) } }
fn sin(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.sin()))) }
fn cos(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.cos()))) }
fn tan(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.tan()))) }
fn floor(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Int(f.floor() as i128)) }
fn ceil(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Int(f.ceil() as i128)) }
fn round(value: Value) -> EvalResult<Value> { let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Int(f.round() as i128)) }

fn compare_lt(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) < b.0)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a.0 < *b as f64)),
        (Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a < b)),
        _ => Err(EvalError::type_error_msg(format!("Cannot compare {} and {}", left.type_name(), right.type_name()))),
    }
}

fn compare_gt(left: Value, right: Value) -> EvalResult<Value> { compare_lt(right, left) }
fn compare_leq(left: Value, right: Value) -> EvalResult<Value> { logical_not(compare_gt(left, right)?) }
fn compare_geq(left: Value, right: Value) -> EvalResult<Value> { logical_not(compare_lt(left, right)?) }

fn logical_and(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) { (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)), _ => Err(EvalError::type_error_msg(format!("Cannot AND {} and {}", left.type_name(), right.type_name()))) }
}

fn logical_or(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) { (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)), _ => Err(EvalError::type_error_msg(format!("Cannot OR {} and {}", left.type_name(), right.type_name()))) }
}

fn logical_not(value: Value) -> EvalResult<Value> { match value { Value::Bool(b) => Ok(Value::Bool(!b)), _ => Err(EvalError::type_error("Bool", &value)) } }

fn sum(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => Ok(t.sum()),
        Value::Tuple(vs) => { let mut acc = Value::Int(0); for v in vs { acc = add(acc, v)?; } Ok(acc) }
        _ => Err(EvalError::type_error("Tensor or Tuple", &value)),
    }
}

fn product(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => Ok(t.product()),
        Value::Tuple(vs) => { let mut acc = Value::Int(1); for v in vs { acc = mul(acc, v)?; } Ok(acc) }
        _ => Err(EvalError::type_error("Tensor or Tuple", &value)),
    }
}

fn scan(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            let mut running = Value::Int(0);
            let scanned: Vec<Value> = t.iter().map(|v| { running = add(running.clone(), v).unwrap_or(Value::Error("scan failed".into())); running.clone() }).collect();
            Ok(Value::Tensor(Tensor::from_values(t.shape.clone(), scanned)))
        }
        _ => Err(EvalError::type_error("Tensor", &value)),
    }
}

fn len(value: Value) -> EvalResult<Value> { match value { Value::Tensor(t) => Ok(Value::Int(t.len() as i128)), Value::Tuple(vs) => Ok(Value::Int(vs.len() as i128)), _ => Err(EvalError::type_error("Tensor or Tuple", &value)) } }
fn shape(value: Value) -> EvalResult<Value> { match value { Value::Tensor(t) => Ok(Value::Tensor(Tensor::from_ints(t.shape.iter().map(|&d| d as i128).collect()))), _ => Err(EvalError::type_error("Tensor", &value)) } }
fn reverse(value: Value) -> EvalResult<Value> { match value { Value::Tensor(t) => { let mut data = t.to_vec(); data.reverse(); Ok(Value::Tensor(Tensor::from_values(t.shape.clone(), data))) } Value::Tuple(mut vs) => { vs.reverse(); Ok(Value::Tuple(vs)) } _ => Err(EvalError::type_error("Tensor or Tuple", &value)) } }

fn concat(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.rank() != 1 || b.rank() != 1 { return Err(EvalError::not_implemented("concat for rank > 1")); }
            let mut data = a.to_vec(); data.extend(b.iter()); let new_len = a.len() + b.len();
            Ok(Value::Tensor(Tensor::from_values(vec![new_len], data)))
        }
        (Value::Tuple(a), Value::Tuple(b)) => { let mut result = a.clone(); result.extend(b.iter().cloned()); Ok(Value::Tuple(result)) }
        _ => Err(EvalError::type_error_msg(format!("Cannot concat {} and {}", left.type_name(), right.type_name()))),
    }
}

fn zip_with(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch(format!("Cannot zip tensors with shapes {:?} and {:?}", a.shape, b.shape))); }
            let zipped: Vec<Value> = a.iter().zip(b.iter()).map(|(x, y)| Value::Tuple(vec![x, y])).collect();
            Ok(Value::Tensor(Tensor::from_values(a.shape.clone(), zipped)))
        }
        _ => Err(EvalError::type_error_msg(format!("Cannot zip {} and {}", left.type_name(), right.type_name()))),
    }
}

fn dot(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape || a.rank() != 1 { return Err(EvalError::shape_mismatch("Dot product requires vectors of same length")); }
            let zipped = a.zip_with(b, |x, y| mul(x, y).unwrap_or(Value::Error("mul failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(zipped.sum())
        }
        _ => Err(EvalError::type_error("Tensor", &left)),
    }
}

fn norm(value: Value) -> EvalResult<Value> {
    match &value {
        Value::Tensor(t) => { let sum_sq = t.map(|x| { let f = x.coerce_float().unwrap_or(0.0); Value::Float(OrderedFloat(f * f)) }).sum(); sqrt(sum_sq) }
        _ => Err(EvalError::type_error("Tensor", &value)),
    }
}

fn matmul(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.rank() != 2 || b.rank() != 2 { return Err(EvalError::shape_mismatch("Matrix multiplication requires 2D tensors")); }
            let m = a.shape[0]; let n = a.shape[1]; let p = b.shape[1];
            if n != b.shape[0] { return Err(EvalError::shape_mismatch(format!("Matrix dimensions incompatible: [{} {}] × [{} {}]", m, n, b.shape[0], p))); }
            let mut result = vec![Value::Float(OrderedFloat(0.0)); m * p];
            for i in 0..m { for j in 0..p { let mut sum = 0.0f64; for k in 0..n { let a_ik = a.get(&[i, k]).and_then(|v| v.coerce_float()).unwrap_or(0.0); let b_kj = b.get(&[k, j]).and_then(|v| v.coerce_float()).unwrap_or(0.0); sum += a_ik * b_kj; } result[i * p + j] = Value::Float(OrderedFloat(sum)); } }
            Ok(Value::Tensor(Tensor::from_values(vec![m, p], result)))
        }
        _ => Err(EvalError::type_error("Tensor", &left)),
    }
}

fn transpose(value: Value) -> EvalResult<Value> {
    match &value {
        Value::Tensor(t) => {
            if t.rank() != 2 { return Err(EvalError::shape_mismatch("Transpose requires 2D tensor")); }
            let m = t.shape[0]; let n = t.shape[1];
            let mut result = vec![Value::Float(OrderedFloat(0.0)); m * n];
            for i in 0..m { for j in 0..n { if let Some(v) = t.get(&[i, j]) { result[j * m + i] = v; } } }
            Ok(Value::Tensor(Tensor::from_values(vec![n, m], result)))
        }
        _ => Err(EvalError::type_error("Tensor", &value)),
    }
}

fn to_int(value: Value) -> EvalResult<Value> {
    match value {
        Value::Int(n) => Ok(Value::Int(n)), Value::Float(f) => Ok(Value::Int(f.0 as i128)),
        Value::Bool(true) => Ok(Value::Int(1)), Value::Bool(false) => Ok(Value::Int(0)),
        Value::Char(c) => Ok(Value::Int(c as i128)),
        _ => Err(EvalError::type_error_msg(format!("Cannot convert {} to Int", value.type_name()))),
    }
}

fn to_float(value: Value) -> EvalResult<Value> {
    match value {
        Value::Int(n) => Ok(Value::Float(OrderedFloat(n as f64))), Value::Float(f) => Ok(Value::Float(f)),
        Value::Bool(true) => Ok(Value::Float(OrderedFloat(1.0))), Value::Bool(false) => Ok(Value::Float(OrderedFloat(0.0))),
        _ => Err(EvalError::type_error_msg(format!("Cannot convert {} to Float", value.type_name()))),
    }
}

fn to_bool(value: Value) -> EvalResult<Value> {
    match value {
        Value::Bool(b) => Ok(Value::Bool(b)), Value::Int(0) => Ok(Value::Bool(false)), Value::Int(_) => Ok(Value::Bool(true)),
        Value::Float(f) if f.0 == 0.0 => Ok(Value::Bool(false)), Value::Float(_) => Ok(Value::Bool(true)),
        _ => Err(EvalError::type_error_msg(format!("Cannot convert {} to Bool", value.type_name()))),
    }
}

fn to_char(value: Value) -> EvalResult<Value> {
    match value {
        Value::Char(c) => Ok(Value::Char(c)),
        Value::Int(n) => if n >= 0 && n <= 0x10FFFF { char::from_u32(n as u32).map(Value::Char).ok_or_else(|| EvalError::type_error_msg("Invalid Unicode scalar")) } else { Err(EvalError::type_error_msg("Integer out of Unicode range")) },
        _ => Err(EvalError::type_error_msg(format!("Cannot convert {} to Char", value.type_name()))),
    }
}

/// iota n: Generate [0, 1, 2, ..., n-1]
fn iota(n: Value) -> EvalResult<Value> {
    match n {
        Value::Int(count) => {
            if count < 0 {
                return Err(EvalError::type_error_msg("iota requires non-negative integer"));
            }
            let data: Vec<i128> = (0..count).collect();
            Ok(Value::Tensor(Tensor::from_ints(data)))
        }
        _ => Err(EvalError::type_error("integer", &n)),
    }
}

/// range start end: Generate [start, start+1, ..., end-1]
fn range(start: Value, end: Value) -> EvalResult<Value> {
    match (&start, &end) {
        (Value::Int(s), Value::Int(e)) => {
            let data: Vec<i128> = (*s..*e).collect();
            Ok(Value::Tensor(Tensor::from_ints(data)))
        }
        _ => Err(EvalError::type_error_msg(format!(
            "range requires two integers, got {} and {}",
            start.type_name(),
            end.type_name()
        ))),
    }
}
