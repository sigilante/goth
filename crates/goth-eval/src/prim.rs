//! Primitive operations for Goth

use std::rc::Rc;
use crate::value::{Value, Tensor, PrimFn};
use crate::error::{EvalError, EvalResult};
use ordered_float::OrderedFloat;

// Store original terminal settings for raw mode restoration
#[cfg(unix)]
static ORIGINAL_TERMIOS: std::sync::Mutex<Option<libc::termios>> = std::sync::Mutex::new(None);

/// Restore terminal to its original settings if raw mode was entered.
/// Safe to call multiple times — no-ops if terminal is already restored.
#[cfg(unix)]
pub fn restore_terminal() {
    use std::os::unix::io::AsRawFd;
    if let Ok(mut guard) = ORIGINAL_TERMIOS.lock() {
        if let Some(termios) = guard.take() {
            let fd = std::io::stdin().as_raw_fd();
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &termios); }
        }
    }
}

#[cfg(not(unix))]
pub fn restore_terminal() {}

// ============ Uncertainty propagation helpers ============

/// Extract (value, uncertainty) as f64 from an Uncertain value.
fn uncertain_parts(v: &Value) -> Option<(f64, f64)> {
    if let Value::Uncertain { value, uncertainty } = v {
        Some((value.coerce_float()?, uncertainty.coerce_float()?))
    } else {
        None
    }
}

/// Construct an Uncertain value from f64 parts (uncertainty is always non-negative).
fn make_uncertain(val: f64, unc: f64) -> Value {
    Value::Uncertain {
        value: Box::new(Value::Float(OrderedFloat(val))),
        uncertainty: Box::new(Value::Float(OrderedFloat(unc.abs()))),
    }
}

/// Extract the value part of an Uncertain, or return the value itself.
/// Used for value-equality comparisons that ignore uncertainty.
fn uncertain_value(v: &Value) -> Value {
    if let Value::Uncertain { value, .. } = v {
        (**value).clone()
    } else {
        v.clone()
    }
}

/// Additive propagation: δ = √(δa² + δb²)
fn additive_unc(da: f64, db: f64) -> f64 {
    (da * da + db * db).sqrt()
}

/// Multiplicative propagation for a*b: δ = |a*b| * √((δa/a)² + (δb/b)²)
fn multiplicative_unc(a: f64, b: f64, da: f64, db: f64) -> f64 {
    let result = a * b;
    if result == 0.0 {
        // Avoid division by zero: fall back to linear propagation
        (b * da).hypot(a * db).abs()
    } else {
        let ra = if a != 0.0 { da / a } else { 0.0 };
        let rb = if b != 0.0 { db / b } else { 0.0 };
        result.abs() * (ra * ra + rb * rb).sqrt()
    }
}

pub fn apply_binop(op: &goth_ast::op::BinOp, left: Value, right: Value) -> EvalResult<Value> {
    use goth_ast::op::BinOp::*;
    match op {
        Add => add(left, right), Sub => sub(left, right), Mul => mul(left, right),
        Div => div(left, right), Pow => pow(left, right), Mod => modulo(left, right),
        PlusMinus => Ok(Value::Uncertain { value: Box::new(left), uncertainty: Box::new(right) }),
        Eq => Ok(Value::Bool(uncertain_value(&left).deep_eq(&uncertain_value(&right)))),
        StructEq => Ok(Value::Bool(left.deep_eq(&right))),
        Neq => Ok(Value::Bool(!uncertain_value(&left).deep_eq(&uncertain_value(&right)))),
        Lt => compare_lt(left, right), Gt => compare_gt(left, right),
        Leq => compare_leq(left, right), Geq => compare_geq(left, right),
        And => logical_and(left, right), Or => logical_or(left, right),
        Compose | Map | Filter | Bind | Write | Read => Err(EvalError::internal("should be handled by evaluator")),
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
        Round => round(value),
        Gamma => gamma(value),
        Ln => ln(value),
        Log10 => log10(value),
        Log2 => log2(value),
        Exp => exp(value),
        Sin => sin(value),
        Cos => cos(value),
        Tan => tan(value),
        Asin => asin(value),
        Acos => acos(value),
        Atan => atan(value),
        Sinh => sinh(value),
        Cosh => cosh(value),
        Tanh => tanh(value),
        Abs => abs(value),
        Sign => sign(value),
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
        PrimFn::Eq => binary_args(&args, |a, b| Ok(Value::Bool(uncertain_value(&a).deep_eq(&uncertain_value(&b))))),
        PrimFn::Neq => binary_args(&args, |a, b| Ok(Value::Bool(!uncertain_value(&a).deep_eq(&uncertain_value(&b))))),
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
        PrimFn::ParseInt => unary_args(&args, parse_int), PrimFn::ParseFloat => unary_args(&args, parse_float),
        PrimFn::Print => {
            for arg in &args {
                // Print strings without quotes for raw output
                if let Value::Tensor(t) = arg {
                    if let Some(s) = t.to_string_value() {
                        print!("{}", s);
                        continue;
                    }
                }
                print!("{}", arg);
            }
            println!();
            Ok(Value::Unit)
        }
        PrimFn::Write => {
            use std::io::Write;
            for arg in &args {
                // Write strings without quotes for raw output (no trailing newline)
                if let Value::Tensor(t) = arg {
                    if let Some(s) = t.to_string_value() {
                        print!("{}", s);
                        continue;
                    }
                }
                print!("{}", arg);
            }
            std::io::stdout().flush().map_err(|e| EvalError::IoError(e.to_string()))?;
            Ok(Value::Unit)
        }
        PrimFn::Flush => {
            use std::io::Write;
            std::io::stdout().flush().map_err(|e| EvalError::IoError(e.to_string()))?;
            Ok(Value::Unit)
        }
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
        PrimFn::ReadKey => {
            // Read a single character/byte from stdin (requires raw mode to be set)
            use std::io::Read;
            let mut buf = [0u8; 1];
            match std::io::stdin().lock().read_exact(&mut buf) {
                Ok(_) => Ok(Value::Int(buf[0] as i128)),
                Err(e) => Err(EvalError::IoError(e.to_string())),
            }
        }
        PrimFn::RawModeEnter => {
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let fd = std::io::stdin().as_raw_fd();
                unsafe {
                    let mut termios: libc::termios = std::mem::zeroed();
                    if libc::tcgetattr(fd, &mut termios) != 0 {
                        return Err(EvalError::IoError("Failed to get terminal attributes".to_string()));
                    }
                    // Save original termios for restoration
                    if let Ok(mut guard) = ORIGINAL_TERMIOS.lock() {
                        *guard = Some(termios);
                    }
                    // Disable canonical mode and echo
                    termios.c_lflag &= !(libc::ICANON | libc::ECHO);
                    // Set minimum characters and timeout
                    termios.c_cc[libc::VMIN] = 1;
                    termios.c_cc[libc::VTIME] = 0;
                    if libc::tcsetattr(fd, libc::TCSANOW, &termios) != 0 {
                        return Err(EvalError::IoError("Failed to set terminal attributes".to_string()));
                    }
                }
                Ok(Value::Unit)
            }
            #[cfg(not(unix))]
            {
                Err(EvalError::not_implemented("rawModeEnter only supported on Unix"))
            }
        }
        PrimFn::RawModeExit => {
            #[cfg(unix)]
            {
                restore_terminal();
                Ok(Value::Unit)
            }
            #[cfg(not(unix))]
            {
                Err(EvalError::not_implemented("rawModeExit only supported on Unix"))
            }
        }
        PrimFn::Sleep => {
            if args.len() != 1 { return Err(EvalError::ArityMismatch { expected: 1, got: args.len() }); }
            let ms = match &args[0] {
                Value::Int(n) => *n as u64,
                Value::Float(f) => f.0 as u64,
                _ => return Err(EvalError::type_error("Int or Float", &args[0])),
            };
            std::thread::sleep(std::time::Duration::from_millis(ms));
            Ok(Value::Unit)
        }
        PrimFn::ReadFile => {
            if args.len() != 1 { return Err(EvalError::ArityMismatch { expected: 1, got: args.len() }); }
            let path = match &args[0] {
                Value::Tensor(t) => t.to_string_value().ok_or_else(|| EvalError::type_error("String", &args[0]))?,
                _ => return Err(EvalError::type_error("String", &args[0])),
            };
            use std::fs;
            let contents = fs::read_to_string(&path).map_err(|e| EvalError::IoError(format!("Failed to read '{}': {}", path, e)))?;
            Ok(Value::string(&contents))
        }
        PrimFn::WriteFile => {
            if args.len() != 2 { return Err(EvalError::ArityMismatch { expected: 2, got: args.len() }); }
            let path = match &args[0] {
                Value::Tensor(t) => t.to_string_value().ok_or_else(|| EvalError::type_error("String", &args[0]))?,
                _ => return Err(EvalError::type_error("String", &args[0])),
            };
            let contents = match &args[1] {
                Value::Tensor(t) => t.to_string_value().ok_or_else(|| EvalError::type_error("String", &args[1]))?,
                _ => return Err(EvalError::type_error("String", &args[1])),
            };
            use std::fs;
            fs::write(&path, &contents).map_err(|e| EvalError::IoError(format!("Failed to write '{}': {}", path, e)))?;
            Ok(Value::Unit)
        }
        PrimFn::ReadBytes => {
            if args.len() != 2 { return Err(EvalError::ArityMismatch { expected: 2, got: args.len() }); }
            let count = match &args[0] {
                Value::Int(n) => *n as usize,
                _ => return Err(EvalError::type_error("Int", &args[0])),
            };
            let path = match &args[1] {
                Value::Tensor(t) => t.to_string_value().ok_or_else(|| EvalError::type_error("String", &args[1]))?,
                _ => return Err(EvalError::type_error("String", &args[1])),
            };
            use std::io::Read;
            let mut file = std::fs::File::open(&path)
                .map_err(|e| EvalError::IoError(format!("Failed to open '{}': {}", path, e)))?;
            let mut buf = vec![0u8; count];
            file.read_exact(&mut buf)
                .map_err(|e| EvalError::IoError(format!("Failed to read {} bytes from '{}': {}", count, path, e)))?;
            let byte_vals: Vec<i128> = buf.iter().map(|&b| b as i128).collect();
            Ok(Value::Tensor(Rc::new(Tensor::from_ints(byte_vals))))
        }
        PrimFn::WriteBytes => {
            if args.len() != 2 { return Err(EvalError::ArityMismatch { expected: 2, got: args.len() }); }
            let bytes: Vec<u8> = match &args[0] {
                Value::Tensor(t) => t.to_vec().iter().map(|v| match v {
                    Value::Int(n) => Ok(*n as u8),
                    _ => Err(EvalError::type_error("Int", v)),
                }).collect::<Result<Vec<u8>, _>>()?,
                _ => return Err(EvalError::type_error("Tensor", &args[0])),
            };
            let path = match &args[1] {
                Value::Tensor(t) => t.to_string_value().ok_or_else(|| EvalError::type_error("String", &args[1]))?,
                _ => return Err(EvalError::type_error("String", &args[1])),
            };
            std::fs::write(&path, &bytes)
                .map_err(|e| EvalError::IoError(format!("Failed to write '{}': {}", path, e)))?;
            Ok(Value::Unit)
        }
        PrimFn::Iota => unary_args(&args, iota),
        PrimFn::Range => binary_args(&args, range),
        PrimFn::ToString => unary_args(&args, to_string),
        PrimFn::Chars => unary_args(&args, chars),
        PrimFn::FromChars => unary_args(&args, from_chars),
        PrimFn::StrConcat => binary_args(&args, str_concat),
        PrimFn::Take => binary_args(&args, take),
        PrimFn::Drop => binary_args(&args, drop_fn),
        PrimFn::Index => binary_args(&args, index),
        PrimFn::Lines => unary_args(&args, lines),
        PrimFn::Words => unary_args(&args, words),
        PrimFn::Bytes => unary_args(&args, bytes),
        PrimFn::StrEq => binary_args(&args, str_eq),
        PrimFn::StartsWith => binary_args(&args, starts_with),
        PrimFn::EndsWith => binary_args(&args, ends_with),
        PrimFn::Contains => binary_args(&args, str_contains),
        PrimFn::BitAnd => binary_args(&args, bitand),
        PrimFn::BitOr => binary_args(&args, bitor),
        PrimFn::BitXor => binary_args(&args, bitxor),
        PrimFn::Shl => binary_args(&args, shl),
        PrimFn::Shr => binary_args(&args, shr),
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
        // Uncertain + Uncertain: additive propagation
        (Value::Uncertain { .. }, Value::Uncertain { .. }) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let (b, db) = uncertain_parts(&right).unwrap();
            Ok(make_uncertain(a + b, additive_unc(da, db)))
        }
        // Uncertain + scalar / scalar + Uncertain: uncertainty passes through
        (Value::Uncertain { .. }, _) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let b = right.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot add Uncertain and {}", right.type_name())))?;
            Ok(make_uncertain(a + b, da))
        }
        (_, Value::Uncertain { .. }) => {
            let a = left.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot add {} and Uncertain", left.type_name())))?;
            let (b, db) = uncertain_parts(&right).unwrap();
            Ok(make_uncertain(a + b, db))
        }
        (Value::Int(a), Value::Int(b)) => a.checked_add(*b).map(Value::Int).ok_or_else(|| EvalError::Overflow(format!("{} + {} overflows", a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0 + b.0))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(*a as f64 + b.0))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0 + *b as f64))),
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch(format!("Cannot add tensors with shapes {:?} and {:?}", a.shape, b.shape))); }
            let result = a.zip_with(b, |x, y| add(x, y).unwrap_or(Value::Error("add failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(Value::Tensor(Rc::new(result)))
        }
        (Value::Tensor(t), scalar) if scalar.is_numeric() => Ok(Value::Tensor(Rc::new(t.map(|x| add(x, scalar.clone()).unwrap_or(Value::Error("add failed".into())))))),
        (scalar, Value::Tensor(t)) if scalar.is_numeric() => Ok(Value::Tensor(Rc::new(t.map(|x| add(scalar.clone(), x).unwrap_or(Value::Error("add failed".into())))))),
        _ => Err(EvalError::type_error_msg(format!("Cannot add {} and {}", left.type_name(), right.type_name()))),
    }
}

fn sub(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Uncertain { .. }, Value::Uncertain { .. }) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let (b, db) = uncertain_parts(&right).unwrap();
            Ok(make_uncertain(a - b, additive_unc(da, db)))
        }
        (Value::Uncertain { .. }, _) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let b = right.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot subtract {} from Uncertain", right.type_name())))?;
            Ok(make_uncertain(a - b, da))
        }
        (_, Value::Uncertain { .. }) => {
            let a = left.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot subtract Uncertain from {}", left.type_name())))?;
            let (b, db) = uncertain_parts(&right).unwrap();
            Ok(make_uncertain(a - b, db))
        }
        (Value::Int(a), Value::Int(b)) => a.checked_sub(*b).map(Value::Int).ok_or_else(|| EvalError::Overflow(format!("{} - {} overflows", a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0 - b.0))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(*a as f64 - b.0))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0 - *b as f64))),
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch("Tensor shapes must match for subtraction")); }
            let result = a.zip_with(b, |x, y| sub(x, y).unwrap_or(Value::Error("sub failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(Value::Tensor(Rc::new(result)))
        }
        _ => Err(EvalError::type_error_msg(format!("Cannot subtract {} and {}", left.type_name(), right.type_name()))),
    }
}

fn mul(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Uncertain { .. }, Value::Uncertain { .. }) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let (b, db) = uncertain_parts(&right).unwrap();
            Ok(make_uncertain(a * b, multiplicative_unc(a, b, da, db)))
        }
        (Value::Uncertain { .. }, _) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let b = right.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot multiply Uncertain and {}", right.type_name())))?;
            Ok(make_uncertain(a * b, (da * b).abs()))
        }
        (_, Value::Uncertain { .. }) => {
            let a = left.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot multiply {} and Uncertain", left.type_name())))?;
            let (b, db) = uncertain_parts(&right).unwrap();
            Ok(make_uncertain(a * b, (a * db).abs()))
        }
        (Value::Int(a), Value::Int(b)) => a.checked_mul(*b).map(Value::Int).ok_or_else(|| EvalError::Overflow(format!("{} × {} overflows", a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0 * b.0))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(*a as f64 * b.0))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(OrderedFloat(a.0 * *b as f64))),
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.shape != b.shape { return Err(EvalError::shape_mismatch("Tensor shapes must match for multiplication")); }
            let result = a.zip_with(b, |x, y| mul(x, y).unwrap_or(Value::Error("mul failed".into()))).ok_or_else(|| EvalError::shape_mismatch("zip failed"))?;
            Ok(Value::Tensor(Rc::new(result)))
        }
        (Value::Tensor(t), scalar) if scalar.is_numeric() => Ok(Value::Tensor(Rc::new(t.map(|x| mul(x, scalar.clone()).unwrap_or(Value::Error("mul failed".into())))))),
        (scalar, Value::Tensor(t)) if scalar.is_numeric() => Ok(Value::Tensor(Rc::new(t.map(|x| mul(scalar.clone(), x).unwrap_or(Value::Error("mul failed".into())))))),
        _ => Err(EvalError::type_error_msg(format!("Cannot multiply {} and {}", left.type_name(), right.type_name()))),
    }
}

fn div(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Uncertain { .. }, Value::Uncertain { .. }) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let (b, db) = uncertain_parts(&right).unwrap();
            if b == 0.0 { return Err(EvalError::DivisionByZero); }
            let result = a / b;
            let ra = if a != 0.0 { da / a } else { 0.0 };
            let rb = db / b;
            Ok(make_uncertain(result, result.abs() * (ra * ra + rb * rb).sqrt()))
        }
        (Value::Uncertain { .. }, _) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let b = right.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot divide Uncertain by {}", right.type_name())))?;
            if b == 0.0 { return Err(EvalError::DivisionByZero); }
            Ok(make_uncertain(a / b, (da / b).abs()))
        }
        (_, Value::Uncertain { .. }) => {
            let a = left.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot divide {} by Uncertain", left.type_name())))?;
            let (b, db) = uncertain_parts(&right).unwrap();
            if b == 0.0 { return Err(EvalError::DivisionByZero); }
            let result = a / b;
            Ok(make_uncertain(result, result.abs() * (db / b).abs()))
        }
        (Value::Int(a), Value::Int(b)) => if *b == 0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Int(a / b)) },
        (Value::Float(a), Value::Float(b)) => if b.0 == 0.0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(a.0 / b.0))) },
        (Value::Int(a), Value::Float(b)) => if b.0 == 0.0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(*a as f64 / b.0))) },
        (Value::Float(a), Value::Int(b)) => if *b == 0 { Err(EvalError::DivisionByZero) } else { Ok(Value::Float(OrderedFloat(a.0 / *b as f64))) },
        (Value::Tensor(t), scalar) if scalar.is_numeric() => Ok(Value::Tensor(Rc::new(t.map(|x| div(x, scalar.clone()).unwrap_or(Value::Error("div failed".into())))))),
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

fn bitand(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
        _ => Err(EvalError::type_error_msg(format!("Cannot bitand {} and {}", left.type_name(), right.type_name())))
    }
}

fn bitor(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
        _ => Err(EvalError::type_error_msg(format!("Cannot bitor {} and {}", left.type_name(), right.type_name())))
    }
}

fn bitxor(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
        _ => Err(EvalError::type_error_msg(format!("Cannot bitxor {} and {}", left.type_name(), right.type_name())))
    }
}

fn shl(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => {
            if *b < 0 || *b > 127 {
                Err(EvalError::Overflow(format!("shift left by {} out of range 0..127", b)))
            } else {
                Ok(Value::Int(a << (*b as u32)))
            }
        }
        _ => Err(EvalError::type_error_msg(format!("Cannot shl {} and {}", left.type_name(), right.type_name())))
    }
}

fn shr(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => {
            if *b < 0 || *b > 127 {
                Err(EvalError::Overflow(format!("shift right by {} out of range 0..127", b)))
            } else {
                Ok(Value::Int(a >> (*b as u32)))
            }
        }
        _ => Err(EvalError::type_error_msg(format!("Cannot shr {} and {}", left.type_name(), right.type_name())))
    }
}

fn pow(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        // (a ± δa) ^ (b ± δb): full general case
        (Value::Uncertain { .. }, Value::Uncertain { .. }) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let (b, db) = uncertain_parts(&right).unwrap();
            let result = a.powf(b);
            let term_a = if a != 0.0 { b * da / a } else { 0.0 };
            let term_b = if a > 0.0 { a.ln() * db } else { 0.0 };
            Ok(make_uncertain(result, result.abs() * (term_a * term_a + term_b * term_b).sqrt()))
        }
        // (a ± δa) ^ n: δ = |a^n * n * δa / a|
        (Value::Uncertain { .. }, _) => {
            let (a, da) = uncertain_parts(&left).unwrap();
            let b = right.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot raise Uncertain to power {}", right.type_name())))?;
            let result = a.powf(b);
            let unc = if a != 0.0 { (result * b * da / a).abs() } else { 0.0 };
            Ok(make_uncertain(result, unc))
        }
        // a ^ (b ± δb): δ = |a^b * ln(a) * δb|
        (_, Value::Uncertain { .. }) => {
            let a = left.coerce_float().ok_or_else(|| EvalError::type_error_msg(format!("Cannot raise {} to uncertain power", left.type_name())))?;
            let (b, db) = uncertain_parts(&right).unwrap();
            let result = a.powf(b);
            let unc = if a > 0.0 { (result * a.ln() * db).abs() } else { 0.0 };
            Ok(make_uncertain(result, unc))
        }
        (Value::Int(a), Value::Int(b)) => {
            if *b < 0 {
                Ok(Value::Float(OrderedFloat((*a as f64).powi(*b as i32))))
            } else if *b > u32::MAX as i128 {
                Err(EvalError::Overflow(format!("exponent {} too large", b)))
            } else {
                a.checked_pow(*b as u32)
                    .map(|r| Value::Int(r))
                    .ok_or_else(|| EvalError::Overflow(format!("{} ^ {} overflows i128", a, b)))
            }
        }
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat(a.0.powf(b.0)))),
        (Value::Float(a), Value::Int(b)) => {
            if *b > i32::MAX as i128 || *b < i32::MIN as i128 {
                Ok(Value::Float(OrderedFloat(a.0.powf(*b as f64))))
            } else {
                Ok(Value::Float(OrderedFloat(a.0.powi(*b as i32))))
            }
        }
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(OrderedFloat((*a as f64).powf(b.0)))),
        _ => Err(EvalError::type_error_msg(format!("Cannot raise {} to power {}", left.type_name(), right.type_name()))),
    }
}

fn negate(value: Value) -> EvalResult<Value> {
    match value {
        Value::Uncertain { .. } => {
            let (a, da) = uncertain_parts(&value).unwrap();
            Ok(make_uncertain(-a, da))
        }
        Value::Int(n) => n.checked_neg().map(Value::Int).ok_or_else(|| EvalError::Overflow(format!("negation of {} overflows", n))),
        Value::Float(f) => Ok(Value::Float(OrderedFloat(-f.0))),
        Value::Tensor(t) => Ok(Value::Tensor(Rc::new(t.map(|x| negate(x).unwrap_or(Value::Error("negate failed".into())))))),
        _ => Err(EvalError::type_error("numeric", &value)),
    }
}

fn abs(value: Value) -> EvalResult<Value> {
    match value {
        Value::Uncertain { .. } => {
            let (a, da) = uncertain_parts(&value).unwrap();
            Ok(make_uncertain(a.abs(), da))
        }
        Value::Int(n) => Ok(Value::Int(n.abs())),
        Value::Float(f) => Ok(Value::Float(OrderedFloat(f.0.abs()))),
        _ => Err(EvalError::type_error("numeric", &value)),
    }
}

fn exp(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { let r = a.exp(); return Ok(make_uncertain(r, r * da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.exp())))
}
fn ln(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { if a <= 0.0 { return Err(EvalError::type_error_msg("ln requires positive argument")); } return Ok(make_uncertain(a.ln(), da / a.abs())); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f <= 0.0 { Err(EvalError::type_error_msg("ln requires positive argument")) } else { Ok(Value::Float(OrderedFloat(f.ln()))) }
}
fn log10(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { if a <= 0.0 { return Err(EvalError::type_error_msg("log10 requires positive argument")); } return Ok(make_uncertain(a.log10(), da / (a.abs() * std::f64::consts::LN_10))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f <= 0.0 { Err(EvalError::type_error_msg("log10 requires positive argument")) } else { Ok(Value::Float(OrderedFloat(f.log10()))) }
}
fn log2(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { if a <= 0.0 { return Err(EvalError::type_error_msg("log2 requires positive argument")); } return Ok(make_uncertain(a.log2(), da / (a.abs() * std::f64::consts::LN_2))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f <= 0.0 { Err(EvalError::type_error_msg("log2 requires positive argument")) } else { Ok(Value::Float(OrderedFloat(f.log2()))) }
}
fn sqrt(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { if a < 0.0 { return Err(EvalError::type_error_msg("sqrt requires non-negative argument")); } let r = a.sqrt(); return Ok(make_uncertain(r, da / (2.0 * r))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f < 0.0 { Err(EvalError::type_error_msg("sqrt requires non-negative argument")) } else { Ok(Value::Float(OrderedFloat(f.sqrt()))) }
}
fn sin(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.sin(), a.cos().abs() * da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.sin())))
}
fn cos(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.cos(), a.sin().abs() * da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.cos())))
}
fn tan(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { let c = a.cos(); return Ok(make_uncertain(a.tan(), da / (c * c))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.tan())))
}
fn asin(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { if a < -1.0 || a > 1.0 { return Err(EvalError::type_error_msg("asin requires argument in [-1, 1]")); } return Ok(make_uncertain(a.asin(), da / (1.0 - a * a).sqrt())); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f < -1.0 || f > 1.0 { Err(EvalError::type_error_msg("asin requires argument in [-1, 1]")) } else { Ok(Value::Float(OrderedFloat(f.asin()))) }
}
fn acos(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { if a < -1.0 || a > 1.0 { return Err(EvalError::type_error_msg("acos requires argument in [-1, 1]")); } return Ok(make_uncertain(a.acos(), da / (1.0 - a * a).sqrt())); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; if f < -1.0 || f > 1.0 { Err(EvalError::type_error_msg("acos requires argument in [-1, 1]")) } else { Ok(Value::Float(OrderedFloat(f.acos()))) }
}
fn atan(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.atan(), da / (1.0 + a * a))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.atan())))
}
fn sinh(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.sinh(), a.cosh() * da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.sinh())))
}
fn cosh(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.cosh(), a.sinh().abs() * da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.cosh())))
}
fn tanh(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { let c = a.cosh(); return Ok(make_uncertain(a.tanh(), da / (c * c))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(f.tanh())))
}
fn floor(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.floor(), da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Int(f.floor() as i128))
}
fn ceil(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.ceil(), da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Int(f.ceil() as i128))
}
fn round(value: Value) -> EvalResult<Value> {
    if let Some((a, da)) = uncertain_parts(&value) { return Ok(make_uncertain(a.round(), da)); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Int(f.round() as i128))
}
fn sign(value: Value) -> EvalResult<Value> {
    // sign is exact: no uncertainty propagation
    if let Some((a, _da)) = uncertain_parts(&value) { return Ok(Value::Float(OrderedFloat(if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 }))); }
    let f = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?; Ok(Value::Float(OrderedFloat(if f > 0.0 { 1.0 } else if f < 0.0 { -1.0 } else { 0.0 })))
}

// Digamma (ψ) function via finite differences for uncertainty propagation.
fn digamma_approx(x: f64) -> f64 {
    let h = 1e-7;
    let gx = lanczos_gamma(x);
    if gx.abs() < 1e-300 { return 0.0; }
    (lanczos_gamma(x + h).ln() - lanczos_gamma(x - h).ln()) / (2.0 * h)
}

/// Raw Lanczos gamma computation (extracted so digamma can call it).
fn lanczos_gamma(x: f64) -> f64 {
    let g = 7.0_f64;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let z = 1.0 - x;
        let mut sum = c[0];
        for i in 1..9 { sum += c[i] / (z + (i as f64) - 1.0); }
        let t = z + g - 0.5;
        let gamma_z = (2.0 * std::f64::consts::PI).sqrt() * t.powf(z - 0.5) * (-t).exp() * sum;
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_z)
    } else {
        let z = x - 1.0;
        let mut sum = c[0];
        for i in 1..9 { sum += c[i] / (z + (i as f64)); }
        let t = z + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum
    }
}

// Gamma function using Lanczos approximation
fn gamma(value: Value) -> EvalResult<Value> {
    // Uncertainty propagation: d/dx Γ(x) = Γ(x) * ψ(x)
    if let Some((a, da)) = uncertain_parts(&value) {
        if a <= 0.0 && a.fract() == 0.0 {
            return Err(EvalError::type_error_msg("gamma undefined for non-positive integers"));
        }
        let gval = lanczos_gamma(a);
        let unc = (gval * digamma_approx(a) * da).abs();
        return Ok(make_uncertain(gval, unc));
    }
    let x = value.coerce_float().ok_or_else(|| EvalError::type_error("numeric", &value))?;
    if x <= 0.0 && x.fract() == 0.0 {
        return Err(EvalError::type_error_msg("gamma undefined for non-positive integers"));
    }
    // Lanczos approximation coefficients
    let g = 7.0_f64;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let result = if x < 0.5 {
        // Use reflection formula: Γ(1-z) * Γ(z) = π / sin(πz)
        let z = 1.0 - x;
        let mut sum = c[0];
        for i in 1..9 {
            sum += c[i] / (z + (i as f64) - 1.0);
        }
        let t = z + g - 0.5;
        let gamma_z = (2.0 * std::f64::consts::PI).sqrt() * t.powf(z - 0.5) * (-t).exp() * sum;
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_z)
    } else {
        let z = x - 1.0;
        let mut sum = c[0];
        for i in 1..9 {
            sum += c[i] / (z + (i as f64));
        }
        let t = z + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * sum
    };

    Ok(Value::Float(OrderedFloat(result)))
}

fn compare_lt(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) < b.0)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(a.0 < *b as f64)),
        (Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a < b)),
        (Value::Uncertain { value: a, .. }, Value::Uncertain { value: b, .. }) =>
            compare_lt(*a.clone(), *b.clone()),
        (Value::Uncertain { value: a, .. }, _) =>
            compare_lt(*a.clone(), right.clone()),
        (_, Value::Uncertain { value: b, .. }) =>
            compare_lt(left.clone(), *b.clone()),
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
            Ok(Value::Tensor(Rc::new(Tensor::from_values(t.shape.clone(), scanned))))
        }
        _ => Err(EvalError::type_error("Tensor", &value)),
    }
}

fn len(value: Value) -> EvalResult<Value> { match value { Value::Tensor(t) => Ok(Value::Int(t.len() as i128)), Value::Tuple(vs) => Ok(Value::Int(vs.len() as i128)), _ => Err(EvalError::type_error("Tensor or Tuple", &value)) } }
fn shape(value: Value) -> EvalResult<Value> { match value { Value::Tensor(t) => Ok(Value::Tensor(Rc::new(Tensor::from_ints(t.shape.iter().map(|&d| d as i128).collect())))), _ => Err(EvalError::type_error("Tensor", &value)) } }
fn reverse(value: Value) -> EvalResult<Value> { match value { Value::Tensor(t) => { let mut data = t.to_vec(); data.reverse(); Ok(Value::Tensor(Rc::new(Tensor::from_values(t.shape.clone(), data)))) } Value::Tuple(mut vs) => { vs.reverse(); Ok(Value::Tuple(vs)) } _ => Err(EvalError::type_error("Tensor or Tuple", &value)) } }

fn concat(left: Value, right: Value) -> EvalResult<Value> {
    use crate::value::TensorData;
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.rank() != 1 || b.rank() != 1 { return Err(EvalError::not_implemented("concat for rank > 1")); }
            let new_len = a.len() + b.len();
            // Preserve type-specific tensor data when both sides match
            let data = match (&a.data, &b.data) {
                (TensorData::Char(ca), TensorData::Char(cb)) => {
                    let mut chars = ca.clone(); chars.extend(cb.iter()); TensorData::Char(chars)
                }
                (TensorData::Int(ia), TensorData::Int(ib)) => {
                    let mut ints = ia.clone(); ints.extend(ib.iter()); TensorData::Int(ints)
                }
                (TensorData::Float(fa), TensorData::Float(fb)) => {
                    let mut floats = fa.clone(); floats.extend(fb.iter()); TensorData::Float(floats)
                }
                (TensorData::Bool(ba), TensorData::Bool(bb)) => {
                    let mut bools = ba.clone(); bools.extend(bb.iter()); TensorData::Bool(bools)
                }
                _ => { let mut data = a.to_vec(); data.extend(b.iter()); TensorData::Generic(data) }
            };
            Ok(Value::Tensor(Rc::new(Tensor { shape: vec![new_len], data })))
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
            Ok(Value::Tensor(Rc::new(Tensor::from_values(a.shape.clone(), zipped))))
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
            Ok(Value::Tensor(Rc::new(Tensor::from_values(vec![m, p], result))))
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
            Ok(Value::Tensor(Rc::new(Tensor::from_values(vec![n, m], result))))
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

/// parseInt str: Parse a string as an integer
fn parse_int(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            if let Some(s) = t.to_string_value() {
                let s = s.trim();
                s.parse::<i128>()
                    .map(Value::Int)
                    .map_err(|_| EvalError::type_error_msg(format!("Cannot parse '{}' as integer", s)))
            } else {
                Err(EvalError::type_error("String", &Value::Tensor(t)))
            }
        }
        _ => Err(EvalError::type_error("String", &value)),
    }
}

/// parseFloat str: Parse a string as a float
fn parse_float(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            if let Some(s) = t.to_string_value() {
                let s = s.trim();
                s.parse::<f64>()
                    .map(|f| Value::Float(OrderedFloat(f)))
                    .map_err(|_| EvalError::type_error_msg(format!("Cannot parse '{}' as float", s)))
            } else {
                Err(EvalError::type_error("String", &Value::Tensor(t)))
            }
        }
        _ => Err(EvalError::type_error("String", &value)),
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
            Ok(Value::Tensor(Rc::new(Tensor::from_ints(data))))
        }
        _ => Err(EvalError::type_error("integer", &n)),
    }
}

/// range start end: Generate [start, start+1, ..., end-1]
fn range(start: Value, end: Value) -> EvalResult<Value> {
    match (&start, &end) {
        (Value::Int(s), Value::Int(e)) => {
            let data: Vec<i128> = (*s..*e).collect();
            Ok(Value::Tensor(Rc::new(Tensor::from_ints(data))))
        }
        _ => Err(EvalError::type_error_msg(format!(
            "range requires two integers, got {} and {}",
            start.type_name(),
            end.type_name()
        ))),
    }
}

/// toString: Convert any value to a string representation
fn to_string(value: Value) -> EvalResult<Value> {
    // Handle Char specially to avoid quotes (Display adds quotes for REPL)
    match &value {
        Value::Char(c) => Ok(Value::string(&c.to_string())),
        _ => Ok(Value::string(&format!("{}", value))),
    }
}

/// chars: Convert a string (char tensor) to an array of individual characters
fn chars(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            if t.to_string_value().is_some() {
                // Already a string, return as-is (it's already a char tensor)
                Ok(Value::Tensor(t))
            } else {
                Err(EvalError::type_error("String", &Value::Tensor(t)))
            }
        }
        _ => Err(EvalError::type_error("String", &value)),
    }
}

/// fromChars: Convert an array of characters to a string
fn from_chars(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            // If already a char tensor (string), return as-is
            if t.to_string_value().is_some() {
                return Ok(Value::Tensor(t));
            }
            // Otherwise try to collect Char values into a string
            let mut s = String::with_capacity(t.len());
            for v in t.iter() {
                match v {
                    Value::Char(c) => s.push(c),
                    _ => return Err(EvalError::type_error("Char", &v)),
                }
            }
            Ok(Value::string(&s))
        }
        _ => Err(EvalError::type_error("Tensor", &value)),
    }
}

/// strConcat: Concatenate two strings
fn str_concat(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            match (a.to_string_value(), b.to_string_value()) {
                (Some(s1), Some(s2)) => {
                    let combined = format!("{}{}", s1, s2);
                    Ok(Value::string(&combined))
                }
                _ => {
                    // Fall back to tensor concat if not strings
                    concat(left, right)
                }
            }
        }
        _ => Err(EvalError::type_error_msg(format!(
            "strConcat requires two strings, got {} and {}",
            left.type_name(),
            right.type_name()
        ))),
    }
}

/// take n arr: Take the first n elements from an array
fn take(n: Value, arr: Value) -> EvalResult<Value> {
    match (&n, &arr) {
        (Value::Int(count), Value::Tensor(t)) => {
            if t.rank() != 1 {
                return Err(EvalError::not_implemented("take for rank > 1"));
            }
            let count = (*count).max(0) as usize;
            let count = count.min(t.len());
            // Preserve string type when taking from strings
            if let Some(s) = t.to_string_value() {
                let taken: String = s.chars().take(count).collect();
                return Ok(Value::string(&taken));
            }
            let data: Vec<Value> = (0..count).map(|i| t.get_flat(i).unwrap()).collect();
            Ok(Value::Tensor(Rc::new(Tensor::from_values(vec![data.len()], data))))
        }
        (Value::Int(count), Value::Tuple(vs)) => {
            let count = (*count).max(0) as usize;
            let count = count.min(vs.len());
            Ok(Value::Tuple(vs[..count].to_vec()))
        }
        _ => Err(EvalError::type_error_msg(format!(
            "take requires (Int, Tensor/Tuple), got ({}, {})",
            n.type_name(),
            arr.type_name()
        ))),
    }
}

/// drop n arr: Drop the first n elements from an array
fn drop_fn(n: Value, arr: Value) -> EvalResult<Value> {
    match (&n, &arr) {
        (Value::Int(count), Value::Tensor(t)) => {
            if t.rank() != 1 {
                return Err(EvalError::not_implemented("drop for rank > 1"));
            }
            let count = (*count).max(0) as usize;
            let count = count.min(t.len());
            // Preserve string type when dropping from strings
            if let Some(s) = t.to_string_value() {
                let dropped: String = s.chars().skip(count).collect();
                return Ok(Value::string(&dropped));
            }
            let data: Vec<Value> = (count..t.len()).map(|i| t.get_flat(i).unwrap()).collect();
            Ok(Value::Tensor(Rc::new(Tensor::from_values(vec![data.len()], data))))
        }
        (Value::Int(count), Value::Tuple(vs)) => {
            let count = (*count).max(0) as usize;
            let count = count.min(vs.len());
            Ok(Value::Tuple(vs[count..].to_vec()))
        }
        _ => Err(EvalError::type_error_msg(format!(
            "drop requires (Int, Tensor/Tuple), got ({}, {})",
            n.type_name(),
            arr.type_name()
        ))),
    }
}

/// index arr idx: Get element at index
fn index(arr: Value, idx: Value) -> EvalResult<Value> {
    match (&arr, &idx) {
        (Value::Tensor(t), Value::Int(i)) => {
            let i = *i as usize;
            t.get_flat(i).ok_or_else(|| EvalError::IndexOutOfBounds { index: i, size: t.len() })
        }
        (Value::Tuple(vs), Value::Int(i)) => {
            let i = *i as usize;
            vs.get(i).cloned().ok_or_else(|| EvalError::IndexOutOfBounds { index: i, size: vs.len() })
        }
        _ => Err(EvalError::type_error_msg(format!(
            "index requires (Tensor/Tuple, Int), got ({}, {})",
            arr.type_name(),
            idx.type_name()
        ))),
    }
}

/// lines str: Split a string by newlines into an array of strings
fn lines(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            if let Some(s) = t.to_string_value() {
                let line_strs: Vec<Value> = s.lines().map(|l| Value::string(l)).collect();
                Ok(Value::Tensor(Rc::new(Tensor::from_values(vec![line_strs.len()], line_strs))))
            } else {
                Err(EvalError::type_error("String", &Value::Tensor(t)))
            }
        }
        _ => Err(EvalError::type_error("String", &value)),
    }
}

/// words str: Split a string by whitespace into an array of words
fn words(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            if let Some(s) = t.to_string_value() {
                let word_strs: Vec<Value> = s.split_whitespace().map(|w| Value::string(w)).collect();
                Ok(Value::Tensor(Rc::new(Tensor::from_values(vec![word_strs.len()], word_strs))))
            } else {
                Err(EvalError::type_error("String", &Value::Tensor(t)))
            }
        }
        _ => Err(EvalError::type_error("String", &value)),
    }
}

/// bytes str: Get the UTF-8 bytes of a string as an array of integers
fn bytes(value: Value) -> EvalResult<Value> {
    match value {
        Value::Tensor(t) => {
            if let Some(s) = t.to_string_value() {
                let byte_vals: Vec<i128> = s.as_bytes().iter().map(|&b| b as i128).collect();
                Ok(Value::Tensor(Rc::new(Tensor::from_ints(byte_vals))))
            } else {
                Err(EvalError::type_error("String", &Value::Tensor(t)))
            }
        }
        _ => Err(EvalError::type_error("String", &value)),
    }
}

/// strEq a b: Check if two strings are equal
fn str_eq(left: Value, right: Value) -> EvalResult<Value> {
    match (&left, &right) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            match (a.to_string_value(), b.to_string_value()) {
                (Some(s1), Some(s2)) => Ok(Value::Bool(s1 == s2)),
                _ => Err(EvalError::type_error_msg("strEq requires two strings")),
            }
        }
        _ => Err(EvalError::type_error_msg(format!(
            "strEq requires two strings, got {} and {}",
            left.type_name(),
            right.type_name()
        ))),
    }
}

/// startsWith str prefix: Check if a string starts with a prefix
fn starts_with(str_val: Value, prefix_val: Value) -> EvalResult<Value> {
    match (&str_val, &prefix_val) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            match (a.to_string_value(), b.to_string_value()) {
                (Some(s), Some(prefix)) => Ok(Value::Bool(s.starts_with(&prefix))),
                _ => Err(EvalError::type_error_msg("startsWith requires two strings")),
            }
        }
        _ => Err(EvalError::type_error_msg(format!(
            "startsWith requires two strings, got {} and {}",
            str_val.type_name(),
            prefix_val.type_name()
        ))),
    }
}

/// endsWith str suffix: Check if a string ends with a suffix
fn ends_with(str_val: Value, suffix_val: Value) -> EvalResult<Value> {
    match (&str_val, &suffix_val) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            match (a.to_string_value(), b.to_string_value()) {
                (Some(s), Some(suffix)) => Ok(Value::Bool(s.ends_with(&suffix))),
                _ => Err(EvalError::type_error_msg("endsWith requires two strings")),
            }
        }
        _ => Err(EvalError::type_error_msg(format!(
            "endsWith requires two strings, got {} and {}",
            str_val.type_name(),
            suffix_val.type_name()
        ))),
    }
}

/// contains str substr: Check if a string contains a substring
fn str_contains(str_val: Value, substr_val: Value) -> EvalResult<Value> {
    match (&str_val, &substr_val) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            match (a.to_string_value(), b.to_string_value()) {
                (Some(s), Some(substr)) => Ok(Value::Bool(s.contains(&substr))),
                _ => Err(EvalError::type_error_msg("contains requires two strings")),
            }
        }
        _ => Err(EvalError::type_error_msg(format!(
            "contains requires two strings, got {} and {}",
            str_val.type_name(),
            substr_val.type_name()
        ))),
    }
}
