//! Runtime values for Goth

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use ordered_float::OrderedFloat;

/// Runtime value
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i128),
    Float(OrderedFloat<f64>),
    Bool(bool),
    Char(char),
    Unit,
    Tensor(Tensor),
    Tuple(Vec<Value>),
    Record(Rc<HashMap<String, Value>>),
    Variant { tag: String, payload: Option<Box<Value>> },
    Closure(Closure),
    Primitive(PrimFn),
    Partial { func: Box<Value>, args: Vec<Value>, remaining: usize },
    Thunk(Thunk),
    Ref(Rc<RefCell<Value>>),
    Uncertain { value: Box<Value>, uncertainty: Box<Value> },
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    Int(Vec<i128>),
    Float(Vec<OrderedFloat<f64>>),
    Bool(Vec<bool>),
    Char(Vec<char>),
    Generic(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub arity: u32,
    pub body: goth_ast::expr::Expr,
    pub env: Env,
    pub preconditions: Vec<goth_ast::expr::Expr>,
    pub postconditions: Vec<goth_ast::expr::Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimFn {
    Add, Sub, Mul, Div, Mod, Neg, Abs,
    Eq, Neq, Lt, Gt, Leq, Geq,
    And, Or, Not,
    Exp, Ln, Sqrt, Sin, Cos, Tan, Pow, Floor, Ceil, Round,
    Map, Filter, Fold, Scan, Zip, Concat, Reverse,
    Sum, Prod, Len, Shape, Reshape, Transpose,
    Index, Slice, Take, Drop,
    Iota, Range,  // Sequence generation
    MatMul, Dot, Outer, Inner, Norm,
    Chars, ToString, StrConcat,
    Print, ReadLine, ReadFile, WriteFile,
    ToInt, ToFloat, ToBool, ToChar,
}

#[derive(Debug, Clone)]
pub struct Thunk {
    pub expr: goth_ast::expr::Expr,
    pub env: Env,
    pub cached: Option<Rc<Value>>,
}

#[derive(Debug, Clone, Default)]
pub struct Env {
    pub values: Vec<Value>,
    globals: Rc<RefCell<HashMap<String, Value>>>,
}

impl Value {
    pub fn int(n: impl Into<i128>) -> Self { Value::Int(n.into()) }
    pub fn float(f: f64) -> Self { Value::Float(OrderedFloat(f)) }
    pub fn bool(b: bool) -> Self { Value::Bool(b) }
    pub fn char(c: char) -> Self { Value::Char(c) }
    pub fn unit() -> Self { Value::Unit }
    pub fn string(s: &str) -> Self { Value::Tensor(Tensor::from_string(s)) }
    pub fn tuple(values: Vec<Value>) -> Self {
        if values.is_empty() { Value::Unit } else { Value::Tuple(values) }
    }
    pub fn variant(tag: impl Into<String>, payload: Option<Value>) -> Self {
        Value::Variant { tag: tag.into(), payload: payload.map(Box::new) }
    }
    pub fn error(msg: impl Into<String>) -> Self { Value::Error(msg.into()) }
    pub fn closure(arity: u32, body: goth_ast::expr::Expr, env: Env) -> Self {
        Value::Closure(Closure { arity, body, env, preconditions: vec![], postconditions: vec![] })
    }
    pub fn closure_with_contracts(arity: u32, body: goth_ast::expr::Expr, env: Env, preconditions: Vec<goth_ast::expr::Expr>, postconditions: Vec<goth_ast::expr::Expr>) -> Self {
        Value::Closure(Closure { arity, body, env, preconditions, postconditions })
    }
    pub fn primitive(prim: PrimFn) -> Self { Value::Primitive(prim) }

    pub fn is_int(&self) -> bool { matches!(self, Value::Int(_)) }
    pub fn is_float(&self) -> bool { matches!(self, Value::Float(_)) }
    pub fn is_numeric(&self) -> bool { matches!(self, Value::Int(_) | Value::Float(_)) }
    pub fn is_bool(&self) -> bool { matches!(self, Value::Bool(_)) }
    pub fn is_tensor(&self) -> bool { matches!(self, Value::Tensor(_)) }
    pub fn is_callable(&self) -> bool { matches!(self, Value::Closure(_) | Value::Primitive(_) | Value::Partial { .. }) }
    pub fn is_error(&self) -> bool { matches!(self, Value::Error(_)) }

    pub fn as_int(&self) -> Option<i128> { match self { Value::Int(n) => Some(*n), _ => None } }
    pub fn as_float(&self) -> Option<f64> { match self { Value::Float(f) => Some(f.0), Value::Int(n) => Some(*n as f64), _ => None } }
    pub fn as_bool(&self) -> Option<bool> { match self { Value::Bool(b) => Some(*b), _ => None } }
    pub fn as_char(&self) -> Option<char> { match self { Value::Char(c) => Some(*c), _ => None } }
    pub fn as_tensor(&self) -> Option<&Tensor> { match self { Value::Tensor(t) => Some(t), _ => None } }
    pub fn as_tuple(&self) -> Option<&[Value]> { match self { Value::Tuple(vs) => Some(vs), Value::Unit => Some(&[]), _ => None } }
    pub fn coerce_float(&self) -> Option<f64> { match self { Value::Float(f) => Some(f.0), Value::Int(n) => Some(*n as f64), _ => None } }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int", Value::Float(_) => "Float", Value::Bool(_) => "Bool",
            Value::Char(_) => "Char", Value::Unit => "Unit", Value::Tensor(_) => "Tensor",
            Value::Tuple(_) => "Tuple", Value::Record(_) => "Record", Value::Variant { .. } => "Variant",
            Value::Closure(_) => "Closure", Value::Primitive(_) => "Primitive",
            Value::Partial { .. } => "Partial", Value::Thunk(_) => "Thunk",
            Value::Ref(_) => "Ref", Value::Uncertain { .. } => "Uncertain", Value::Error(_) => "Error",
        }
    }

    pub fn deep_eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            (Value::Tensor(a), Value::Tensor(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.deep_eq(y)),
            (Value::Variant { tag: t1, payload: p1 }, Value::Variant { tag: t2, payload: p2 }) => {
                t1 == t2 && match (p1, p2) { (None, None) => true, (Some(a), Some(b)) => a.deep_eq(b), _ => false }
            }
            (Value::Uncertain { value: v1, uncertainty: u1 }, Value::Uncertain { value: v2, uncertainty: u2 }) => {
                v1.deep_eq(v2) && u1.deep_eq(u2)
            }
            _ => false,
        }
    }
}

impl Tensor {
    pub fn scalar(value: Value) -> Self {
        match value {
            Value::Int(n) => Tensor { shape: vec![], data: TensorData::Int(vec![n]) },
            Value::Float(f) => Tensor { shape: vec![], data: TensorData::Float(vec![f]) },
            Value::Bool(b) => Tensor { shape: vec![], data: TensorData::Bool(vec![b]) },
            Value::Char(c) => Tensor { shape: vec![], data: TensorData::Char(vec![c]) },
            _ => Tensor { shape: vec![], data: TensorData::Generic(vec![value]) },
        }
    }
    pub fn from_ints(data: Vec<i128>) -> Self { let len = data.len(); Tensor { shape: vec![len], data: TensorData::Int(data) } }
    pub fn from_floats(data: Vec<f64>) -> Self { let len = data.len(); Tensor { shape: vec![len], data: TensorData::Float(data.into_iter().map(OrderedFloat).collect()) } }
    pub fn from_bools(data: Vec<bool>) -> Self { let len = data.len(); Tensor { shape: vec![len], data: TensorData::Bool(data) } }
    pub fn from_string(s: &str) -> Self { let chars: Vec<char> = s.chars().collect(); let len = chars.len(); Tensor { shape: vec![len], data: TensorData::Char(chars) } }
    pub fn from_values(shape: Vec<usize>, data: Vec<Value>) -> Self { Tensor { shape, data: TensorData::Generic(data) } }
    pub fn from_matrix(rows: Vec<Vec<f64>>) -> Self {
        let m = rows.len(); let n = rows.first().map(|r| r.len()).unwrap_or(0);
        let data: Vec<OrderedFloat<f64>> = rows.into_iter().flat_map(|row| row.into_iter().map(OrderedFloat)).collect();
        Tensor { shape: vec![m, n], data: TensorData::Float(data) }
    }
    pub fn zeros(shape: Vec<usize>) -> Self { let size: usize = shape.iter().product(); Tensor { shape, data: TensorData::Float(vec![OrderedFloat(0.0); size]) } }
    pub fn ones(shape: Vec<usize>) -> Self { let size: usize = shape.iter().product(); Tensor { shape, data: TensorData::Float(vec![OrderedFloat(1.0); size]) } }
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn len(&self) -> usize { self.shape.iter().product() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn get_flat(&self, idx: usize) -> Option<Value> {
        if idx >= self.len() { return None; }
        Some(match &self.data {
            TensorData::Int(v) => Value::Int(v[idx]),
            TensorData::Float(v) => Value::Float(v[idx]),
            TensorData::Bool(v) => Value::Bool(v[idx]),
            TensorData::Char(v) => Value::Char(v[idx]),
            TensorData::Generic(v) => v[idx].clone(),
        })
    }

    pub fn get(&self, indices: &[usize]) -> Option<Value> {
        if indices.len() != self.shape.len() { return None; }
        let flat_idx = self.flatten_index(indices)?;
        self.get_flat(flat_idx)
    }

    fn flatten_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() { return None; }
        let mut flat = 0; let mut stride = 1;
        for (idx, &dim) in indices.iter().zip(&self.shape).rev() {
            if *idx >= dim { return None; }
            flat += idx * stride; stride *= dim;
        }
        Some(flat)
    }

    pub fn iter(&self) -> impl Iterator<Item = Value> + '_ { (0..self.len()).map(|i| self.get_flat(i).unwrap()) }
    pub fn to_vec(&self) -> Vec<Value> { self.iter().collect() }
    pub fn to_string_value(&self) -> Option<String> { match &self.data { TensorData::Char(chars) => Some(chars.iter().collect()), _ => None } }

    pub fn map<F>(&self, f: F) -> Tensor where F: Fn(Value) -> Value {
        let mapped: Vec<Value> = self.iter().map(f).collect();
        Tensor::from_values(self.shape.clone(), mapped)
    }

    pub fn zip_with<F>(&self, other: &Tensor, f: F) -> Option<Tensor> where F: Fn(Value, Value) -> Value {
        if self.shape != other.shape { return None; }
        let zipped: Vec<Value> = self.iter().zip(other.iter()).map(|(a, b)| f(a, b)).collect();
        Some(Tensor::from_values(self.shape.clone(), zipped))
    }

    pub fn fold<F>(&self, init: Value, f: F) -> Value where F: Fn(Value, Value) -> Value { self.iter().fold(init, f) }

    pub fn sum(&self) -> Value {
        match &self.data {
            TensorData::Int(v) => Value::Int(v.iter().copied().sum()),
            TensorData::Float(v) => Value::Float(OrderedFloat(v.iter().map(|x| x.0).sum())),
            TensorData::Bool(v) => Value::Int(v.iter().filter(|&&b| b).count() as i128),
            TensorData::Generic(v) => {
                // Try to sum generic tensor if it contains numeric values
                let mut sum = Value::Int(0);
                for val in v {
                    if val.is_numeric() {
                        // Use the add function from prim module
                        sum = match (sum, val.clone()) {
                            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                            (Value::Float(a), Value::Float(b)) => Value::Float(OrderedFloat(a.0 + b.0)),
                            (Value::Int(a), Value::Float(b)) => Value::Float(OrderedFloat(a as f64 + b.0)),
                            (Value::Float(a), Value::Int(b)) => Value::Float(OrderedFloat(a.0 + b as f64)),
                            _ => return Value::Error("Cannot sum non-numeric tensor".into()),
                        };
                    } else {
                        return Value::Error("Cannot sum non-numeric tensor".into());
                    }
                }
                sum
            }
            _ => Value::Error("Cannot sum non-numeric tensor".into()),
        }
    }

    pub fn product(&self) -> Value {
        match &self.data {
            TensorData::Int(v) => Value::Int(v.iter().copied().product()),
            TensorData::Float(v) => Value::Float(OrderedFloat(v.iter().map(|x| x.0).product())),
            _ => Value::Error("Cannot multiply non-numeric tensor".into()),
        }
    }
}

impl Env {
    pub fn new() -> Self { Env { values: Vec::new(), globals: Rc::new(RefCell::new(HashMap::new())) } }
    pub fn with_globals(globals: Rc<RefCell<HashMap<String, Value>>>) -> Self { Env { values: Vec::new(), globals } }
    pub fn push(&mut self, value: Value) { self.values.push(value); }
    pub fn push_many(&mut self, values: impl IntoIterator<Item = Value>) { self.values.extend(values); }
    pub fn pop(&mut self) -> Option<Value> { self.values.pop() }
    pub fn pop_n(&mut self, n: usize) { for _ in 0..n { self.values.pop(); } }
    pub fn get(&self, idx: u32) -> Option<&Value> { let len = self.values.len(); if (idx as usize) < len { Some(&self.values[len - 1 - idx as usize]) } else { None } }
    pub fn get_global(&self, name: &str) -> Option<Value> { self.globals.borrow().get(name).cloned() }
    pub fn define_global(&self, name: impl Into<String>, value: Value) { self.globals.borrow_mut().insert(name.into(), value); }
    pub fn depth(&self) -> usize { self.values.len() }
    pub fn capture(&self) -> Self { Env { values: self.values.clone(), globals: Rc::clone(&self.globals) } }
    pub fn extend(&self, other: &Env) -> Self { let mut values = self.values.clone(); values.extend(other.values.iter().cloned()); Env { values, globals: Rc::clone(&self.globals) } }
}

impl PartialEq for Closure { fn eq(&self, other: &Self) -> bool { self.arity == other.arity && self.body == other.body } }
impl PartialEq for Thunk { fn eq(&self, other: &Self) -> bool { self.expr == other.expr } }

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(x) => write!(f, "{}", x.0),
            Value::Bool(true) => write!(f, "⊤"),
            Value::Bool(false) => write!(f, "⊥"),
            Value::Char(c) => write!(f, "'{}'", c),
            Value::Unit => write!(f, "⟨⟩"),
            Value::Tensor(t) => write!(f, "{}", t),
            Value::Tuple(vs) => { write!(f, "⟨")?; for (i, v) in vs.iter().enumerate() { if i > 0 { write!(f, ", ")?; } write!(f, "{}", v)?; } write!(f, "⟩") }
            Value::Record(fields) => { write!(f, "⟨")?; for (i, (k, v)) in fields.iter().enumerate() { if i > 0 { write!(f, ", ")?; } write!(f, "{}: {}", k, v)?; } write!(f, "⟩") }
            Value::Variant { tag, payload } => { write!(f, "{}", tag)?; if let Some(p) = payload { write!(f, " {}", p)?; } Ok(()) }
            Value::Closure(c) => write!(f, "<closure/{}>", c.arity),
            Value::Primitive(p) => write!(f, "<prim:{:?}>", p),
            Value::Partial { remaining, .. } => write!(f, "<partial/{}>", remaining),
            Value::Thunk(_) => write!(f, "<thunk>"),
            Value::Ref(_) => write!(f, "<ref>"),
            Value::Uncertain { value, uncertainty } => write!(f, "{}±{}", value, uncertainty),
            Value::Error(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(s) = self.to_string_value() { return write!(f, "\"{}\"", s); }
        if self.shape.is_empty() { if let Some(v) = self.get_flat(0) { return write!(f, "{}", v); } }
        if self.shape.len() == 1 { write!(f, "[")?; for (i, v) in self.iter().enumerate() { if i > 0 { write!(f, " ")?; } write!(f, "{}", v)?; } return write!(f, "]"); }
        write!(f, "<tensor {:?}>", self.shape)
    }
}
