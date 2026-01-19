//! Operators in Goth
//!
//! Both built-in operators and user-defined operator metadata.

use serde::{Deserialize, Serialize};

/// Binary operators
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    // Arithmetic
    Add,      // +
    Sub,      // -
    Mul,      // ×
    Div,      // /
    Pow,      // ^
    Mod,      // %
    PlusMinus, // ± (uncertain value)

    // Comparison
    Eq,       // =
    Neq,      // ≠
    Lt,       // <
    Gt,       // >
    Leq,      // ≤
    Geq,      // ≥

    // Logical
    And,      // ∧
    Or,       // ∨

    // Composition
    Compose,  // ∘

    // Array operations
    Map,      // ↦
    Filter,   // ▸
    Bind,     // ⤇
    ZipWith,  // ⊗
    Concat,   // ⊕

    // User-defined (by name)
    Custom(Box<str>),
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,      // - (numeric negation)
    Not,      // ¬ (logical not)
    Sum,      // Σ (fold with +)
    Prod,     // Π (fold with ×)
    Scan,     // ⍀ (prefix fold)
    Sqrt,     // √ (square root)
    Floor,    // ⌊⌋ (floor)
    Ceil,     // ⌈⌉ (ceiling)
    Gamma,    // Γ (gamma function)
    Ln,       // ln (natural log)
    Exp,      // exp (e^x)
    Sin,      // sin
    Cos,      // cos
    Abs,      // |x| (absolute value)
}

/// Operator associativity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Assoc {
    Left,
    Right,
    None,
}

/// Operator metadata for user-defined operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpMeta {
    pub name: Box<str>,
    pub glyph: Box<str>,      // Unicode representation
    pub ascii: Box<str>,      // ASCII fallback
    pub assoc: Assoc,
    pub precedence: u8,       // 1-12, higher binds tighter
}

impl BinOp {
    /// Get the Unicode glyph for this operator
    pub fn glyph(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "×",
            BinOp::Div => "/",
            BinOp::Pow => "^",
            BinOp::Mod => "%",
            BinOp::PlusMinus => "±",
            BinOp::Eq => "=",
            BinOp::Neq => "≠",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Leq => "≤",
            BinOp::Geq => "≥",
            BinOp::And => "∧",
            BinOp::Or => "∨",
            BinOp::Compose => "∘",
            BinOp::Map => "↦",
            BinOp::Filter => "▸",
            BinOp::Bind => "⤇",
            BinOp::ZipWith => "⊗",
            BinOp::Concat => "⊕",
            BinOp::Custom(_) => "?",
        }
    }

    /// Get the ASCII fallback for this operator
    pub fn ascii(&self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "^",
            BinOp::Mod => "%",
            BinOp::PlusMinus => "+-",
            BinOp::Eq => "==",
            BinOp::Neq => "/=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Leq => "<=",
            BinOp::Geq => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::Compose => ".:",
            BinOp::Map => "-:",
            BinOp::Filter => "|>",
            BinOp::Bind => "=>",
            BinOp::ZipWith => "*:",
            BinOp::Concat => "+:",
            BinOp::Custom(_) => "??",
        }
    }

    /// Get operator precedence (higher = tighter binding)
    pub fn precedence(&self) -> u8 {
        match self {
            BinOp::Pow => 10,
            BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::ZipWith => 9,
            BinOp::Add | BinOp::Sub | BinOp::PlusMinus | BinOp::Concat => 8,
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Leq | BinOp::Geq => 7,
            BinOp::And => 6,
            BinOp::Or => 5,
            BinOp::Map | BinOp::Filter | BinOp::Bind | BinOp::Compose => 4,
            BinOp::Custom(_) => 3,
        }
    }

    /// Get operator associativity
    pub fn assoc(&self) -> Assoc {
        match self {
            BinOp::Pow | BinOp::Compose | BinOp::Map | BinOp::Filter | BinOp::Bind => Assoc::Right,
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Leq | BinOp::Geq => Assoc::None,
            _ => Assoc::Left,
        }
    }
}

impl UnaryOp {
    pub fn glyph(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "¬",
            UnaryOp::Sum => "Σ",
            UnaryOp::Prod => "Π",
            UnaryOp::Scan => "⍀",
            UnaryOp::Sqrt => "√",
            UnaryOp::Floor => "⌊⌋",
            UnaryOp::Ceil => "⌈⌉",
            UnaryOp::Gamma => "Γ",
            UnaryOp::Ln => "ln",
            UnaryOp::Exp => "exp",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Abs => "abs",
        }
    }

    pub fn ascii(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::Sum => "+/",
            UnaryOp::Prod => "*/",
            UnaryOp::Scan => "\\/",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Gamma => "gamma",
            UnaryOp::Ln => "ln",
            UnaryOp::Exp => "exp",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Abs => "abs",
        }
    }
}