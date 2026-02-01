//! Literal values in Goth

use serde::{Deserialize, Serialize};

/// Literal values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Integer literal (arbitrary precision in AST)
    Int(i128),

    /// Floating point literal
    Float(f64),

    /// Character literal (Unicode scalar)
    Char(char),

    /// String literal (sequence of Unicode scalars)
    String(Box<str>),

    /// Boolean true (⊤)
    True,

    /// Boolean false (⊥)
    False,

    /// Unit value (empty tuple)
    Unit,

    /// Imaginary-i literal (coefficient stored as f64)
    ImagI(f64),

    /// Imaginary-j literal (quaternion j component)
    ImagJ(f64),

    /// Imaginary-k literal (quaternion k component)
    ImagK(f64),
}

impl Literal {
    pub fn int(n: impl Into<i128>) -> Self {
        Literal::Int(n.into())
    }

    pub fn float(f: f64) -> Self {
        Literal::Float(f)
    }

    pub fn string(s: impl Into<Box<str>>) -> Self {
        Literal::String(s.into())
    }

    pub fn imag_i(f: f64) -> Self { Literal::ImagI(f) }
    pub fn imag_j(f: f64) -> Self { Literal::ImagJ(f) }
    pub fn imag_k(f: f64) -> Self { Literal::ImagK(f) }

    pub fn bool(b: bool) -> Self {
        if b { Literal::True } else { Literal::False }
    }

    /// Check if this literal is a boolean
    pub fn is_bool(&self) -> bool {
        matches!(self, Literal::True | Literal::False)
    }

    /// Convert to bool if this is a boolean literal
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Literal::True => Some(true),
            Literal::False => Some(false),
            _ => None,
        }
    }

    /// Check if this is a numeric literal
    pub fn is_numeric(&self) -> bool {
        matches!(self, Literal::Int(_) | Literal::Float(_) | Literal::ImagI(_) | Literal::ImagJ(_) | Literal::ImagK(_))
    }
}

impl From<i64> for Literal {
    fn from(n: i64) -> Self {
        Literal::Int(n as i128)
    }
}

impl From<f64> for Literal {
    fn from(f: f64) -> Self {
        Literal::Float(f)
    }
}

impl From<bool> for Literal {
    fn from(b: bool) -> Self {
        Literal::bool(b)
    }
}

impl From<&str> for Literal {
    fn from(s: &str) -> Self {
        Literal::String(s.into())
    }
}

impl From<char> for Literal {
    fn from(c: char) -> Self {
        Literal::Char(c)
    }
}
