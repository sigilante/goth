//! Error handling for Goth evaluation

use thiserror::Error;
use crate::value::Value;

#[derive(Error, Debug, Clone)]
pub enum EvalError {
    #[error("Unbound variable: index {0}")]
    UnboundIndex(u32),
    #[error("Undefined name: {0}")]
    UndefinedName(String),
    #[error("Type error: expected {expected}, got {got}")]
    TypeError { expected: &'static str, got: &'static str },
    #[error("Type error: {0}")]
    TypeErrorMsg(String),
    #[error("Arity mismatch: expected {expected}, got {got}")]
    ArityMismatch { expected: usize, got: usize },
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Index out of bounds: {index} for size {size}")]
    IndexOutOfBounds { index: usize, size: usize },
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Pattern match failed")]
    MatchFailed,
    #[error("Non-exhaustive pattern match")]
    NonExhaustiveMatch,
    #[error("Assertion failed: {0}")]
    AssertionFailed(String),
    #[error("Precondition violated: {0}")]
    PreconditionViolated(String),
    #[error("Postcondition violated: {0}")]
    PostconditionViolated(String),
    #[error("Effect not allowed: {0}")]
    EffectNotAllowed(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Arithmetic overflow: {0}")]
    Overflow(String),
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("User error: {0}")]
    UserError(String),
}

pub type EvalResult<T> = Result<T, EvalError>;

impl EvalError {
    pub fn type_error(expected: &'static str, got: &Value) -> Self {
        EvalError::TypeError { expected, got: got.type_name() }
    }
    pub fn type_error_msg(msg: impl Into<String>) -> Self { EvalError::TypeErrorMsg(msg.into()) }
    pub fn shape_mismatch(msg: impl Into<String>) -> Self { EvalError::ShapeMismatch(msg.into()) }
    pub fn not_implemented(what: impl Into<String>) -> Self { EvalError::NotImplemented(what.into()) }
    pub fn internal(msg: impl Into<String>) -> Self { EvalError::Internal(msg.into()) }
    pub fn io_error(msg: impl Into<String>) -> Self { EvalError::IoError(msg.into()) }
}

pub trait OptionExt<T> {
    fn ok_or_unbound(self, idx: u32) -> EvalResult<T>;
    fn ok_or_undefined(self, name: &str) -> EvalResult<T>;
    fn ok_or_type(self, expected: &'static str, got: &Value) -> EvalResult<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_unbound(self, idx: u32) -> EvalResult<T> { self.ok_or(EvalError::UnboundIndex(idx)) }
    fn ok_or_undefined(self, name: &str) -> EvalResult<T> { self.ok_or_else(|| EvalError::UndefinedName(name.to_string())) }
    fn ok_or_type(self, expected: &'static str, got: &Value) -> EvalResult<T> { self.ok_or_else(|| EvalError::type_error(expected, got)) }
}
