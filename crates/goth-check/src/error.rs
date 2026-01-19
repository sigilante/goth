//! Type checking errors
//!
//! This module defines error types for the Goth type checker, with special
//! attention to shape-related errors since shape checking is a key feature.

use goth_ast::types::Type;
use goth_ast::shape::Shape;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum TypeError {
    #[error("Type mismatch: expected {expected}, found {found}")]
    Mismatch {
        expected: Type,
        found: Type,
    },

    #[error("Cannot infer type of lambda without annotation")]
    CannotInferLambda,

    #[error("Not a function type: {0}")]
    NotAFunction(Type),

    #[error("Not a tensor type: {0}")]
    NotATensor(Type),

    #[error("Unbound index: _{0}")]
    UnboundIndex(u32),

    #[error("Undefined name: {0}")]
    UndefinedName(String),

    #[error("Undefined type variable: {0}")]
    UndefinedTypeVar(String),

    // ============================================================
    // Shape Errors - THE key feature of Goth
    // ============================================================

    #[error("Shape mismatch: expected {expected}, found {found}")]
    ShapeMismatch {
        expected: Shape,
        found: Shape,
    },

    #[error("Rank mismatch: expected {expected} dimension(s), found {found}")]
    RankMismatch {
        expected: usize,
        found: usize,
    },

    #[error("Dimension mismatch at position {position}: expected {expected}, found {found}")]
    DimMismatch {
        position: usize,
        expected: String,
        found: String,
    },

    #[error("Shape error in {operation}: {message}")]
    ShapeError {
        operation: String,
        message: String,
    },

    #[error("Matrix multiplication shape error: left columns ({left_cols}) must match right rows ({right_rows})")]
    MatmulShapeError {
        left_cols: String,
        right_rows: String,
    },

    #[error("Infinite shape: dimension variable {var} would be infinite")]
    InfiniteShape {
        var: String,
    },

    #[error("Occurs check failed: {var} occurs in {ty}")]
    InfiniteType {
        var: String,
        ty: Type,
    },

    #[error("Cannot unify types: {t1} and {t2}")]
    UnificationFailure {
        t1: Type,
        t2: Type,
    },

    #[error("Pattern type mismatch: pattern expects {expected}, but value has type {found}")]
    PatternMismatch {
        expected: String,
        found: Type,
    },

    #[error("Tuple has {found} elements, but pattern expects {expected}")]
    TupleArityMismatch {
        expected: usize,
        found: usize,
    },

    #[error("Array has {found} elements, but pattern expects {expected}")]
    ArrayArityMismatch {
        expected: usize,
        found: usize,
    },

    #[error("Match is not exhaustive")]
    NonExhaustiveMatch,

    #[error("Binary operator {op} cannot be applied to {left} and {right}")]
    InvalidBinOp {
        op: String,
        left: Type,
        right: Type,
    },

    #[error("Unary operator {op} cannot be applied to {operand}")]
    InvalidUnaryOp {
        op: String,
        operand: Type,
    },

    #[error("Field {field} not found in type {ty}")]
    FieldNotFound {
        field: String,
        ty: Type,
    },

    #[error("Cannot index into non-tensor type: {0}")]
    CannotIndex(Type),

    #[error("Wrong number of indices: expected {expected}, found {found}")]
    WrongIndexCount {
        expected: usize,
        found: usize,
    },
}

pub type TypeResult<T> = Result<T, TypeError>;