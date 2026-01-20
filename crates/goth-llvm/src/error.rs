//! Error types for LLVM IR emission

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlvmError {
    #[error("Unsupported type: {0}")]
    UnsupportedType(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("Code generation error: {0}")]
    CodeGen(String),

    #[error("Undefined local: {0}")]
    UndefinedLocal(String),
}

pub type Result<T> = std::result::Result<T, LlvmError>;
