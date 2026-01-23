//! Error types for MLIR emission

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MlirError {
    #[error("Unsupported type: {0}")]
    UnsupportedType(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("Invalid function: {0}")]
    InvalidFunction(String),

    #[error("Code generation error: {0}")]
    CodeGen(String),

    #[error("Verification error: {0}")]
    Verification(String),

    #[error("Pass error: {0}")]
    PassError(String),

    #[error("MIR error: {0}")]
    MirError(#[from] goth_mir::MirError),
}

pub type Result<T> = std::result::Result<T, MlirError>;
