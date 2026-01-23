//! MLIR dialect operations for Goth
//!
//! This module provides builders for operations from various MLIR dialects:
//!
//! - `arith`: Arithmetic operations (add, sub, mul, div, cmp)
//! - `func`: Function definitions and calls
//! - `cf`: Control flow (branch, cond_br, switch)
//! - `scf`: Structured control flow (if, for, while)
//! - `tensor`: Tensor operations
//! - `linalg`: Linear algebra operations (for tensor computations) - TODO
//! - `goth`: Custom Goth dialect for domain-specific operations

pub mod arith;
pub mod func;
pub mod cf;
pub mod scf;
pub mod tensor;
pub mod goth;

// Re-export commonly used items
pub use arith::*;
pub use func::*;
pub use scf::*;
