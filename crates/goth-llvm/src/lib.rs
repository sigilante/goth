//! LLVM IR code generation for Goth
//!
//! This crate emits LLVM IR text format from MIR, which can then be
//! compiled to native code using clang or llc.

mod emit;
mod error;
mod runtime;

pub use emit::emit_program;
pub use error::{LlvmError, Result};
