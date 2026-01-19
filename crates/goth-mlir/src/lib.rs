//! MLIR code generation for Goth
//!
//! This crate lowers MIR (Mid-level IR) to MLIR (Multi-Level Intermediate Representation).
//!
//! # Architecture
//!
//! ```text
//! MIR → MLIR dialects:
//!   - func: Function definitions
//!   - arith: Arithmetic operations
//!   - scf: Structured control flow
//!   - tensor: Array operations
//!   - math: Mathematical functions
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use goth_mlir::emit_program;
//! use goth_mir::lower_expr;
//! use goth_ast::expr::Expr;
//!
//! let expr = /* ... */;
//! let mir_program = lower_expr(&expr)?;
//! let mlir_code = emit_program(&mir_program)?;
//! println!("{}", mlir_code);
//! ```

pub mod error;
pub mod emit;

pub use error::{MlirError, Result};
pub use emit::{emit_program, emit_function, emit_type};
