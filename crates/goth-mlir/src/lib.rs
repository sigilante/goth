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
//!   - cf: Unstructured control flow
//!   - scf: Structured control flow
//!   - tensor: Array operations
//!   - linalg: Linear algebra operations
//!   - memref: Memory reference operations
//!   - math: Mathematical functions
//!   - goth: Custom Goth dialect for domain-specific ops
//! ```
//!
//! # Compilation Pipeline
//!
//! ```text
//! MIR → MLIR (goth dialect)
//!     → Lower goth dialect to standard MLIR
//!     → Optimize (CSE, DCE, constant folding)
//!     → Bufferize (tensor → memref)
//!     → Lower to LLVM dialect
//!     → Generate LLVM IR
//! ```
//!
//! # Features
//!
//! - `melior`: Enable proper MLIR bindings via the melior crate (requires LLVM/MLIR)
//! - `text-emit`: Use text-based MLIR generation (default, no external dependencies)
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
//!
//! # Using the Pass Pipeline
//!
//! ```rust,ignore
//! use goth_mlir::passes::{PassManager, OptLevel, default_pipeline};
//!
//! let mlir_code = emit_program(&mir_program)?;
//! let pm = default_pipeline(OptLevel::O2);
//! let optimized = pm.run(&mlir_code)?;
//! ```

pub mod error;
pub mod context;
pub mod types;
pub mod dialects;
pub mod builder;
pub mod emit;
pub mod passes;

// Re-exports
pub use error::{MlirError, Result};
pub use context::TextMlirContext;
pub use types::type_to_mlir_string;
pub use builder::MlirBuilder;
pub use emit::{emit_program, emit_function, emit_type};

// Pass-related exports
pub use passes::{Pass, PassManager, OptLevel, default_pipeline, llvm_pipeline};
pub use passes::{bufferize_module, lower_goth_dialect, optimize_module, lower_to_llvm};

// LLVM dialect exports
pub use dialects::llvm::{LlvmBuilder, LlvmType, mlir_type_to_llvm};

#[cfg(feature = "melior")]
pub use context::GothMlirContext;
