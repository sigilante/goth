//! MLIR dialect operations for Goth
//!
//! This module provides builders for operations from various MLIR dialects:
//!
//! - `arith`: Arithmetic operations (add, sub, mul, div, cmp)
//! - `func`: Function definitions and calls
//! - `cf`: Control flow (branch, cond_br, switch)
//! - `scf`: Structured control flow (if, for, while)
//! - `tensor`: Tensor operations
//! - `linalg`: Linear algebra operations (generic, reduce, matmul, etc.)
//! - `memref`: Memory reference operations (alloc, load, store, etc.)
//! - `goth`: Custom Goth dialect for domain-specific operations
//! - `llvm`: LLVM dialect for final lowering to LLVM IR

pub mod arith;
pub mod func;
pub mod cf;
pub mod scf;
pub mod tensor;
pub mod linalg;
pub mod memref;
pub mod goth;
pub mod llvm;

// Re-export commonly used items
pub use arith::*;
pub use func::*;
pub use scf::*;
pub use linalg::{AffineMap, IteratorType, GenericBuilder};
pub use memref::{MemRefBuilder, DimSize, type_to_memref_string};
pub use llvm::{LlvmBuilder, LlvmType, IcmpPredicate, FcmpPredicate, mlir_type_to_llvm};
