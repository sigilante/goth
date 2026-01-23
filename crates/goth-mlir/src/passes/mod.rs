//! MLIR passes for Goth compilation
//!
//! This module provides transformation passes for the Goth MLIR backend:
//!
//! - `bufferize`: Convert tensor types to memref types for memory management
//! - `lower_goth`: Lower custom Goth dialect operations to standard MLIR dialects
//! - `optimize`: Run optimization passes (canonicalization, CSE, DCE, etc.)
//!
//! # Compilation Pipeline
//!
//! The typical compilation pipeline runs passes in this order:
//!
//! 1. **Lower Goth Dialect** (`lower_goth`)
//!    - `goth.iota` → `linalg.generic` with index computation
//!    - `goth.map` → `linalg.generic` with elementwise computation
//!    - `goth.reduce_*` → `linalg.reduce` with appropriate combiner
//!    - `goth.filter` → `scf.if` + `tensor.insert`
//!    - `goth.zip` → `linalg.generic` with paired iteration
//!
//! 2. **Optimize** (`optimize`)
//!    - Canonicalization (normalize patterns)
//!    - Common subexpression elimination
//!    - Dead code elimination
//!    - Loop fusion (for tensor operations)
//!
//! 3. **Bufferize** (`bufferize`)
//!    - Convert tensor types to memref types
//!    - Insert alloc/dealloc operations
//!    - Handle tensor copies
//!
//! 4. **Lower to LLVM** (`to_llvm`)
//!    - Convert all dialects to LLVM dialect
//!    - Generate LLVM IR
//!
//! # Example
//!
//! ```rust,ignore
//! use goth_mlir::passes::{PassManager, OptLevel};
//!
//! let mut pm = PassManager::new();
//! pm.add_pass(passes::lower_goth());
//! pm.add_pass(passes::optimize(OptLevel::O2));
//! pm.add_pass(passes::bufferize());
//! pm.add_pass(passes::to_llvm());
//!
//! let result = pm.run(&mut module)?;
//! ```

pub mod bufferize;
pub mod lower_goth;
pub mod optimize;
pub mod lower_llvm;

// Re-exports
pub use bufferize::{BufferizePass, BufferizeOptions, bufferize_module};
pub use lower_goth::{LowerGothPass, lower_goth_dialect};
pub use optimize::{OptimizePass, OptLevel, optimize_module};
pub use lower_llvm::{LowerLlvmPass, lower_to_llvm};

use crate::error::{MlirError, Result};

/// A pass that transforms MLIR IR
pub trait Pass {
    /// Get the name of this pass
    fn name(&self) -> &'static str;

    /// Get a description of what this pass does
    fn description(&self) -> &'static str;

    /// Run the pass on MLIR text, returning transformed text
    fn run(&self, mlir: &str) -> Result<String>;

    /// Check if this pass should run given the current optimization level
    fn should_run(&self, _level: OptLevel) -> bool {
        true
    }
}

/// Pass manager for running a sequence of passes
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    opt_level: OptLevel,
    verify_after_each: bool,
    print_after_each: bool,
}

impl PassManager {
    /// Create a new pass manager
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            opt_level: OptLevel::O1,
            verify_after_each: true,
            print_after_each: false,
        }
    }

    /// Create a pass manager with a specific optimization level
    pub fn with_opt_level(level: OptLevel) -> Self {
        Self {
            passes: Vec::new(),
            opt_level: level,
            verify_after_each: true,
            print_after_each: false,
        }
    }

    /// Add a pass to the pipeline
    pub fn add_pass<P: Pass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Set whether to verify after each pass
    pub fn set_verify_after_each(&mut self, verify: bool) {
        self.verify_after_each = verify;
    }

    /// Set whether to print IR after each pass
    pub fn set_print_after_each(&mut self, print: bool) {
        self.print_after_each = print;
    }

    /// Get the current optimization level
    pub fn opt_level(&self) -> OptLevel {
        self.opt_level
    }

    /// Run all passes on the given MLIR text
    pub fn run(&self, mlir: &str) -> Result<String> {
        let mut current = mlir.to_string();

        for pass in &self.passes {
            if !pass.should_run(self.opt_level) {
                continue;
            }

            current = pass.run(&current)?;

            if self.print_after_each {
                eprintln!("=== After {} ===\n{}", pass.name(), current);
            }

            if self.verify_after_each {
                self.verify(&current)?;
            }
        }

        Ok(current)
    }

    /// Verify that MLIR text is well-formed (basic syntax check)
    fn verify(&self, mlir: &str) -> Result<()> {
        // Basic verification: check for matching braces
        let open_braces = mlir.chars().filter(|&c| c == '{').count();
        let close_braces = mlir.chars().filter(|&c| c == '}').count();

        if open_braces != close_braces {
            return Err(MlirError::Verification(format!(
                "Mismatched braces: {} open, {} close",
                open_braces, close_braces
            )));
        }

        // Check for matching parentheses
        let open_parens = mlir.chars().filter(|&c| c == '(').count();
        let close_parens = mlir.chars().filter(|&c| c == ')').count();

        if open_parens != close_parens {
            return Err(MlirError::Verification(format!(
                "Mismatched parentheses: {} open, {} close",
                open_parens, close_parens
            )));
        }

        Ok(())
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a default compilation pipeline
pub fn default_pipeline(level: OptLevel) -> PassManager {
    let mut pm = PassManager::with_opt_level(level);

    // 1. Lower Goth dialect to standard MLIR
    pm.add_pass(LowerGothPass::new());

    // 2. Optimize
    pm.add_pass(OptimizePass::new(level));

    // 3. Bufferize (tensor -> memref)
    pm.add_pass(BufferizePass::new());

    pm
}

/// Create a pipeline for lowering to LLVM
pub fn llvm_pipeline(level: OptLevel) -> PassManager {
    let mut pm = default_pipeline(level);

    // 4. Lower to LLVM dialect
    pm.add_pass(LowerLlvmPass::new());

    pm
}

/// Statistics from running a pass
#[derive(Debug, Default, Clone)]
pub struct PassStatistics {
    /// Number of operations before the pass
    pub ops_before: usize,
    /// Number of operations after the pass
    pub ops_after: usize,
    /// Number of operations eliminated
    pub ops_eliminated: usize,
    /// Number of operations created
    pub ops_created: usize,
}

impl PassStatistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate statistics from before/after IR
    pub fn from_ir(before: &str, after: &str) -> Self {
        let ops_before = Self::count_ops(before);
        let ops_after = Self::count_ops(after);

        Self {
            ops_before,
            ops_after,
            ops_eliminated: ops_before.saturating_sub(ops_after),
            ops_created: ops_after.saturating_sub(ops_before),
        }
    }

    /// Count approximate number of operations in MLIR text
    fn count_ops(mlir: &str) -> usize {
        // Count lines that look like operations (contain " = " or are terminators)
        mlir.lines()
            .filter(|line| {
                let trimmed = line.trim();
                (trimmed.contains(" = ") && !trimmed.starts_with("//"))
                    || trimmed.starts_with("return ")
                    || trimmed.starts_with("cf.br ")
                    || trimmed.starts_with("cf.cond_br ")
                    || trimmed.starts_with("scf.yield ")
                    || trimmed.starts_with("linalg.yield ")
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_manager_creation() {
        let pm = PassManager::new();
        assert_eq!(pm.opt_level(), OptLevel::O1);
    }

    #[test]
    fn test_pass_manager_with_level() {
        let pm = PassManager::with_opt_level(OptLevel::O3);
        assert_eq!(pm.opt_level(), OptLevel::O3);
    }

    #[test]
    fn test_pass_statistics() {
        let before = r#"
            %0 = arith.constant 1 : i64
            %1 = arith.constant 2 : i64
            %2 = arith.addi %0, %1 : i64
            return %2 : i64
        "#;

        let after = r#"
            %0 = arith.constant 3 : i64
            return %0 : i64
        "#;

        let stats = PassStatistics::from_ir(before, after);
        assert_eq!(stats.ops_before, 4);
        assert_eq!(stats.ops_after, 2);
        assert_eq!(stats.ops_eliminated, 2);
    }

    #[test]
    fn test_default_pipeline() {
        let pm = default_pipeline(OptLevel::O2);
        assert_eq!(pm.passes.len(), 3);
    }
}
