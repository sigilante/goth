//! Optimization passes for MLIR
//!
//! This module provides optimization passes that improve generated code:
//!
//! - **Canonicalization**: Normalize IR patterns to canonical forms
//! - **CSE**: Common Subexpression Elimination
//! - **DCE**: Dead Code Elimination
//! - **Constant Folding**: Evaluate constant expressions at compile time
//! - **Loop Fusion**: Combine adjacent loops for better cache utilization
//!
//! # Optimization Levels
//!
//! | Level | Optimizations Enabled |
//! |-------|----------------------|
//! | O0 | None (debug) |
//! | O1 | DCE, Canonicalization |
//! | O2 | O1 + CSE, Constant Folding |
//! | O3 | O2 + Loop Fusion, Aggressive Inlining |
//!
//! # Example
//!
//! ```rust,ignore
//! use goth_mlir::passes::{OptimizePass, OptLevel};
//!
//! let pass = OptimizePass::new(OptLevel::O2);
//! let optimized = pass.run(&mlir)?;
//! ```

use crate::error::{MlirError, Result};
use super::{Pass, PassStatistics};
use std::collections::{HashMap, HashSet};
use regex::Regex;

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum OptLevel {
    /// No optimizations (debug mode)
    O0,
    /// Basic optimizations (DCE, canonicalization)
    #[default]
    O1,
    /// Standard optimizations (CSE, constant folding)
    O2,
    /// Aggressive optimizations (loop fusion, inlining)
    O3,
}

impl OptLevel {
    /// Check if dead code elimination is enabled
    pub fn dce_enabled(&self) -> bool {
        *self >= OptLevel::O1
    }

    /// Check if canonicalization is enabled
    pub fn canonicalization_enabled(&self) -> bool {
        *self >= OptLevel::O1
    }

    /// Check if CSE is enabled
    pub fn cse_enabled(&self) -> bool {
        *self >= OptLevel::O2
    }

    /// Check if constant folding is enabled
    pub fn constant_folding_enabled(&self) -> bool {
        *self >= OptLevel::O2
    }

    /// Check if loop fusion is enabled
    pub fn loop_fusion_enabled(&self) -> bool {
        *self >= OptLevel::O3
    }
}

/// Optimization pass
pub struct OptimizePass {
    level: OptLevel,
    /// Maximum number of optimization iterations
    max_iterations: usize,
    /// Statistics from the last run
    stats: Option<PassStatistics>,
}

impl OptimizePass {
    /// Create a new optimization pass
    pub fn new(level: OptLevel) -> Self {
        Self {
            level,
            max_iterations: 10,
            stats: None,
        }
    }

    /// Get statistics from the last run
    pub fn stats(&self) -> Option<&PassStatistics> {
        self.stats.as_ref()
    }

    /// Run dead code elimination
    fn run_dce(&self, mlir: &str) -> String {
        let mut lines: Vec<&str> = mlir.lines().collect();
        let mut used_values = HashSet::new();

        // Collect all used values (multiple passes for dependencies)
        for _ in 0..3 {
            for line in &lines {
                // Find uses on the right side of assignments
                if let Some(eq_pos) = line.find('=') {
                    let rhs = &line[eq_pos + 1..];
                    for cap in Regex::new(r"%\w+").unwrap().find_iter(rhs) {
                        used_values.insert(cap.as_str().to_string());
                    }
                }

                // Find uses in terminators and function calls
                if line.contains("return ")
                    || line.contains("cf.br ")
                    || line.contains("cf.cond_br ")
                    || line.contains("scf.yield ")
                    || line.contains("func.call ")
                {
                    for cap in Regex::new(r"%\w+").unwrap().find_iter(line) {
                        used_values.insert(cap.as_str().to_string());
                    }
                }
            }
        }

        // Filter out dead definitions
        let result: Vec<&str> = lines
            .into_iter()
            .filter(|line| {
                // Keep non-assignment lines
                if !line.contains(" = ") {
                    return true;
                }

                // Check if the defined value is used
                if let Some(cap) = Regex::new(r"(%\w+)\s*=").unwrap().captures(line) {
                    let defined = cap.get(1).unwrap().as_str();
                    used_values.contains(defined)
                } else {
                    true
                }
            })
            .collect();

        result.join("\n")
    }

    /// Run common subexpression elimination
    fn run_cse(&self, mlir: &str) -> String {
        let mut seen_expressions: HashMap<String, String> = HashMap::new();
        let mut replacements: HashMap<String, String> = HashMap::new();
        let mut result = Vec::new();

        for line in mlir.lines() {
            // Look for pure operations that can be CSE'd
            if let Some(caps) = Regex::new(r"(%\w+)\s*=\s*(arith\.\w+|math\.\w+)\s+([^:]+):\s*(.+)")
                .unwrap()
                .captures(line)
            {
                let defined = caps.get(1).unwrap().as_str();
                let op = caps.get(2).unwrap().as_str();
                let operands = caps.get(3).unwrap().as_str().trim();
                let ty = caps.get(4).unwrap().as_str();

                // Normalize operands for commutative ops
                let normalized_operands = if Self::is_commutative(op) {
                    let mut parts: Vec<&str> = operands.split(',').map(|s| s.trim()).collect();
                    parts.sort();
                    parts.join(", ")
                } else {
                    operands.to_string()
                };

                let expr_key = format!("{} {} : {}", op, normalized_operands, ty);

                if let Some(existing) = seen_expressions.get(&expr_key) {
                    // Replace this definition with the existing one
                    replacements.insert(defined.to_string(), existing.clone());
                    // Skip this line
                    continue;
                } else {
                    seen_expressions.insert(expr_key, defined.to_string());
                }
            }

            // Apply replacements to this line
            let mut new_line = line.to_string();
            for (old, new) in &replacements {
                new_line = new_line.replace(old, new);
            }

            result.push(new_line);
        }

        result.join("\n")
    }

    /// Check if an operation is commutative
    fn is_commutative(op: &str) -> bool {
        matches!(
            op,
            "arith.addi"
                | "arith.addf"
                | "arith.muli"
                | "arith.mulf"
                | "arith.andi"
                | "arith.ori"
                | "arith.xori"
                | "arith.maxsi"
                | "arith.minsi"
                | "arith.maximumf"
                | "arith.minimumf"
        )
    }

    /// Run constant folding
    fn run_constant_folding(&self, mlir: &str) -> String {
        let mut constants: HashMap<String, i64> = HashMap::new();
        let mut float_constants: HashMap<String, f64> = HashMap::new();
        let mut result = Vec::new();

        for line in mlir.lines() {
            // Collect constant definitions
            if let Some(caps) = Regex::new(r"(%\w+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i\d+")
                .unwrap()
                .captures(line)
            {
                let name = caps.get(1).unwrap().as_str().to_string();
                let value: i64 = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
                constants.insert(name, value);
                result.push(line.to_string());
                continue;
            }

            if let Some(caps) =
                Regex::new(r"(%\w+)\s*=\s*arith\.constant\s+([0-9.e+-]+)\s*:\s*f\d+")
                    .unwrap()
                    .captures(line)
            {
                let name = caps.get(1).unwrap().as_str().to_string();
                let value: f64 = caps.get(2).unwrap().as_str().parse().unwrap_or(0.0);
                float_constants.insert(name, value);
                result.push(line.to_string());
                continue;
            }

            // Try to fold integer operations
            if let Some(caps) =
                Regex::new(r"(%\w+)\s*=\s*arith\.(addi|subi|muli|divsi)\s+(%\w+),\s*(%\w+)\s*:\s*(i\d+)")
                    .unwrap()
                    .captures(line)
            {
                let defined = caps.get(1).unwrap().as_str();
                let op = caps.get(2).unwrap().as_str();
                let lhs = caps.get(3).unwrap().as_str();
                let rhs = caps.get(4).unwrap().as_str();
                let ty = caps.get(5).unwrap().as_str();

                if let (Some(&l), Some(&r)) = (constants.get(lhs), constants.get(rhs)) {
                    let folded = match op {
                        "addi" => Some(l.wrapping_add(r)),
                        "subi" => Some(l.wrapping_sub(r)),
                        "muli" => Some(l.wrapping_mul(r)),
                        "divsi" if r != 0 => Some(l / r),
                        _ => None,
                    };

                    if let Some(value) = folded {
                        constants.insert(defined.to_string(), value);
                        result.push(format!(
                            "{} = arith.constant {} : {}",
                            defined, value, ty
                        ));
                        continue;
                    }
                }
            }

            // Try to fold float operations
            if let Some(caps) =
                Regex::new(r"(%\w+)\s*=\s*arith\.(addf|subf|mulf|divf)\s+(%\w+),\s*(%\w+)\s*:\s*(f\d+)")
                    .unwrap()
                    .captures(line)
            {
                let defined = caps.get(1).unwrap().as_str();
                let op = caps.get(2).unwrap().as_str();
                let lhs = caps.get(3).unwrap().as_str();
                let rhs = caps.get(4).unwrap().as_str();
                let ty = caps.get(5).unwrap().as_str();

                if let (Some(&l), Some(&r)) = (float_constants.get(lhs), float_constants.get(rhs)) {
                    let folded = match op {
                        "addf" => Some(l + r),
                        "subf" => Some(l - r),
                        "mulf" => Some(l * r),
                        "divf" if r != 0.0 => Some(l / r),
                        _ => None,
                    };

                    if let Some(value) = folded {
                        float_constants.insert(defined.to_string(), value);
                        result.push(format!(
                            "{} = arith.constant {} : {}",
                            defined, value, ty
                        ));
                        continue;
                    }
                }
            }

            result.push(line.to_string());
        }

        result.join("\n")
    }

    /// Run canonicalization (algebraic simplifications)
    fn run_canonicalization(&self, mlir: &str) -> String {
        let mut result = Vec::new();

        for line in mlir.lines() {
            let mut new_line = line.to_string();

            // x + 0 -> x
            if let Some(caps) = Regex::new(r"(%\w+)\s*=\s*arith\.addi\s+(%\w+),\s*(%\w+)\s*:")
                .unwrap()
                .captures(&new_line)
            {
                let defined = caps.get(1).unwrap().as_str();
                let lhs = caps.get(2).unwrap().as_str();
                let rhs = caps.get(3).unwrap().as_str();

                // Check if rhs is zero constant (would need tracking)
                // For now, skip this optimization as it requires constant tracking
                let _ = (defined, lhs, rhs);
            }

            // x * 1 -> x
            // x * 0 -> 0
            // x - x -> 0

            // Simplify double negation
            // -(-x) -> x

            result.push(new_line);
        }

        result.join("\n")
    }

    /// Run a single optimization iteration
    fn run_iteration(&self, mlir: &str) -> String {
        let mut current = mlir.to_string();

        if self.level.dce_enabled() {
            current = self.run_dce(&current);
        }

        if self.level.canonicalization_enabled() {
            current = self.run_canonicalization(&current);
        }

        if self.level.cse_enabled() {
            current = self.run_cse(&current);
        }

        if self.level.constant_folding_enabled() {
            current = self.run_constant_folding(&current);
        }

        current
    }
}

impl Pass for OptimizePass {
    fn name(&self) -> &'static str {
        "optimize"
    }

    fn description(&self) -> &'static str {
        "Run optimization passes on MLIR"
    }

    fn run(&self, mlir: &str) -> Result<String> {
        if self.level == OptLevel::O0 {
            return Ok(mlir.to_string());
        }

        let mut current = mlir.to_string();
        let mut prev = String::new();

        // Iterate until fixed point or max iterations
        for _ in 0..self.max_iterations {
            if current == prev {
                break;
            }
            prev = current.clone();
            current = self.run_iteration(&current);
        }

        Ok(current)
    }

    fn should_run(&self, level: OptLevel) -> bool {
        level >= OptLevel::O1
    }
}

/// Run optimization on MLIR text
pub fn optimize_module(mlir: &str, level: OptLevel) -> Result<String> {
    let pass = OptimizePass::new(level);
    pass.run(mlir)
}

/// Perform only dead code elimination
pub fn eliminate_dead_code(mlir: &str) -> Result<String> {
    let pass = OptimizePass::new(OptLevel::O1);
    Ok(pass.run_dce(mlir))
}

/// Perform only common subexpression elimination
pub fn eliminate_common_subexpressions(mlir: &str) -> Result<String> {
    let pass = OptimizePass::new(OptLevel::O2);
    Ok(pass.run_cse(mlir))
}

/// Perform only constant folding
pub fn fold_constants(mlir: &str) -> Result<String> {
    let pass = OptimizePass::new(OptLevel::O2);
    Ok(pass.run_constant_folding(mlir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_levels() {
        assert!(OptLevel::O1 > OptLevel::O0);
        assert!(OptLevel::O2 > OptLevel::O1);
        assert!(OptLevel::O3 > OptLevel::O2);

        assert!(OptLevel::O2.cse_enabled());
        assert!(!OptLevel::O1.cse_enabled());
    }

    #[test]
    fn test_dce() {
        let pass = OptimizePass::new(OptLevel::O1);

        let mlir = r#"
%0 = arith.constant 1 : i64
%1 = arith.constant 2 : i64
%2 = arith.addi %0, %1 : i64
%dead = arith.constant 99 : i64
return %2 : i64
"#;

        let result = pass.run_dce(mlir);

        // Dead constant should be eliminated
        assert!(!result.contains("%dead"));
        // Used values should be kept
        assert!(result.contains("%0"));
        assert!(result.contains("%1"));
        assert!(result.contains("%2"));
    }

    #[test]
    fn test_cse() {
        let pass = OptimizePass::new(OptLevel::O2);

        let mlir = r#"
%0 = arith.constant 1 : i64
%1 = arith.constant 2 : i64
%2 = arith.addi %0, %1 : i64
%3 = arith.addi %0, %1 : i64
%4 = arith.addi %2, %3 : i64
return %4 : i64
"#;

        let result = pass.run_cse(mlir);

        // Second identical add should be eliminated
        let add_count = result.matches("arith.addi %0, %1").count();
        assert_eq!(add_count, 1, "CSE should eliminate duplicate expression");
    }

    #[test]
    fn test_constant_folding() {
        let pass = OptimizePass::new(OptLevel::O2);

        let mlir = r#"
%0 = arith.constant 10 : i64
%1 = arith.constant 20 : i64
%2 = arith.addi %0, %1 : i64
return %2 : i64
"#;

        let result = pass.run_constant_folding(mlir);

        // Should fold 10 + 20 = 30
        assert!(result.contains("30"), "Should fold constant addition");
    }

    #[test]
    fn test_constant_folding_multiply() {
        let pass = OptimizePass::new(OptLevel::O2);

        let mlir = r#"
%0 = arith.constant 6 : i64
%1 = arith.constant 7 : i64
%2 = arith.muli %0, %1 : i64
return %2 : i64
"#;

        let result = pass.run_constant_folding(mlir);

        // Should fold 6 * 7 = 42
        assert!(result.contains("42"), "Should fold constant multiplication");
    }

    #[test]
    fn test_optimize_module() {
        let mlir = r#"
func.func @test() -> i64 {
  %0 = arith.constant 1 : i64
  %1 = arith.constant 2 : i64
  %2 = arith.addi %0, %1 : i64
  %dead = arith.constant 99 : i64
  %3 = arith.addi %0, %1 : i64
  %4 = arith.addi %2, %3 : i64
  return %4 : i64
}
"#;

        let result = optimize_module(mlir, OptLevel::O2).unwrap();

        // Dead code should be eliminated
        assert!(!result.contains("%dead"));

        // CSE should work
        let add_count = result.matches("arith.addi %0, %1").count();
        assert!(add_count <= 1, "CSE should reduce duplicate expressions");
    }

    #[test]
    fn test_o0_no_optimization() {
        let mlir = r#"
%0 = arith.constant 1 : i64
%dead = arith.constant 99 : i64
return %0 : i64
"#;

        let result = optimize_module(mlir, OptLevel::O0).unwrap();

        // O0 should not optimize anything
        assert!(result.contains("%dead"));
    }

    #[test]
    fn test_is_commutative() {
        assert!(OptimizePass::is_commutative("arith.addi"));
        assert!(OptimizePass::is_commutative("arith.muli"));
        assert!(!OptimizePass::is_commutative("arith.subi"));
        assert!(!OptimizePass::is_commutative("arith.divsi"));
    }
}
