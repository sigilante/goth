//! Lower Goth dialect operations to standard MLIR dialects
//!
//! This pass transforms custom Goth dialect operations to their equivalent
//! implementations using standard MLIR dialects (linalg, scf, arith, etc.).
//!
//! # Lowering Rules
//!
//! | Goth Operation | Target Dialect | Implementation |
//! |----------------|----------------|----------------|
//! | `goth.iota` | `linalg.generic` | Index-based fill |
//! | `goth.range` | `linalg.generic` | Offset index fill |
//! | `goth.map` | `linalg.generic` | Elementwise apply |
//! | `goth.filter` | `scf.for` + conditionals | Compact copy |
//! | `goth.reduce_sum` | `linalg.reduce` | Addition combiner |
//! | `goth.reduce_prod` | `linalg.reduce` | Multiplication combiner |
//! | `goth.reduce_min` | `linalg.reduce` | Minimum combiner |
//! | `goth.reduce_max` | `linalg.reduce` | Maximum combiner |
//! | `goth.zip` | `linalg.generic` | Paired iteration |
//!
//! # Example
//!
//! Before lowering:
//! ```mlir
//! %0 = goth.iota %n : tensor<?xi64>
//! %1 = goth.map %0, @double : tensor<?xi64>
//! %2 = goth.reduce_sum %1 : i64
//! ```
//!
//! After lowering:
//! ```mlir
//! %0 = tensor.empty(%n) : tensor<?xi64>
//! %1 = linalg.generic {indexing_maps = [...], iterator_types = ["parallel"]}
//!      outs(%0 : tensor<?xi64>) {
//!   ^bb0(%out: i64):
//!     %idx = linalg.index 0 : index
//!     %val = arith.index_cast %idx : index to i64
//!     linalg.yield %val : i64
//!   } -> tensor<?xi64>
//! // ... and so on
//! ```

use crate::error::{MlirError, Result};
use super::{Pass, OptLevel};
use regex::Regex;

/// Pass for lowering Goth dialect to standard MLIR
pub struct LowerGothPass {
    /// Whether to inline small functions during lowering
    inline_small_funcs: bool,
    /// Whether to preserve original operations as comments
    preserve_comments: bool,
}

impl LowerGothPass {
    /// Create a new Goth lowering pass
    pub fn new() -> Self {
        Self {
            inline_small_funcs: true,
            preserve_comments: false,
        }
    }

    /// Set whether to inline small functions
    pub fn with_inlining(mut self, inline: bool) -> Self {
        self.inline_small_funcs = inline;
        self
    }

    /// Set whether to preserve original ops as comments
    pub fn with_comments(mut self, comments: bool) -> Self {
        self.preserve_comments = comments;
        self
    }

    /// Lower goth.iota to linalg.generic
    fn lower_iota(&self, line: &str) -> Option<String> {
        // Pattern: %0 = goth.iota %n : tensor<?xi64>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.iota\s+(%\w+)\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let size = &caps[3];
            let tensor_type = &caps[4];

            // Extract element type from tensor type
            let elem_type = Self::extract_element_type(tensor_type);

            let mut code = String::new();

            // Comment original if requested
            if self.preserve_comments {
                code.push_str(&format!("{}// Original: {}\n", indent, line.trim()));
            }

            // Create empty tensor
            code.push_str(&format!(
                "{}{}_empty = tensor.empty({}) : {}\n",
                indent, result, size, tensor_type
            ));

            // Fill with indices using linalg.generic
            code.push_str(&format!(
                "{}{} = linalg.generic {{\n",
                indent, result
            ));
            code.push_str(&format!(
                "{}  indexing_maps = [affine_map<(d0) -> (d0)>],\n",
                indent
            ));
            code.push_str(&format!(
                "{}  iterator_types = [\"parallel\"]\n",
                indent
            ));
            code.push_str(&format!(
                "{}}} outs({}_empty : {}) {{\n",
                indent, result, tensor_type
            ));
            code.push_str(&format!(
                "{}^bb0(%out: {}):\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}  %idx = linalg.index 0 : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}  %val = arith.index_cast %idx : index to {}\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}  linalg.yield %val : {}\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}}} -> {}\n",
                indent, tensor_type
            ));

            code.trim_end().to_string()
        })
    }

    /// Lower goth.range to linalg.generic
    fn lower_range(&self, line: &str) -> Option<String> {
        // Pattern: %0 = goth.range %start, %end : tensor<?xi64>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.range\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let start = &caps[3];
            let end = &caps[4];
            let tensor_type = &caps[5];
            let elem_type = Self::extract_element_type(tensor_type);

            let mut code = String::new();

            if self.preserve_comments {
                code.push_str(&format!("{}// Original: {}\n", indent, line.trim()));
            }

            // Compute size
            code.push_str(&format!(
                "{}{}_size = arith.subi {}, {} : {}\n",
                indent, result, end, start, elem_type
            ));

            // Create empty tensor
            code.push_str(&format!(
                "{}{}_empty = tensor.empty({}_size) : {}\n",
                indent, result, result, tensor_type
            ));

            // Fill with range values
            code.push_str(&format!(
                "{}{} = linalg.generic {{\n",
                indent, result
            ));
            code.push_str(&format!(
                "{}  indexing_maps = [affine_map<(d0) -> (d0)>],\n",
                indent
            ));
            code.push_str(&format!(
                "{}  iterator_types = [\"parallel\"]\n",
                indent
            ));
            code.push_str(&format!(
                "{}}} outs({}_empty : {}) {{\n",
                indent, result, tensor_type
            ));
            code.push_str(&format!(
                "{}^bb0(%out: {}):\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}  %idx = linalg.index 0 : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}  %idx_cast = arith.index_cast %idx : index to {}\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}  %val = arith.addi %idx_cast, {} : {}\n",
                indent, start, elem_type
            ));
            code.push_str(&format!(
                "{}  linalg.yield %val : {}\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}}} -> {}\n",
                indent, tensor_type
            ));

            code.trim_end().to_string()
        })
    }

    /// Lower goth.map to linalg.generic
    fn lower_map(&self, line: &str) -> Option<String> {
        // Pattern: %0 = goth.map %tensor, %func : tensor<...>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.map\s+(%\w+),\s*(%\w+|@\w+)\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let input = &caps[3];
            let func = &caps[4];
            let tensor_type = &caps[5];
            let elem_type = Self::extract_element_type(tensor_type);

            let mut code = String::new();

            if self.preserve_comments {
                code.push_str(&format!("{}// Original: {}\n", indent, line.trim()));
            }

            // Create empty output tensor
            code.push_str(&format!(
                "{}{}_out = tensor.empty() : {}\n",
                indent, result, tensor_type
            ));

            // Apply function elementwise
            code.push_str(&format!(
                "{}{} = linalg.generic {{\n",
                indent, result
            ));
            code.push_str(&format!(
                "{}  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],\n",
                indent
            ));
            code.push_str(&format!(
                "{}  iterator_types = [\"parallel\"]\n",
                indent
            ));
            code.push_str(&format!(
                "{}}} ins({} : {}) outs({}_out : {}) {{\n",
                indent, input, tensor_type, result, tensor_type
            ));
            code.push_str(&format!(
                "{}^bb0(%in: {}, %out: {}):\n",
                indent, elem_type, elem_type
            ));
            code.push_str(&format!(
                "{}  %mapped = func.call_indirect {}(%in) : ({}) -> {}\n",
                indent, func, elem_type, elem_type
            ));
            code.push_str(&format!(
                "{}  linalg.yield %mapped : {}\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}}} -> {}\n",
                indent, tensor_type
            ));

            code.trim_end().to_string()
        })
    }

    /// Lower goth.reduce_sum to linalg.reduce
    fn lower_reduce_sum(&self, line: &str) -> Option<String> {
        // Pattern: %0 = goth.reduce_sum %tensor : i64
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.reduce_sum\s+(%\w+)\s*:\s*(\w+)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let input = &caps[3];
            let result_type = &caps[4];

            self.emit_reduce(indent, result, input, result_type, "arith.addf", "0.0")
        })
    }

    /// Lower goth.reduce_prod to linalg.reduce
    fn lower_reduce_prod(&self, line: &str) -> Option<String> {
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.reduce_prod\s+(%\w+)\s*:\s*(\w+)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let input = &caps[3];
            let result_type = &caps[4];

            self.emit_reduce(indent, result, input, result_type, "arith.mulf", "1.0")
        })
    }

    /// Lower goth.reduce_min to linalg.reduce
    fn lower_reduce_min(&self, line: &str) -> Option<String> {
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.reduce_min\s+(%\w+)\s*:\s*(\w+)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let input = &caps[3];
            let result_type = &caps[4];

            self.emit_reduce(indent, result, input, result_type, "arith.minimumf", "inf")
        })
    }

    /// Lower goth.reduce_max to linalg.reduce
    fn lower_reduce_max(&self, line: &str) -> Option<String> {
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.reduce_max\s+(%\w+)\s*:\s*(\w+)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let input = &caps[3];
            let result_type = &caps[4];

            self.emit_reduce(indent, result, input, result_type, "arith.maximumf", "-inf")
        })
    }

    /// Lower goth.zip to linalg.generic
    fn lower_zip(&self, line: &str) -> Option<String> {
        // Pattern: %0 = goth.zip %a, %b : tensor<...>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.zip\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let left = &caps[3];
            let right = &caps[4];
            let result_type = &caps[5];

            let mut code = String::new();

            if self.preserve_comments {
                code.push_str(&format!("{}// Original: {}\n", indent, line.trim()));
            }

            // Create empty output for tuples
            code.push_str(&format!(
                "{}{}_out = tensor.empty() : {}\n",
                indent, result, result_type
            ));

            // Zip using linalg.generic with paired iteration
            code.push_str(&format!(
                "{}{} = linalg.generic {{\n",
                indent, result
            ));
            code.push_str(&format!(
                "{}  indexing_maps = [\n",
                indent
            ));
            code.push_str(&format!(
                "{}    affine_map<(d0) -> (d0)>,\n",
                indent
            ));
            code.push_str(&format!(
                "{}    affine_map<(d0) -> (d0)>,\n",
                indent
            ));
            code.push_str(&format!(
                "{}    affine_map<(d0) -> (d0)>\n",
                indent
            ));
            code.push_str(&format!(
                "{}  ],\n",
                indent
            ));
            code.push_str(&format!(
                "{}  iterator_types = [\"parallel\"]\n",
                indent
            ));
            code.push_str(&format!(
                "{}}} ins({}, {} : tensor<*>, tensor<*>) outs({}_out : {}) {{\n",
                indent, left, right, result, result_type
            ));
            code.push_str(&format!(
                "{}^bb0(%l: ?, %r: ?, %out: ?):\n",
                indent
            ));
            code.push_str(&format!(
                "{}  // Construct tuple (%l, %r)\n",
                indent
            ));
            code.push_str(&format!(
                "{}  linalg.yield %out : ?\n",
                indent
            ));
            code.push_str(&format!(
                "{}}} -> {}\n",
                indent, result_type
            ));

            code.trim_end().to_string()
        })
    }

    /// Lower goth.filter to scf.for with conditional insert
    fn lower_filter(&self, line: &str) -> Option<String> {
        // Pattern: %0 = goth.filter %tensor, %pred : tensor<...>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*goth\.filter\s+(%\w+),\s*(%\w+|@\w+)\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let result = &caps[2];
            let input = &caps[3];
            let pred = &caps[4];
            let tensor_type = &caps[5];
            let elem_type = Self::extract_element_type(tensor_type);

            let mut code = String::new();

            if self.preserve_comments {
                code.push_str(&format!("{}// Original: {}\n", indent, line.trim()));
            }

            // Filter is more complex - requires counting and compacting
            code.push_str(&format!(
                "{}// Filter implementation using scf.for\n",
                indent
            ));
            code.push_str(&format!(
                "{}{}_dim = tensor.dim {}, %c0 : {}\n",
                indent, result, input, tensor_type
            ));

            // Count matching elements first
            code.push_str(&format!(
                "{}{}_count = scf.for %i = %c0 to {}_dim step %c1 iter_args(%cnt = %c0) -> (index) {{\n",
                indent, result, result
            ));
            code.push_str(&format!(
                "{}  %elem = tensor.extract {}[%i] : {}\n",
                indent, input, tensor_type
            ));
            code.push_str(&format!(
                "{}  %keep = func.call_indirect {}(%elem) : ({}) -> i1\n",
                indent, pred, elem_type
            ));
            code.push_str(&format!(
                "{}  %new_cnt = scf.if %keep -> (index) {{\n",
                indent
            ));
            code.push_str(&format!(
                "{}    %inc = arith.addi %cnt, %c1 : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}    scf.yield %inc : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}  }} else {{\n",
                indent
            ));
            code.push_str(&format!(
                "{}    scf.yield %cnt : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}  }}\n",
                indent
            ));
            code.push_str(&format!(
                "{}  scf.yield %new_cnt : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}}}\n",
                indent
            ));

            // Allocate result and fill
            code.push_str(&format!(
                "{}{}_empty = tensor.empty({}_count) : tensor<?x{}>\n",
                indent, result, result, elem_type
            ));
            code.push_str(&format!(
                "{}{} = scf.for %i = %c0 to {}_dim step %c1 iter_args(%out = {}_empty, %j = %c0) -> (tensor<?x{}>, index) {{\n",
                indent, result, result, result, elem_type
            ));
            code.push_str(&format!(
                "{}  %elem = tensor.extract {}[%i] : {}\n",
                indent, input, tensor_type
            ));
            code.push_str(&format!(
                "{}  %keep = func.call_indirect {}(%elem) : ({}) -> i1\n",
                indent, pred, elem_type
            ));
            code.push_str(&format!(
                "{}  %result:2 = scf.if %keep -> (tensor<?x{}>, index) {{\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}    %new_out = tensor.insert %elem into %out[%j] : tensor<?x{}>\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}    %new_j = arith.addi %j, %c1 : index\n",
                indent
            ));
            code.push_str(&format!(
                "{}    scf.yield %new_out, %new_j : tensor<?x{}>, index\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}  }} else {{\n",
                indent
            ));
            code.push_str(&format!(
                "{}    scf.yield %out, %j : tensor<?x{}>, index\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}  }}\n",
                indent
            ));
            code.push_str(&format!(
                "{}  scf.yield %result#0, %result#1 : tensor<?x{}>, index\n",
                indent, elem_type
            ));
            code.push_str(&format!(
                "{}}}\n",
                indent
            ));

            code.trim_end().to_string()
        })
    }

    /// Helper to emit reduction code
    fn emit_reduce(
        &self,
        indent: &str,
        result: &str,
        input: &str,
        result_type: &str,
        combiner: &str,
        init_value: &str,
    ) -> String {
        let mut code = String::new();

        // Initialize accumulator
        code.push_str(&format!(
            "{}{}_init = arith.constant {} : {}\n",
            indent, result, init_value, result_type
        ));

        // Use linalg.reduce
        code.push_str(&format!(
            "{}{} = linalg.reduce ins({} : tensor<*x{}>) outs({}_init : {}) dimensions = [0]\n",
            indent, result, input, result_type, result, result_type
        ));
        code.push_str(&format!(
            "{}  ({{\n",
            indent
        ));
        code.push_str(&format!(
            "{}  ^bb0(%in: {}, %acc: {}):\n",
            indent, result_type, result_type
        ));
        code.push_str(&format!(
            "{}    %new_acc = {} %in, %acc : {}\n",
            indent, combiner, result_type
        ));
        code.push_str(&format!(
            "{}    linalg.yield %new_acc : {}\n",
            indent, result_type
        ));
        code.push_str(&format!(
            "{}  }})\n",
            indent
        ));

        code.trim_end().to_string()
    }

    /// Extract element type from tensor type string
    fn extract_element_type(tensor_type: &str) -> String {
        // tensor<4x8xf64> -> f64
        // tensor<?xi64> -> i64
        if let Some(x_pos) = tensor_type.rfind('x') {
            let after_x = &tensor_type[x_pos + 1..];
            if let Some(end) = after_x.find('>') {
                return after_x[..end].to_string();
            }
        }
        "f64".to_string() // Default fallback
    }

    /// Process a single line
    fn process_line(&self, line: &str) -> String {
        // Try each lowering pattern
        if let Some(lowered) = self.lower_iota(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_range(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_map(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_reduce_sum(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_reduce_prod(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_reduce_min(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_reduce_max(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_zip(line) {
            return lowered;
        }
        if let Some(lowered) = self.lower_filter(line) {
            return lowered;
        }

        // No transformation needed
        line.to_string()
    }
}

impl Default for LowerGothPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for LowerGothPass {
    fn name(&self) -> &'static str {
        "lower-goth"
    }

    fn description(&self) -> &'static str {
        "Lower Goth dialect operations to standard MLIR dialects"
    }

    fn run(&self, mlir: &str) -> Result<String> {
        let mut result = Vec::new();

        for line in mlir.lines() {
            result.push(self.process_line(line));
        }

        Ok(result.join("\n"))
    }
}

/// Run Goth dialect lowering on MLIR text
pub fn lower_goth_dialect(mlir: &str) -> Result<String> {
    let pass = LowerGothPass::new();
    pass.run(mlir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_element_type() {
        assert_eq!(LowerGothPass::extract_element_type("tensor<4x8xf64>"), "f64");
        assert_eq!(LowerGothPass::extract_element_type("tensor<?xi64>"), "i64");
        assert_eq!(LowerGothPass::extract_element_type("tensor<4xi32>"), "i32");
    }

    #[test]
    fn test_lower_iota() {
        let pass = LowerGothPass::new();
        let input = "  %0 = goth.iota %n : tensor<?xi64>";
        let output = pass.lower_iota(input).unwrap();

        assert!(output.contains("tensor.empty"));
        assert!(output.contains("linalg.generic"));
        assert!(output.contains("linalg.index 0"));
        assert!(output.contains("arith.index_cast"));
    }

    #[test]
    fn test_lower_range() {
        let pass = LowerGothPass::new();
        let input = "  %0 = goth.range %start, %end : tensor<?xi64>";
        let output = pass.lower_range(input).unwrap();

        assert!(output.contains("arith.subi"));
        assert!(output.contains("tensor.empty"));
        assert!(output.contains("linalg.generic"));
        assert!(output.contains("arith.addi"));
    }

    #[test]
    fn test_lower_map() {
        let pass = LowerGothPass::new();
        let input = "  %1 = goth.map %0, @double : tensor<?xf64>";
        let output = pass.lower_map(input).unwrap();

        assert!(output.contains("tensor.empty"));
        assert!(output.contains("linalg.generic"));
        assert!(output.contains("func.call_indirect"));
    }

    #[test]
    fn test_lower_reduce_sum() {
        let pass = LowerGothPass::new();
        let input = "  %2 = goth.reduce_sum %1 : f64";
        let output = pass.lower_reduce_sum(input).unwrap();

        assert!(output.contains("arith.constant 0.0"));
        assert!(output.contains("linalg.reduce"));
        assert!(output.contains("arith.addf"));
    }

    #[test]
    fn test_lower_reduce_prod() {
        let pass = LowerGothPass::new();
        let input = "  %2 = goth.reduce_prod %1 : f64";
        let output = pass.lower_reduce_prod(input).unwrap();

        assert!(output.contains("arith.constant 1.0"));
        assert!(output.contains("arith.mulf"));
    }

    #[test]
    fn test_lower_goth_dialect() {
        let mlir = r#"
func.func @test(%n: i64) -> i64 {
  %0 = goth.iota %n : tensor<?xi64>
  %1 = goth.reduce_sum %0 : i64
  return %1 : i64
}
"#;

        let result = lower_goth_dialect(mlir).unwrap();
        assert!(result.contains("linalg.generic"));
        assert!(result.contains("linalg.reduce"));
        assert!(!result.contains("goth.iota"));
    }

    #[test]
    fn test_pass_preserves_non_goth() {
        let pass = LowerGothPass::new();
        let input = "  %0 = arith.constant 42 : i64";
        let output = pass.process_line(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_with_comments() {
        let pass = LowerGothPass::new().with_comments(true);
        let input = "  %0 = goth.iota %n : tensor<?xi64>";
        let output = pass.lower_iota(input).unwrap();

        assert!(output.contains("// Original:"));
    }
}
