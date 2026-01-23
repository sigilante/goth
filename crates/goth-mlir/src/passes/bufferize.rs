//! Bufferization pass: Convert tensor types to memref types
//!
//! This pass transforms tensor-based IR to memory-based IR by:
//! 1. Converting tensor types to memref types
//! 2. Inserting alloc/dealloc operations for memory management
//! 3. Converting tensor operations to their memref equivalents
//!
//! # Example
//!
//! Before bufferization:
//! ```mlir
//! %0 = tensor.empty() : tensor<4x8xf64>
//! %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<4x8xf64>) -> tensor<4x8xf64>
//! ```
//!
//! After bufferization:
//! ```mlir
//! %0 = memref.alloc() : memref<4x8xf64>
//! linalg.fill ins(%cst : f64) outs(%0 : memref<4x8xf64>)
//! ```

use crate::error::{MlirError, Result};
use super::{Pass, OptLevel};
use std::collections::HashMap;
use regex::Regex;

/// Options for the bufferization pass
#[derive(Debug, Clone)]
pub struct BufferizeOptions {
    /// Whether to use stack allocation for small tensors
    pub use_alloca_for_small: bool,
    /// Maximum size (in elements) for stack allocation
    pub alloca_max_size: usize,
    /// Whether to insert dealloc operations
    pub insert_deallocs: bool,
    /// Alignment for allocations (in bytes)
    pub alignment: Option<u64>,
    /// Whether to copy-on-write for function arguments
    pub copy_function_args: bool,
}

impl Default for BufferizeOptions {
    fn default() -> Self {
        Self {
            use_alloca_for_small: true,
            alloca_max_size: 1024,
            insert_deallocs: true,
            alignment: Some(64), // Cache-line alignment
            copy_function_args: false,
        }
    }
}

/// Bufferization pass implementation
pub struct BufferizePass {
    options: BufferizeOptions,
}

impl BufferizePass {
    /// Create a new bufferization pass with default options
    pub fn new() -> Self {
        Self {
            options: BufferizeOptions::default(),
        }
    }

    /// Create a bufferization pass with custom options
    pub fn with_options(options: BufferizeOptions) -> Self {
        Self { options }
    }

    /// Convert a tensor type string to a memref type string
    fn tensor_to_memref(tensor_type: &str) -> String {
        if tensor_type.starts_with("tensor<") && tensor_type.ends_with(">") {
            let inner = &tensor_type[7..tensor_type.len()-1];
            format!("memref<{}>", inner)
        } else {
            tensor_type.to_string()
        }
    }

    /// Estimate tensor size in elements from type string
    fn estimate_size(tensor_type: &str) -> Option<usize> {
        // Parse tensor<4x8xf64> -> 4 * 8 = 32
        if !tensor_type.starts_with("tensor<") {
            return None;
        }

        let inner = &tensor_type[7..tensor_type.len()-1];
        let parts: Vec<&str> = inner.split('x').collect();

        // Skip if any dimension is dynamic (?)
        if parts.iter().any(|p| p.contains('?')) {
            return None;
        }

        let mut size = 1usize;
        for part in &parts[..parts.len()-1] { // Skip element type
            if let Ok(dim) = part.parse::<usize>() {
                size = size.saturating_mul(dim);
            } else {
                return None;
            }
        }

        Some(size)
    }

    /// Determine allocation strategy for a tensor
    fn allocation_strategy(&self, tensor_type: &str) -> AllocationStrategy {
        if self.options.use_alloca_for_small {
            if let Some(size) = Self::estimate_size(tensor_type) {
                if size <= self.options.alloca_max_size {
                    return AllocationStrategy::StackAlloc;
                }
            }
        }
        AllocationStrategy::HeapAlloc
    }

    /// Transform tensor.empty to memref.alloc
    fn transform_tensor_empty(&self, line: &str) -> Option<String> {
        // Pattern: %0 = tensor.empty() : tensor<...>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*tensor\.empty\(\)\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let ssa = &caps[2];
            let tensor_type = &caps[3];
            let memref_type = Self::tensor_to_memref(tensor_type);

            match self.allocation_strategy(tensor_type) {
                AllocationStrategy::StackAlloc => {
                    format!("{}{} = memref.alloca() : {}", indent, ssa, memref_type)
                }
                AllocationStrategy::HeapAlloc => {
                    if let Some(align) = self.options.alignment {
                        format!("{}{} = memref.alloc() {{ alignment = {} }} : {}",
                            indent, ssa, align, memref_type)
                    } else {
                        format!("{}{} = memref.alloc() : {}", indent, ssa, memref_type)
                    }
                }
            }
        })
    }

    /// Transform tensor.from_elements to memref + stores
    fn transform_tensor_from_elements(&self, line: &str) -> Option<String> {
        // Pattern: %0 = tensor.from_elements %a, %b : tensor<2xf64>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*tensor\.from_elements\s+([^:]+):\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let ssa = &caps[2];
            let elements = caps[3].trim();
            let tensor_type = &caps[4];
            let memref_type = Self::tensor_to_memref(tensor_type);

            let elem_list: Vec<&str> = elements.split(',').map(|e| e.trim()).collect();
            let mut result = format!("{}{} = memref.alloca() : {}\n", indent, ssa, memref_type);

            for (i, elem) in elem_list.iter().enumerate() {
                result.push_str(&format!(
                    "{}memref.store {}, {}[{}] : {}\n",
                    indent, elem, ssa, i, memref_type
                ));
            }

            result.trim_end().to_string()
        })
    }

    /// Transform tensor.extract to memref.load
    fn transform_tensor_extract(&self, line: &str) -> Option<String> {
        // Pattern: %0 = tensor.extract %t[%i] : tensor<...>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*tensor\.extract\s+(%\w+)\[([^\]]+)\]\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let ssa = &caps[2];
            let memref = &caps[3];
            let indices = &caps[4];
            let tensor_type = &caps[5];
            let memref_type = Self::tensor_to_memref(tensor_type);

            format!("{}{} = memref.load {}[{}] : {}", indent, ssa, memref, indices, memref_type)
        })
    }

    /// Transform tensor.insert to memref.store
    fn transform_tensor_insert(&self, line: &str) -> Option<String> {
        // Pattern: %0 = tensor.insert %v into %t[%i] : tensor<...>
        let re = Regex::new(r"(\s*)(%\w+)\s*=\s*tensor\.insert\s+(%\w+)\s+into\s+(%\w+)\[([^\]]+)\]\s*:\s*(tensor<[^>]+>)").ok()?;

        re.captures(line).map(|caps| {
            let indent = &caps[1];
            let ssa = &caps[2];
            let value = &caps[3];
            let memref = &caps[4];
            let indices = &caps[5];
            let tensor_type = &caps[6];
            let memref_type = Self::tensor_to_memref(tensor_type);

            // Store in-place, return same memref
            format!(
                "{}memref.store {}, {}[{}] : {}\n{}{} = memref.cast {} : {} to {}",
                indent, value, memref, indices, memref_type,
                indent, ssa, memref, memref_type, memref_type
            )
        })
    }

    /// Replace tensor types with memref types in a line
    fn replace_tensor_types(&self, line: &str) -> String {
        let re = Regex::new(r"tensor<([^>]+)>").unwrap();
        re.replace_all(line, "memref<$1>").to_string()
    }

    /// Process a single line
    fn process_line(&self, line: &str) -> String {
        // Try specific transformations first
        if let Some(transformed) = self.transform_tensor_empty(line) {
            return transformed;
        }
        if let Some(transformed) = self.transform_tensor_from_elements(line) {
            return transformed;
        }
        if let Some(transformed) = self.transform_tensor_extract(line) {
            return transformed;
        }
        if let Some(transformed) = self.transform_tensor_insert(line) {
            return transformed;
        }

        // Generic type replacement for other operations
        if line.contains("tensor<") {
            self.replace_tensor_types(line)
        } else {
            line.to_string()
        }
    }
}

impl Default for BufferizePass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for BufferizePass {
    fn name(&self) -> &'static str {
        "bufferize"
    }

    fn description(&self) -> &'static str {
        "Convert tensor types to memref types for memory management"
    }

    fn run(&self, mlir: &str) -> Result<String> {
        let mut result = Vec::new();

        for line in mlir.lines() {
            result.push(self.process_line(line));
        }

        Ok(result.join("\n"))
    }
}

/// Allocation strategy for bufferization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationStrategy {
    /// Stack allocation (memref.alloca)
    StackAlloc,
    /// Heap allocation (memref.alloc)
    HeapAlloc,
}

/// Run bufferization on MLIR text
pub fn bufferize_module(mlir: &str) -> Result<String> {
    let pass = BufferizePass::new();
    pass.run(mlir)
}

/// Run bufferization with custom options
pub fn bufferize_module_with_options(mlir: &str, options: BufferizeOptions) -> Result<String> {
    let pass = BufferizePass::with_options(options);
    pass.run(mlir)
}

/// Analyze tensor lifetimes for optimal allocation
#[derive(Debug)]
pub struct TensorLifetimeAnalysis {
    /// Map from SSA value to its definition site
    definitions: HashMap<String, usize>,
    /// Map from SSA value to its last use site
    last_uses: HashMap<String, usize>,
}

impl TensorLifetimeAnalysis {
    /// Analyze lifetimes in MLIR text
    pub fn analyze(mlir: &str) -> Self {
        let mut definitions = HashMap::new();
        let mut last_uses = HashMap::new();

        let def_re = Regex::new(r"(%\w+)\s*=").unwrap();
        let use_re = Regex::new(r"%\w+").unwrap();

        for (line_num, line) in mlir.lines().enumerate() {
            // Find definitions
            for cap in def_re.captures_iter(line) {
                let ssa = cap[1].to_string();
                definitions.insert(ssa, line_num);
            }

            // Find uses (excluding definitions)
            let def_end = if let Some(eq_pos) = line.find('=') {
                eq_pos + 1
            } else {
                0
            };

            for mat in use_re.find_iter(&line[def_end..]) {
                let ssa = mat.as_str().to_string();
                last_uses.insert(ssa, line_num);
            }
        }

        Self { definitions, last_uses }
    }

    /// Get the lifetime span for an SSA value
    pub fn lifetime(&self, ssa: &str) -> Option<(usize, usize)> {
        let def = self.definitions.get(ssa)?;
        let last = self.last_uses.get(ssa).unwrap_or(def);
        Some((*def, *last))
    }

    /// Check if two SSA values have overlapping lifetimes
    pub fn overlaps(&self, a: &str, b: &str) -> bool {
        if let (Some((a_start, a_end)), Some((b_start, b_end))) =
            (self.lifetime(a), self.lifetime(b))
        {
            !(a_end < b_start || b_end < a_start)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_to_memref() {
        assert_eq!(
            BufferizePass::tensor_to_memref("tensor<4x8xf64>"),
            "memref<4x8xf64>"
        );
        assert_eq!(
            BufferizePass::tensor_to_memref("tensor<?xf32>"),
            "memref<?xf32>"
        );
    }

    #[test]
    fn test_estimate_size() {
        assert_eq!(BufferizePass::estimate_size("tensor<4x8xf64>"), Some(32));
        assert_eq!(BufferizePass::estimate_size("tensor<2x3x4xi32>"), Some(24));
        assert_eq!(BufferizePass::estimate_size("tensor<?x8xf64>"), None);
    }

    #[test]
    fn test_transform_tensor_empty() {
        let pass = BufferizePass::new();

        let input = "  %0 = tensor.empty() : tensor<4x8xf64>";
        let output = pass.transform_tensor_empty(input).unwrap();
        assert!(output.contains("memref.alloca()") || output.contains("memref.alloc()"));
        assert!(output.contains("memref<4x8xf64>"));
    }

    #[test]
    fn test_transform_tensor_extract() {
        let pass = BufferizePass::new();

        let input = "  %1 = tensor.extract %0[%i, %j] : tensor<4x8xf64>";
        let output = pass.transform_tensor_extract(input).unwrap();
        assert!(output.contains("memref.load"));
        assert!(output.contains("memref<4x8xf64>"));
    }

    #[test]
    fn test_bufferize_module() {
        let mlir = r#"
func.func @test() -> f64 {
  %0 = tensor.empty() : tensor<4x8xf64>
  %c0 = arith.constant 0 : index
  %1 = tensor.extract %0[%c0, %c0] : tensor<4x8xf64>
  return %1 : f64
}
"#;

        let result = bufferize_module(mlir).unwrap();
        assert!(result.contains("memref<4x8xf64>"));
        assert!(result.contains("memref.alloc") || result.contains("memref.alloca"));
        assert!(result.contains("memref.load"));
    }

    #[test]
    fn test_lifetime_analysis() {
        let mlir = r#"
%0 = arith.constant 1 : i64
%1 = arith.constant 2 : i64
%2 = arith.addi %0, %1 : i64
%3 = arith.muli %2, %0 : i64
"#;

        let analysis = TensorLifetimeAnalysis::analyze(mlir);

        assert!(analysis.lifetime("%0").is_some());
        assert!(analysis.overlaps("%0", "%2"));
    }

    #[test]
    fn test_allocation_strategy() {
        let pass = BufferizePass::new();

        // Small tensor -> stack
        assert_eq!(
            pass.allocation_strategy("tensor<4x4xf64>"),
            AllocationStrategy::StackAlloc
        );

        // Large tensor -> heap
        assert_eq!(
            pass.allocation_strategy("tensor<1000x1000xf64>"),
            AllocationStrategy::HeapAlloc
        );

        // Dynamic tensor -> heap
        assert_eq!(
            pass.allocation_strategy("tensor<?x8xf64>"),
            AllocationStrategy::HeapAlloc
        );
    }
}
