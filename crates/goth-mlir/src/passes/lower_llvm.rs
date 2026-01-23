//! LLVM dialect lowering pass
//!
//! This pass converts MLIR operations to LLVM dialect operations,
//! preparing for final LLVM IR emission.
//!
//! Lowering rules:
//! - arith.* → llvm.* (arithmetic ops)
//! - memref.* → llvm.* (memory ops with explicit pointer handling)
//! - func.* → llvm.* (function ops)
//! - cf.* → llvm.* (control flow)
//! - index type → i64

use crate::error::{MlirError, Result};
use crate::passes::{Pass, OptLevel};
use regex::Regex;
use std::collections::HashMap;

/// Pass to lower MLIR to LLVM dialect
pub struct LowerLlvmPass {
    /// Track value types for proper lowering
    value_types: HashMap<String, String>,
}

impl LowerLlvmPass {
    pub fn new() -> Self {
        Self {
            value_types: HashMap::new(),
        }
    }

    /// Lower a single line of MLIR to LLVM dialect
    fn lower_line(&mut self, line: &str) -> Result<String> {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with("//") {
            return Ok(line.to_string());
        }

        // Module and function markers pass through
        if trimmed.starts_with("module") || trimmed == "}" || trimmed == "{" {
            return Ok(line.to_string());
        }

        // Lower arith operations
        if trimmed.contains("arith.") {
            return self.lower_arith(line);
        }

        // Lower memref operations
        if trimmed.contains("memref.") {
            return self.lower_memref(line);
        }

        // Lower func operations
        if trimmed.contains("func.func") || trimmed.contains("func.return") || trimmed.contains("func.call") {
            return self.lower_func(line);
        }

        // Lower cf operations
        if trimmed.contains("cf.") {
            return self.lower_cf(line);
        }

        // Lower scf operations that remain after bufferization
        if trimmed.contains("scf.") {
            return self.lower_scf(line);
        }

        // Pass through unchanged (already LLVM or unknown)
        Ok(line.to_string())
    }

    /// Lower arith dialect operations
    fn lower_arith(&mut self, line: &str) -> Result<String> {
        let mut result = line.to_string();

        // Integer arithmetic
        result = result.replace("arith.addi", "llvm.add");
        result = result.replace("arith.subi", "llvm.sub");
        result = result.replace("arith.muli", "llvm.mul");
        result = result.replace("arith.divsi", "llvm.sdiv");
        result = result.replace("arith.divui", "llvm.udiv");
        result = result.replace("arith.remsi", "llvm.srem");
        result = result.replace("arith.remui", "llvm.urem");

        // Floating-point arithmetic
        result = result.replace("arith.addf", "llvm.fadd");
        result = result.replace("arith.subf", "llvm.fsub");
        result = result.replace("arith.mulf", "llvm.fmul");
        result = result.replace("arith.divf", "llvm.fdiv");
        result = result.replace("arith.negf", "llvm.fneg");

        // Bitwise operations
        result = result.replace("arith.andi", "llvm.and");
        result = result.replace("arith.ori", "llvm.or");
        result = result.replace("arith.xori", "llvm.xor");
        result = result.replace("arith.shli", "llvm.shl");
        result = result.replace("arith.shrsi", "llvm.ashr");
        result = result.replace("arith.shrui", "llvm.lshr");

        // Comparisons - need special handling for predicate syntax
        if result.contains("arith.cmpi") {
            result = self.lower_cmpi(&result)?;
        }
        if result.contains("arith.cmpf") {
            result = self.lower_cmpf(&result)?;
        }

        // Type conversions
        result = result.replace("arith.extsi", "llvm.sext");
        result = result.replace("arith.extui", "llvm.zext");
        result = result.replace("arith.trunci", "llvm.trunc");
        result = result.replace("arith.sitofp", "llvm.sitofp");
        result = result.replace("arith.uitofp", "llvm.uitofp");
        result = result.replace("arith.fptosi", "llvm.fptosi");
        result = result.replace("arith.fptoui", "llvm.fptoui");
        result = result.replace("arith.extf", "llvm.fpext");
        result = result.replace("arith.truncf", "llvm.fptrunc");

        // Constants
        if result.contains("arith.constant") {
            result = self.lower_constant(&result)?;
        }

        // Lower index type to i64
        result = self.lower_index_type(&result);

        Ok(result)
    }

    /// Lower arith.cmpi to llvm.icmp
    fn lower_cmpi(&self, line: &str) -> Result<String> {
        // Pattern: arith.cmpi <pred>, %a, %b : <type>
        // Convert to: llvm.icmp "<pred>" %a, %b : <type>
        let re = Regex::new(r"arith\.cmpi\s+(\w+),\s*(%\w+),\s*(%\w+)")
            .map_err(|e| MlirError::PassError(format!("Regex error: {}", e)))?;

        if let Some(caps) = re.captures(line) {
            let pred = &caps[1];
            let lhs = &caps[2];
            let rhs = &caps[3];
            let result = re.replace(line, format!("llvm.icmp \"{}\" {}, {}", pred, lhs, rhs));
            return Ok(result.to_string());
        }

        Ok(line.to_string())
    }

    /// Lower arith.cmpf to llvm.fcmp
    fn lower_cmpf(&self, line: &str) -> Result<String> {
        // Pattern: arith.cmpf <pred>, %a, %b : <type>
        // Convert to: llvm.fcmp "<pred>" %a, %b : <type>
        let re = Regex::new(r"arith\.cmpf\s+(\w+),\s*(%\w+),\s*(%\w+)")
            .map_err(|e| MlirError::PassError(format!("Regex error: {}", e)))?;

        if let Some(caps) = re.captures(line) {
            let pred = &caps[1];
            let lhs = &caps[2];
            let rhs = &caps[3];
            let result = re.replace(line, format!("llvm.fcmp \"{}\" {}, {}", pred, lhs, rhs));
            return Ok(result.to_string());
        }

        Ok(line.to_string())
    }

    /// Lower arith.constant to llvm.mlir.constant
    fn lower_constant(&self, line: &str) -> Result<String> {
        // Pattern: %x = arith.constant <value> : <type>
        // Convert to: %x = llvm.mlir.constant(<value> : <type>) : <type>
        let re = Regex::new(r"(arith\.constant)\s+([^:]+)\s*:\s*(\S+)")
            .map_err(|e| MlirError::PassError(format!("Regex error: {}", e)))?;

        if let Some(caps) = re.captures(line) {
            let value = caps[2].trim();
            let ty = &caps[3];
            let llvm_ty = self.mlir_type_to_llvm_type(ty);
            let result = re.replace(line, format!("llvm.mlir.constant({} : {}) : {}", value, llvm_ty, llvm_ty));
            return Ok(result.to_string());
        }

        Ok(line.to_string())
    }

    /// Lower memref dialect operations
    fn lower_memref(&mut self, line: &str) -> Result<String> {
        let mut result = line.to_string();

        // memref.alloc -> llvm.alloca (for stack) or llvm.call @malloc
        if result.contains("memref.alloc") {
            result = self.lower_memref_alloc(&result)?;
        }

        // memref.dealloc -> llvm.call @free
        if result.contains("memref.dealloc") {
            result = self.lower_memref_dealloc(&result)?;
        }

        // memref.load -> llvm.load
        if result.contains("memref.load") {
            result = self.lower_memref_load(&result)?;
        }

        // memref.store -> llvm.store
        if result.contains("memref.store") {
            result = self.lower_memref_store(&result)?;
        }

        // memref.dim -> appropriate computation
        // memref.cast -> llvm.bitcast or passthrough

        Ok(result)
    }

    /// Lower memref.alloc to LLVM
    fn lower_memref_alloc(&self, line: &str) -> Result<String> {
        // For now, convert to llvm.alloca for stack allocation
        // More sophisticated handling would use malloc for heap allocation
        let result = line.replace("memref.alloc()", "llvm.alloca");

        // Convert memref types to ptr
        let result = self.lower_memref_type(&result);
        Ok(result)
    }

    /// Lower memref.dealloc to LLVM
    fn lower_memref_dealloc(&self, line: &str) -> Result<String> {
        // Convert to llvm.call @free
        let re = Regex::new(r"memref\.dealloc\s+(%\w+)")
            .map_err(|e| MlirError::PassError(format!("Regex error: {}", e)))?;

        if let Some(caps) = re.captures(line) {
            let ptr = &caps[1];
            let indent = line.len() - line.trim_start().len();
            let indent_str = " ".repeat(indent);
            return Ok(format!("{}llvm.call @free({}) : (!llvm.ptr) -> ()\n", indent_str, ptr));
        }

        Ok(line.to_string())
    }

    /// Lower memref.load to llvm.load
    fn lower_memref_load(&self, line: &str) -> Result<String> {
        let result = line.replace("memref.load", "llvm.load");
        let result = self.lower_memref_type(&result);
        Ok(result)
    }

    /// Lower memref.store to llvm.store
    fn lower_memref_store(&self, line: &str) -> Result<String> {
        let result = line.replace("memref.store", "llvm.store");
        let result = self.lower_memref_type(&result);
        Ok(result)
    }

    /// Convert memref types to LLVM pointer types
    fn lower_memref_type(&self, line: &str) -> String {
        // Replace memref<...> with !llvm.ptr
        let re = Regex::new(r"memref<[^>]+>").unwrap();
        re.replace_all(line, "!llvm.ptr").to_string()
    }

    /// Lower func dialect operations
    fn lower_func(&mut self, line: &str) -> Result<String> {
        let mut result = line.to_string();

        // func.func -> llvm.func
        result = result.replace("func.func", "llvm.func");

        // func.return -> llvm.return
        result = result.replace("func.return", "llvm.return");

        // func.call -> llvm.call
        result = result.replace("func.call", "llvm.call");

        // Lower index type to i64
        result = self.lower_index_type(&result);

        Ok(result)
    }

    /// Lower cf dialect operations
    fn lower_cf(&self, line: &str) -> Result<String> {
        let mut result = line.to_string();

        // cf.br -> llvm.br
        result = result.replace("cf.br", "llvm.br");

        // cf.cond_br -> llvm.cond_br
        result = result.replace("cf.cond_br", "llvm.cond_br");

        // cf.switch -> series of llvm.cond_br (simplified)
        // For now, just pass through as it needs more complex handling

        Ok(result)
    }

    /// Lower scf dialect operations (should be expanded before LLVM lowering)
    fn lower_scf(&self, line: &str) -> Result<String> {
        // SCF operations should ideally be lowered to cf before LLVM lowering
        // For now, emit a warning comment
        let trimmed = line.trim();
        if trimmed.contains("scf.if") || trimmed.contains("scf.for") || trimmed.contains("scf.while") {
            let indent = line.len() - line.trim_start().len();
            let indent_str = " ".repeat(indent);
            return Ok(format!(
                "{}// TODO: Lower SCF to CF before LLVM lowering\n{}",
                indent_str, line
            ));
        }

        // scf.yield -> llvm.br or removed (handled by parent)
        let result = line.replace("scf.yield", "// scf.yield lowered");

        Ok(result)
    }

    /// Lower index type to i64
    fn lower_index_type(&self, line: &str) -> String {
        // Replace standalone 'index' type with 'i64'
        // Be careful not to replace inside identifiers
        let re = Regex::new(r"\bindex\b").unwrap();
        re.replace_all(line, "i64").to_string()
    }

    /// Convert MLIR type to LLVM type string
    fn mlir_type_to_llvm_type(&self, mlir_ty: &str) -> String {
        let ty = mlir_ty.trim();
        match ty {
            "index" => "i64".to_string(),
            "i1" | "i8" | "i16" | "i32" | "i64" => ty.to_string(),
            "f32" => "f32".to_string(),
            "f64" => "f64".to_string(),
            _ if ty.starts_with("memref<") => "!llvm.ptr".to_string(),
            _ => ty.to_string(),
        }
    }
}

impl Default for LowerLlvmPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for LowerLlvmPass {
    fn name(&self) -> &'static str {
        "lower-to-llvm"
    }

    fn description(&self) -> &'static str {
        "Lower MLIR dialects to LLVM dialect for code generation"
    }

    fn run(&self, mlir: &str) -> Result<String> {
        let mut pass = LowerLlvmPass::new();
        let mut result = String::new();

        for line in mlir.lines() {
            let lowered = pass.lower_line(line)?;
            result.push_str(&lowered);
            if !lowered.ends_with('\n') {
                result.push('\n');
            }
        }

        Ok(result)
    }

    fn should_run(&self, _level: OptLevel) -> bool {
        true // Always run when included in pipeline
    }
}

/// Lower MLIR module to LLVM dialect
pub fn lower_to_llvm(mlir: &str) -> Result<String> {
    let pass = LowerLlvmPass::new();
    pass.run(mlir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_arith_add() {
        let pass = LowerLlvmPass::new();
        let input = "  %2 = arith.addi %0, %1 : i64\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.add"));
        assert!(!result.contains("arith.addi"));
    }

    #[test]
    fn test_lower_arith_constant() {
        let pass = LowerLlvmPass::new();
        let input = "  %0 = arith.constant 42 : i64\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.mlir.constant"));
        assert!(result.contains("42"));
    }

    #[test]
    fn test_lower_cmpi() {
        let pass = LowerLlvmPass::new();
        let input = "  %2 = arith.cmpi slt, %0, %1 : i64\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.icmp"));
        assert!(result.contains("\"slt\""));
    }

    #[test]
    fn test_lower_func() {
        let pass = LowerLlvmPass::new();
        let input = "func.func @main() -> i64 {\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.func"));
        assert!(!result.contains("func.func"));
    }

    #[test]
    fn test_lower_return() {
        let pass = LowerLlvmPass::new();
        let input = "  func.return %0 : i64\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.return"));
    }

    #[test]
    fn test_lower_index_type() {
        let pass = LowerLlvmPass::new();
        let input = "  %0 = arith.constant 10 : index\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("i64"));
        assert!(!result.contains("index"));
    }

    #[test]
    fn test_lower_memref_type() {
        let pass = LowerLlvmPass::new();
        let input = "  %0 = memref.load %buf[%i] : memref<10xi64>\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.load"));
        assert!(result.contains("!llvm.ptr"));
    }

    #[test]
    fn test_lower_control_flow() {
        let pass = LowerLlvmPass::new();
        let input = "  cf.br ^bb1\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.br"));
    }

    #[test]
    fn test_lower_cond_br() {
        let pass = LowerLlvmPass::new();
        let input = "  cf.cond_br %cond, ^bb1, ^bb2\n";
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.cond_br"));
    }

    #[test]
    fn test_full_module() {
        let pass = LowerLlvmPass::new();
        let input = r#"module {
  func.func @add(%a: i64, %b: i64) -> i64 {
    %0 = arith.addi %a, %b : i64
    func.return %0 : i64
  }
}
"#;
        let result = pass.run(input).unwrap();
        assert!(result.contains("llvm.func"));
        assert!(result.contains("llvm.add"));
        assert!(result.contains("llvm.return"));
    }
}
