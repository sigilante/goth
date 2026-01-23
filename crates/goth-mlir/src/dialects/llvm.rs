//! LLVM dialect operations for Goth MLIR emission
//!
//! The LLVM dialect provides a mapping from MLIR to LLVM IR concepts:
//! - Function definitions and calls
//! - Memory operations (alloca, load, store, GEP)
//! - Control flow (br, cond_br)
//! - Type conversions and casts
//! - External function declarations
//!
//! This dialect serves as the final lowering target before LLVM IR text emission.

use crate::error::{MlirError, Result};

/// LLVM type representations
#[derive(Debug, Clone, PartialEq)]
pub enum LlvmType {
    /// Void type
    Void,
    /// Integer type with bit width (i1, i8, i32, i64, etc.)
    Int(u32),
    /// Floating-point types
    Float(FloatType),
    /// Pointer type (opaque in modern LLVM)
    Ptr,
    /// Array type: [count x element_type]
    Array(u64, Box<LlvmType>),
    /// Struct type: { type1, type2, ... }
    Struct(Vec<LlvmType>),
    /// Function type: (args) -> return
    Function(Vec<LlvmType>, Box<LlvmType>),
}

/// LLVM floating-point types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatType {
    Float,   // 32-bit
    Double,  // 64-bit
}

impl LlvmType {
    /// Convert to LLVM IR type string
    pub fn to_string(&self) -> String {
        match self {
            LlvmType::Void => "void".to_string(),
            LlvmType::Int(bits) => format!("i{}", bits),
            LlvmType::Float(FloatType::Float) => "float".to_string(),
            LlvmType::Float(FloatType::Double) => "double".to_string(),
            LlvmType::Ptr => "ptr".to_string(),
            LlvmType::Array(count, elem) => format!("[{} x {}]", count, elem.to_string()),
            LlvmType::Struct(fields) => {
                let fields_str = fields.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{ {} }}", fields_str)
            }
            LlvmType::Function(args, ret) => {
                let args_str = args.iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({}) -> {}", args_str, ret.to_string())
            }
        }
    }
}

/// LLVM integer comparison predicates
#[derive(Debug, Clone, Copy)]
pub enum IcmpPredicate {
    Eq,   // equal
    Ne,   // not equal
    Slt,  // signed less than
    Sle,  // signed less or equal
    Sgt,  // signed greater than
    Sge,  // signed greater or equal
    Ult,  // unsigned less than
    Ule,  // unsigned less or equal
    Ugt,  // unsigned greater than
    Uge,  // unsigned greater or equal
}

impl IcmpPredicate {
    pub fn as_str(&self) -> &'static str {
        match self {
            IcmpPredicate::Eq => "eq",
            IcmpPredicate::Ne => "ne",
            IcmpPredicate::Slt => "slt",
            IcmpPredicate::Sle => "sle",
            IcmpPredicate::Sgt => "sgt",
            IcmpPredicate::Sge => "sge",
            IcmpPredicate::Ult => "ult",
            IcmpPredicate::Ule => "ule",
            IcmpPredicate::Ugt => "ugt",
            IcmpPredicate::Uge => "uge",
        }
    }
}

/// LLVM floating-point comparison predicates
#[derive(Debug, Clone, Copy)]
pub enum FcmpPredicate {
    Oeq,  // ordered equal
    One,  // ordered not equal
    Olt,  // ordered less than
    Ole,  // ordered less or equal
    Ogt,  // ordered greater than
    Oge,  // ordered greater or equal
    Ord,  // ordered (no NaN)
    Uno,  // unordered (either NaN)
}

impl FcmpPredicate {
    pub fn as_str(&self) -> &'static str {
        match self {
            FcmpPredicate::Oeq => "oeq",
            FcmpPredicate::One => "one",
            FcmpPredicate::Olt => "olt",
            FcmpPredicate::Ole => "ole",
            FcmpPredicate::Ogt => "ogt",
            FcmpPredicate::Oge => "oge",
            FcmpPredicate::Ord => "ord",
            FcmpPredicate::Uno => "uno",
        }
    }
}

/// Builder for LLVM dialect operations
pub struct LlvmBuilder {
    indent: usize,
    ssa_counter: u32,
}

impl LlvmBuilder {
    pub fn new() -> Self {
        Self {
            indent: 0,
            ssa_counter: 0,
        }
    }

    fn indent_str(&self) -> String {
        "  ".repeat(self.indent)
    }

    fn fresh_ssa(&mut self) -> String {
        let ssa = format!("%{}", self.ssa_counter);
        self.ssa_counter += 1;
        ssa
    }

    /// Increment indentation level
    pub fn push_indent(&mut self) {
        self.indent += 1;
    }

    /// Decrement indentation level
    pub fn pop_indent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }

    // === Arithmetic Operations ===

    /// Emit llvm.add (integer addition)
    pub fn emit_add(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.add {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.sub (integer subtraction)
    pub fn emit_sub(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.sub {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.mul (integer multiplication)
    pub fn emit_mul(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.mul {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.sdiv (signed integer division)
    pub fn emit_sdiv(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.sdiv {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.srem (signed integer remainder)
    pub fn emit_srem(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.srem {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.fadd (floating-point addition)
    pub fn emit_fadd(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fadd {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.fsub (floating-point subtraction)
    pub fn emit_fsub(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fsub {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.fmul (floating-point multiplication)
    pub fn emit_fmul(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fmul {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.fdiv (floating-point division)
    pub fn emit_fdiv(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fdiv {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    // === Comparison Operations ===

    /// Emit llvm.icmp (integer comparison)
    pub fn emit_icmp(&mut self, pred: IcmpPredicate, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.icmp \"{}\" {}, {} : {}\n",
            self.indent_str(), ssa, pred.as_str(), lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.fcmp (floating-point comparison)
    pub fn emit_fcmp(&mut self, pred: FcmpPredicate, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fcmp \"{}\" {}, {} : {}\n",
            self.indent_str(), ssa, pred.as_str(), lhs, rhs, ty.to_string()
        )
    }

    // === Bitwise Operations ===

    /// Emit llvm.and
    pub fn emit_and(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.and {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.or
    pub fn emit_or(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.or {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    /// Emit llvm.xor
    pub fn emit_xor(&mut self, lhs: &str, rhs: &str, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.xor {}, {} : {}\n",
            self.indent_str(), ssa, lhs, rhs, ty.to_string()
        )
    }

    // === Memory Operations ===

    /// Emit llvm.alloca (stack allocation)
    pub fn emit_alloca(&mut self, elem_ty: &LlvmType, count: Option<&str>) -> String {
        let ssa = self.fresh_ssa();
        if let Some(n) = count {
            format!(
                "{}{} = llvm.alloca {} x {} : (i64) -> !llvm.ptr\n",
                self.indent_str(), ssa, n, elem_ty.to_string()
            )
        } else {
            format!(
                "{}{} = llvm.alloca {} : () -> !llvm.ptr\n",
                self.indent_str(), ssa, elem_ty.to_string()
            )
        }
    }

    /// Emit llvm.load
    pub fn emit_load(&mut self, ptr: &str, result_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.load {} : !llvm.ptr -> {}\n",
            self.indent_str(), ssa, ptr, result_ty.to_string()
        )
    }

    /// Emit llvm.store
    pub fn emit_store(&mut self, value: &str, ptr: &str, value_ty: &LlvmType) -> String {
        format!(
            "{}llvm.store {}, {} : {}, !llvm.ptr\n",
            self.indent_str(), value, ptr, value_ty.to_string()
        )
    }

    /// Emit llvm.getelementptr (pointer arithmetic)
    pub fn emit_gep(&mut self, base_ptr: &str, indices: &[&str], elem_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        let indices_str = indices.join(", ");
        format!(
            "{}{} = llvm.getelementptr {} [{}] : (!llvm.ptr, i64) -> !llvm.ptr, {}\n",
            self.indent_str(), ssa, base_ptr, indices_str, elem_ty.to_string()
        )
    }

    // === Control Flow ===

    /// Emit llvm.br (unconditional branch)
    pub fn emit_br(&self, dest: &str) -> String {
        format!("{}llvm.br ^{}\n", self.indent_str(), dest)
    }

    /// Emit llvm.cond_br (conditional branch)
    pub fn emit_cond_br(&self, cond: &str, true_dest: &str, false_dest: &str) -> String {
        format!(
            "{}llvm.cond_br {}, ^{}, ^{}\n",
            self.indent_str(), cond, true_dest, false_dest
        )
    }

    /// Emit llvm.return
    pub fn emit_return(&self, value: Option<(&str, &LlvmType)>) -> String {
        match value {
            Some((val, ty)) => format!(
                "{}llvm.return {} : {}\n",
                self.indent_str(), val, ty.to_string()
            ),
            None => format!("{}llvm.return\n", self.indent_str()),
        }
    }

    // === Function Operations ===

    /// Emit llvm.func definition
    pub fn emit_func_start(&mut self, name: &str, args: &[(String, LlvmType)], ret_ty: &LlvmType) -> String {
        let args_str = args.iter()
            .map(|(name, ty)| format!("{}: {}", name, ty.to_string()))
            .collect::<Vec<_>>()
            .join(", ");

        self.push_indent();
        format!(
            "llvm.func @{}({}) -> {} {{\n",
            name, args_str, ret_ty.to_string()
        )
    }

    /// Emit llvm.func end
    pub fn emit_func_end(&mut self) -> String {
        self.pop_indent();
        "}\n".to_string()
    }

    /// Emit llvm.call
    pub fn emit_call(&mut self, callee: &str, args: &[&str], ret_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        let args_str = args.join(", ");
        format!(
            "{}{} = llvm.call @{}({}) : ({}) -> {}\n",
            self.indent_str(), ssa, callee, args_str,
            args.iter().map(|_| "...").collect::<Vec<_>>().join(", "),
            ret_ty.to_string()
        )
    }

    /// Emit external function declaration
    pub fn emit_func_decl(&self, name: &str, args: &[LlvmType], ret_ty: &LlvmType) -> String {
        let args_str = args.iter()
            .map(|ty| ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "llvm.func @{}({}) -> {} attributes {{ sym_visibility = \"private\" }}\n",
            name, args_str, ret_ty.to_string()
        )
    }

    // === Type Conversions ===

    /// Emit llvm.sext (sign extend)
    pub fn emit_sext(&mut self, value: &str, from_ty: &LlvmType, to_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.sext {} : {} to {}\n",
            self.indent_str(), ssa, value, from_ty.to_string(), to_ty.to_string()
        )
    }

    /// Emit llvm.trunc (truncate)
    pub fn emit_trunc(&mut self, value: &str, from_ty: &LlvmType, to_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.trunc {} : {} to {}\n",
            self.indent_str(), ssa, value, from_ty.to_string(), to_ty.to_string()
        )
    }

    /// Emit llvm.sitofp (signed int to float)
    pub fn emit_sitofp(&mut self, value: &str, from_ty: &LlvmType, to_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.sitofp {} : {} to {}\n",
            self.indent_str(), ssa, value, from_ty.to_string(), to_ty.to_string()
        )
    }

    /// Emit llvm.fptosi (float to signed int)
    pub fn emit_fptosi(&mut self, value: &str, from_ty: &LlvmType, to_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fptosi {} : {} to {}\n",
            self.indent_str(), ssa, value, from_ty.to_string(), to_ty.to_string()
        )
    }

    /// Emit llvm.fpext (float extend)
    pub fn emit_fpext(&mut self, value: &str, from_ty: &LlvmType, to_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fpext {} : {} to {}\n",
            self.indent_str(), ssa, value, from_ty.to_string(), to_ty.to_string()
        )
    }

    /// Emit llvm.fptrunc (float truncate)
    pub fn emit_fptrunc(&mut self, value: &str, from_ty: &LlvmType, to_ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.fptrunc {} : {} to {}\n",
            self.indent_str(), ssa, value, from_ty.to_string(), to_ty.to_string()
        )
    }

    // === Constants ===

    /// Emit llvm.mlir.constant (integer)
    pub fn emit_constant_int(&mut self, value: i64, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.mlir.constant({} : {}) : {}\n",
            self.indent_str(), ssa, value, ty.to_string(), ty.to_string()
        )
    }

    /// Emit llvm.mlir.constant (float)
    pub fn emit_constant_float(&mut self, value: f64, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        let val_str = if value.fract() == 0.0 {
            format!("{:.1}", value)
        } else {
            format!("{}", value)
        };
        format!(
            "{}{} = llvm.mlir.constant({} : {}) : {}\n",
            self.indent_str(), ssa, val_str, ty.to_string(), ty.to_string()
        )
    }

    /// Emit llvm.mlir.null (null pointer)
    pub fn emit_null(&mut self) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.mlir.null : !llvm.ptr\n",
            self.indent_str(), ssa
        )
    }

    /// Emit llvm.mlir.undef (undefined value)
    pub fn emit_undef(&mut self, ty: &LlvmType) -> String {
        let ssa = self.fresh_ssa();
        format!(
            "{}{} = llvm.mlir.undef : {}\n",
            self.indent_str(), ssa, ty.to_string()
        )
    }
}

impl Default for LlvmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert MLIR type string to LLVM type
pub fn mlir_type_to_llvm(mlir_ty: &str) -> Result<LlvmType> {
    let ty = mlir_ty.trim();

    // Integer types
    if ty == "i1" {
        return Ok(LlvmType::Int(1));
    }
    if ty == "i8" {
        return Ok(LlvmType::Int(8));
    }
    if ty == "i16" {
        return Ok(LlvmType::Int(16));
    }
    if ty == "i32" {
        return Ok(LlvmType::Int(32));
    }
    if ty == "i64" {
        return Ok(LlvmType::Int(64));
    }

    // Float types
    if ty == "f32" {
        return Ok(LlvmType::Float(FloatType::Float));
    }
    if ty == "f64" {
        return Ok(LlvmType::Float(FloatType::Double));
    }

    // Index type (lower to i64)
    if ty == "index" {
        return Ok(LlvmType::Int(64));
    }

    // Pointer types (memref becomes ptr)
    if ty.starts_with("memref<") || ty.starts_with("!llvm.ptr") {
        return Ok(LlvmType::Ptr);
    }

    Err(MlirError::UnsupportedType(format!("Cannot convert MLIR type to LLVM: {}", ty)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llvm_type_to_string() {
        assert_eq!(LlvmType::Int(64).to_string(), "i64");
        assert_eq!(LlvmType::Float(FloatType::Double).to_string(), "double");
        assert_eq!(LlvmType::Ptr.to_string(), "ptr");
        assert_eq!(
            LlvmType::Array(10, Box::new(LlvmType::Int(32))).to_string(),
            "[10 x i32]"
        );
    }

    #[test]
    fn test_emit_arithmetic() {
        let mut builder = LlvmBuilder::new();
        let code = builder.emit_add("%0", "%1", &LlvmType::Int(64));
        assert!(code.contains("llvm.add"));
        assert!(code.contains("i64"));
    }

    #[test]
    fn test_emit_comparison() {
        let mut builder = LlvmBuilder::new();
        let code = builder.emit_icmp(IcmpPredicate::Slt, "%0", "%1", &LlvmType::Int(64));
        assert!(code.contains("llvm.icmp"));
        assert!(code.contains("slt"));
    }

    #[test]
    fn test_emit_memory() {
        let mut builder = LlvmBuilder::new();
        let alloca = builder.emit_alloca(&LlvmType::Int(64), None);
        assert!(alloca.contains("llvm.alloca"));

        let load = builder.emit_load("%ptr", &LlvmType::Int(64));
        assert!(load.contains("llvm.load"));

        let store = builder.emit_store("%val", "%ptr", &LlvmType::Int(64));
        assert!(store.contains("llvm.store"));
    }

    #[test]
    fn test_emit_control_flow() {
        let builder = LlvmBuilder::new();

        let br = builder.emit_br("bb1");
        assert!(br.contains("llvm.br ^bb1"));

        let cond_br = builder.emit_cond_br("%cond", "bb1", "bb2");
        assert!(cond_br.contains("llvm.cond_br"));
    }

    #[test]
    fn test_mlir_type_to_llvm() {
        assert_eq!(mlir_type_to_llvm("i64").unwrap(), LlvmType::Int(64));
        assert_eq!(mlir_type_to_llvm("f64").unwrap(), LlvmType::Float(FloatType::Double));
        assert_eq!(mlir_type_to_llvm("index").unwrap(), LlvmType::Int(64));
    }
}
