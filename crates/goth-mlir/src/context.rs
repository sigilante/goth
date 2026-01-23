//! MLIR context management for Goth
//!
//! This module provides a wrapper around the MLIR context that manages:
//! - Dialect registration
//! - Module creation
//! - Location tracking for diagnostics

#[cfg(feature = "melior")]
use melior::{
    ir::{Location, Module, Block, Region, Type as MlirType, Value, Operation},
    Context,
    dialect::DialectRegistry,
};

use std::collections::HashMap;
use crate::error::{MlirError, Result};

/// MLIR context wrapper for Goth compilation
///
/// This struct manages the MLIR context and module, providing a high-level
/// interface for building MLIR operations from Goth MIR.
#[cfg(feature = "melior")]
pub struct GothMlirContext<'ctx> {
    /// The underlying MLIR context
    ctx: &'ctx Context,
    /// The MLIR module being built
    module: Module<'ctx>,
    /// Current source location for debugging
    current_location: Location<'ctx>,
    /// SSA value counter for fresh names
    next_ssa: usize,
    /// Mapping from MIR locals to MLIR values
    value_map: HashMap<goth_mir::mir::LocalId, Value<'ctx, 'ctx>>,
}

#[cfg(feature = "melior")]
impl<'ctx> GothMlirContext<'ctx> {
    /// Create a new Goth MLIR context
    pub fn new(ctx: &'ctx Context) -> Self {
        // Load all standard dialects
        ctx.load_all_available_dialects();

        let unknown_loc = Location::unknown(ctx);
        let module = Module::new(unknown_loc);

        Self {
            ctx,
            module,
            current_location: unknown_loc,
            next_ssa: 0,
            value_map: HashMap::new(),
        }
    }

    /// Get the underlying MLIR context
    pub fn context(&self) -> &'ctx Context {
        self.ctx
    }

    /// Get the MLIR module
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    /// Get mutable reference to the MLIR module
    pub fn module_mut(&mut self) -> &mut Module<'ctx> {
        &mut self.module
    }

    /// Get the current source location
    pub fn location(&self) -> Location<'ctx> {
        self.current_location
    }

    /// Set the current source location
    pub fn set_location(&mut self, loc: Location<'ctx>) {
        self.current_location = loc;
    }

    /// Create an unknown location
    pub fn unknown_location(&self) -> Location<'ctx> {
        Location::unknown(self.ctx)
    }

    /// Generate a fresh SSA name
    pub fn fresh_ssa(&mut self) -> usize {
        let id = self.next_ssa;
        self.next_ssa += 1;
        id
    }

    /// Register a MIR local with its corresponding MLIR value
    pub fn register_value(&mut self, local: goth_mir::mir::LocalId, value: Value<'ctx, 'ctx>) {
        self.value_map.insert(local, value);
    }

    /// Look up an MLIR value for a MIR local
    pub fn get_value(&self, local: &goth_mir::mir::LocalId) -> Result<Value<'ctx, 'ctx>> {
        self.value_map.get(local)
            .copied()
            .ok_or_else(|| MlirError::CodeGen(format!("Undefined local: {:?}", local)))
    }

    /// Verify the module for correctness
    pub fn verify(&self) -> bool {
        self.module.as_operation().verify()
    }

    /// Get the module as a string (MLIR textual format)
    pub fn to_string(&self) -> String {
        self.module.as_operation().to_string()
    }
}

/// Text-based MLIR context (fallback when melior is not available)
///
/// This provides the same interface but generates MLIR text directly,
/// maintaining backwards compatibility with the original implementation.
pub struct TextMlirContext {
    /// Current indentation level
    indent: usize,
    /// SSA value counter
    next_ssa: usize,
    /// Local variable to SSA value mapping
    local_map: HashMap<goth_mir::mir::LocalId, String>,
    /// Type information for locals
    local_types: HashMap<goth_mir::mir::LocalId, goth_ast::types::Type>,
    /// Output buffer
    output: String,
}

impl TextMlirContext {
    /// Create a new text-based MLIR context
    pub fn new() -> Self {
        Self {
            indent: 0,
            next_ssa: 0,
            local_map: HashMap::new(),
            local_types: HashMap::new(),
            output: String::new(),
        }
    }

    /// Generate fresh SSA value name
    pub fn fresh_ssa(&mut self) -> String {
        let name = format!("%{}", self.next_ssa);
        self.next_ssa += 1;
        name
    }

    /// Get SSA value for a local
    pub fn get_ssa(&self, local: &goth_mir::mir::LocalId) -> Result<String> {
        self.local_map.get(local)
            .cloned()
            .ok_or_else(|| MlirError::CodeGen(format!("Undefined local: {:?}", local)))
    }

    /// Register local with SSA value
    pub fn register_local(&mut self, local: goth_mir::mir::LocalId, ssa: String, ty: goth_ast::types::Type) {
        self.local_map.insert(local, ssa);
        self.local_types.insert(local, ty);
    }

    /// Get type for a local
    pub fn get_local_type(&self, local: &goth_mir::mir::LocalId) -> Option<&goth_ast::types::Type> {
        self.local_types.get(local)
    }

    /// Increase indentation
    pub fn push_indent(&mut self) {
        self.indent += 1;
    }

    /// Decrease indentation
    pub fn pop_indent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }

    /// Get current indentation string
    pub fn indent_str(&self) -> String {
        "  ".repeat(self.indent)
    }

    /// Append to output
    pub fn emit(&mut self, s: &str) {
        self.output.push_str(s);
    }

    /// Append line with current indentation
    pub fn emit_line(&mut self, s: &str) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(s);
        self.output.push('\n');
    }

    /// Get the output
    pub fn into_output(self) -> String {
        self.output
    }

    /// Get the output as a reference
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Clear and reset the context
    pub fn reset(&mut self) {
        self.indent = 0;
        self.next_ssa = 0;
        self.local_map.clear();
        self.local_types.clear();
        self.output.clear();
    }
}

impl Default for TextMlirContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_context_ssa_generation() {
        let mut ctx = TextMlirContext::new();
        assert_eq!(ctx.fresh_ssa(), "%0");
        assert_eq!(ctx.fresh_ssa(), "%1");
        assert_eq!(ctx.fresh_ssa(), "%2");
    }

    #[test]
    fn test_text_context_indentation() {
        let mut ctx = TextMlirContext::new();
        assert_eq!(ctx.indent_str(), "");

        ctx.push_indent();
        assert_eq!(ctx.indent_str(), "  ");

        ctx.push_indent();
        assert_eq!(ctx.indent_str(), "    ");

        ctx.pop_indent();
        assert_eq!(ctx.indent_str(), "  ");
    }

    #[test]
    fn test_text_context_emit() {
        let mut ctx = TextMlirContext::new();
        ctx.emit_line("module {");
        ctx.push_indent();
        ctx.emit_line("func.func @main() {");
        ctx.push_indent();
        ctx.emit_line("return");
        ctx.pop_indent();
        ctx.emit_line("}");
        ctx.pop_indent();
        ctx.emit_line("}");

        let output = ctx.into_output();
        assert!(output.contains("module {"));
        assert!(output.contains("  func.func @main()"));
    }

    #[cfg(feature = "melior")]
    #[test]
    fn test_melior_context_creation() {
        let ctx = Context::new();
        let goth_ctx = GothMlirContext::new(&ctx);

        // Should be able to create a valid empty module
        assert!(goth_ctx.verify());
    }
}
