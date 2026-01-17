//! Type checking context (Γ)

use std::collections::{HashMap, HashSet};
use goth_ast::types::Type;

/// Type checking context
#[derive(Debug, Clone)]
pub struct Context {
    /// Type stack for de Bruijn indices (index 0 = last element)
    stack: Vec<Type>,
    
    /// Global name → type bindings
    globals: HashMap<String, Type>,
    
    /// Type variables in scope
    type_vars: HashSet<String>,
    
    /// Shape variables in scope
    shape_vars: HashSet<String>,
}

impl Context {
    pub fn new() -> Self {
        Context {
            stack: Vec::new(),
            globals: HashMap::new(),
            type_vars: HashSet::new(),
            shape_vars: HashSet::new(),
        }
    }

    /// Look up type by de Bruijn index
    pub fn lookup_index(&self, idx: u32) -> Option<&Type> {
        let idx = idx as usize;
        if idx < self.stack.len() {
            Some(&self.stack[self.stack.len() - 1 - idx])
        } else {
            None
        }
    }

    /// Look up type by global name
    pub fn lookup_global(&self, name: &str) -> Option<&Type> {
        self.globals.get(name)
    }

    /// Push a type onto the stack (for lambda/let bindings)
    pub fn push(&mut self, ty: Type) {
        self.stack.push(ty);
    }

    /// Pop a type from the stack
    pub fn pop(&mut self) {
        self.stack.pop();
    }

    /// Push multiple types (for multi-binding patterns)
    pub fn push_many(&mut self, types: &[Type]) {
        self.stack.extend(types.iter().cloned());
    }

    /// Pop multiple types
    pub fn pop_many(&mut self, n: usize) {
        for _ in 0..n {
            self.stack.pop();
        }
    }

    /// Execute with temporary binding
    pub fn with_binding<T>(&mut self, ty: Type, f: impl FnOnce(&mut Self) -> T) -> T {
        self.push(ty);
        let result = f(self);
        self.pop();
        result
    }

    /// Execute with multiple temporary bindings
    pub fn with_bindings<T>(&mut self, types: &[Type], f: impl FnOnce(&mut Self) -> T) -> T {
        self.push_many(types);
        let result = f(self);
        self.pop_many(types.len());
        result
    }

    /// Define a global binding
    pub fn define_global(&mut self, name: impl Into<String>, ty: Type) {
        self.globals.insert(name.into(), ty);
    }

    /// Add type variable to scope
    pub fn add_type_var(&mut self, name: impl Into<String>) {
        self.type_vars.insert(name.into());
    }

    /// Check if type variable is in scope
    pub fn has_type_var(&self, name: &str) -> bool {
        self.type_vars.contains(name)
    }

    /// Add shape variable to scope
    pub fn add_shape_var(&mut self, name: impl Into<String>) {
        self.shape_vars.insert(name.into());
    }

    /// Check if shape variable is in scope
    pub fn has_shape_var(&self, name: &str) -> bool {
        self.shape_vars.contains(name)
    }

    /// Execute with type variables in scope
    pub fn with_type_vars<T>(&mut self, vars: &[String], f: impl FnOnce(&mut Self) -> T) -> T {
        for v in vars {
            self.type_vars.insert(v.clone());
        }
        let result = f(self);
        for v in vars {
            self.type_vars.remove(v);
        }
        result
    }

    /// Current stack depth
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}