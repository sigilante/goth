//! Module loader for Goth
//!
//! Resolves `use "path"` declarations by loading and inlining referenced files.
//! Handles circular import detection and deduplication.
//! Supports three file formats:
//! - `.goth` - Unicode text source
//! - `.gast` - JSON AST (via serde_json)
//! - `.gbin` - Binary AST (via bincode)

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::fs;
use thiserror::Error;

use goth_ast::decl::{Module, Decl};
use crate::parser::parse_module;

/// Load error
#[derive(Error, Debug)]
pub enum LoadError {
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    #[error("IO error reading {path}: {source}")]
    IoError { path: PathBuf, source: std::io::Error },

    #[error("Parse error in {path}: {source}")]
    ParseError { path: PathBuf, source: crate::parser::ParseError },

    #[error("Circular import detected: {0}")]
    CircularImport(PathBuf),

    #[error("JSON deserialization error in {path}: {source}")]
    JsonError { path: PathBuf, source: serde_json::Error },

    #[error("Binary deserialization error in {path}: {source}")]
    BinaryError { path: PathBuf, source: bincode::Error },
}

pub type LoadResult<T> = Result<T, LoadError>;

/// Module loader with import tracking
pub struct Loader {
    /// Base directory for resolving relative paths
    base_dir: PathBuf,
    /// Set of already-loaded file paths (canonical)
    loaded: HashSet<PathBuf>,
    /// Current import stack for cycle detection
    import_stack: Vec<PathBuf>,
}

impl Loader {
    /// Create a new loader with the given base directory
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Loader {
            base_dir: base_dir.as_ref().to_path_buf(),
            loaded: HashSet::new(),
            import_stack: Vec::new(),
        }
    }

    /// Load and resolve a module from a file path
    /// Supports .goth (text), .gast (JSON), and .gbin (binary) formats
    pub fn load_file(&mut self, path: impl AsRef<Path>) -> LoadResult<Module> {
        let path = self.resolve_path(path.as_ref())?;

        // Check for circular import
        if self.import_stack.contains(&path) {
            return Err(LoadError::CircularImport(path));
        }

        // Check if already loaded (deduplication)
        if self.loaded.contains(&path) {
            return Ok(Module::new(vec![])); // Return empty, already included
        }

        // Load module based on file extension
        let module = self.load_module_by_format(&path)?;

        // Mark as loaded and push to import stack
        self.loaded.insert(path.clone());
        self.import_stack.push(path.clone());

        // Resolve all use declarations
        let resolved = self.resolve_uses(module, &path)?;

        // Pop from import stack
        self.import_stack.pop();

        Ok(resolved)
    }

    /// Load a module based on file extension
    fn load_module_by_format(&self, path: &Path) -> LoadResult<Module> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        match ext {
            "gast" => {
                // JSON AST format
                let bytes = fs::read(path)
                    .map_err(|e| LoadError::IoError { path: path.to_path_buf(), source: e })?;
                serde_json::from_slice(&bytes)
                    .map_err(|e| LoadError::JsonError { path: path.to_path_buf(), source: e })
            }
            "gbin" => {
                // Binary AST format
                let bytes = fs::read(path)
                    .map_err(|e| LoadError::IoError { path: path.to_path_buf(), source: e })?;
                bincode::deserialize(&bytes)
                    .map_err(|e| LoadError::BinaryError { path: path.to_path_buf(), source: e })
            }
            _ => {
                // Default: text source (.goth or unknown)
                let source = fs::read_to_string(path)
                    .map_err(|e| LoadError::IoError { path: path.to_path_buf(), source: e })?;
                parse_module(&source, name)
                    .map_err(|e| LoadError::ParseError { path: path.to_path_buf(), source: e })
            }
        }
    }

    /// Load and resolve a module from source string
    pub fn load_source(&mut self, source: &str, name: &str) -> LoadResult<Module> {
        let module = parse_module(source, name)
            .map_err(|e| LoadError::ParseError {
                path: PathBuf::from("<source>"),
                source: e
            })?;

        self.resolve_uses(module, &self.base_dir.clone())
    }

    /// Resolve use declarations in a module
    fn resolve_uses(&mut self, module: Module, current_file: &Path) -> LoadResult<Module> {
        let current_dir = current_file.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.base_dir.clone());
        let mut resolved_decls = Vec::new();

        for decl in module.decls {
            match decl {
                Decl::Use(use_decl) => {
                    // Resolve the import path relative to current file's directory
                    let import_path = current_dir.join(use_decl.path.as_ref());

                    // Recursively load the imported module
                    let old_base = self.base_dir.clone();
                    self.base_dir = import_path.parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| old_base.clone());

                    let imported = self.load_file(&import_path)?;

                    self.base_dir = old_base;

                    // Inline the imported declarations
                    resolved_decls.extend(imported.decls);
                }
                other => {
                    resolved_decls.push(other);
                }
            }
        }

        Ok(Module {
            name: module.name,
            decls: resolved_decls,
        })
    }

    /// Resolve a path to an absolute canonical path
    fn resolve_path(&self, path: &Path) -> LoadResult<PathBuf> {
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };

        // Try with and without .goth extension
        let candidates = if full_path.extension().is_some() {
            vec![full_path.clone()]
        } else {
            vec![
                full_path.clone(),
                full_path.with_extension("goth"),
            ]
        };

        for candidate in candidates {
            if candidate.exists() {
                return candidate.canonicalize()
                    .map_err(|e| LoadError::IoError { path: candidate, source: e });
            }
        }

        Err(LoadError::FileNotFound(full_path))
    }
}

/// Convenience function to load a file with default settings
pub fn load_file(path: impl AsRef<Path>) -> LoadResult<Module> {
    let path = path.as_ref();
    let base_dir = path.parent().unwrap_or(Path::new("."));
    let file_name = path.file_name()
        .map(|n| Path::new(n))
        .unwrap_or(path);
    let mut loader = Loader::new(base_dir);
    loader.load_file(file_name)
}

/// Convenience function to load source with a base directory
pub fn load_source(source: &str, name: &str, base_dir: impl AsRef<Path>) -> LoadResult<Module> {
    let mut loader = Loader::new(base_dir);
    loader.load_source(source, name)
}

/// Save a module to a file (format determined by extension)
/// Supports .goth (text), .gast (JSON), and .gbin (binary) formats
pub fn save_file(module: &Module, path: impl AsRef<Path>) -> LoadResult<()> {
    use std::io::Write;

    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let bytes = match ext {
        "gast" => {
            // JSON AST format (pretty-printed)
            serde_json::to_vec_pretty(module)
                .map_err(|e| LoadError::JsonError { path: path.to_path_buf(), source: e })?
        }
        "gbin" => {
            // Binary AST format
            bincode::serialize(module)
                .map_err(|e| LoadError::BinaryError { path: path.to_path_buf(), source: e })?
        }
        _ => {
            // Default: text source (.goth)
            goth_ast::pretty::print_module(module).into_bytes()
        }
    };

    let mut file = fs::File::create(path)
        .map_err(|e| LoadError::IoError { path: path.to_path_buf(), source: e })?;
    file.write_all(&bytes)
        .map_err(|e| LoadError::IoError { path: path.to_path_buf(), source: e })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_use_decl() {
        let source = r#"use "math.goth""#;
        let module = parse_module(source, "test").unwrap();
        assert_eq!(module.decls.len(), 1);
        match &module.decls[0] {
            Decl::Use(u) => assert_eq!(u.path.as_ref(), "math.goth"),
            _ => panic!("Expected Use decl"),
        }
    }
}
