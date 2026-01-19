//! Serialization for Goth AST
//!
//! Three formats:
//! - `.goth` - Unicode text (via pretty printer)
//! - `.gast` - JSON AST (via serde_json)  
//! - `.gbin` - Binary AST (via bincode)

use crate::decl::Module;
use crate::expr::Expr;
use thiserror::Error;

/// Serialization error
#[derive(Error, Debug)]
pub enum SerError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Binary error: {0}")]
    Binary(#[from] bincode::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, SerError>;

// ============ JSON (.gast) ============

/// Serialize module to JSON string
pub fn to_json(module: &Module) -> Result<String> {
    Ok(serde_json::to_string_pretty(module)?)
}

/// Serialize module to JSON bytes
pub fn to_json_bytes(module: &Module) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec_pretty(module)?)
}

/// Serialize module to compact JSON (no whitespace)
pub fn to_json_compact(module: &Module) -> Result<String> {
    Ok(serde_json::to_string(module)?)
}

/// Deserialize module from JSON string
pub fn from_json(json: &str) -> Result<Module> {
    Ok(serde_json::from_str(json)?)
}

/// Deserialize module from JSON bytes
pub fn from_json_bytes(bytes: &[u8]) -> Result<Module> {
    Ok(serde_json::from_slice(bytes)?)
}

// ============ Binary (.gbin) ============

/// Serialize module to binary
pub fn to_binary(module: &Module) -> Result<Vec<u8>> {
    Ok(bincode::serialize(module)?)
}

/// Deserialize module from binary
pub fn from_binary(bytes: &[u8]) -> Result<Module> {
    Ok(bincode::deserialize(bytes)?)
}

// ============ Expression-level ============

/// Serialize expression to JSON
pub fn expr_to_json(expr: &Expr) -> Result<String> {
    Ok(serde_json::to_string_pretty(expr)?)
}

/// Deserialize expression from JSON
pub fn expr_from_json(json: &str) -> Result<Expr> {
    Ok(serde_json::from_str(json)?)
}

/// Serialize expression to binary
pub fn expr_to_binary(expr: &Expr) -> Result<Vec<u8>> {
    Ok(bincode::serialize(expr)?)
}

/// Deserialize expression from binary
pub fn expr_from_binary(bytes: &[u8]) -> Result<Expr> {
    Ok(bincode::deserialize(bytes)?)
}

// ============ File I/O ============

/// Write module to file (format inferred from extension)
pub fn write_file(module: &Module, path: &std::path::Path) -> Result<()> {
    use std::io::Write;
    
    let bytes = match path.extension().and_then(|e| e.to_str()) {
        Some("gast") => to_json_bytes(module)?,
        Some("gbin") => to_binary(module)?,
        Some("goth") => crate::pretty::print_module(module).into_bytes(),
        _ => to_json_bytes(module)?, // default to JSON
    };
    
    let mut file = std::fs::File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Read module from file (format inferred from extension)
pub fn read_file(path: &std::path::Path) -> Result<Module> {
    let bytes = std::fs::read(path)?;
    
    match path.extension().and_then(|e| e.to_str()) {
        Some("gast") => from_json_bytes(&bytes),
        Some("gbin") => from_binary(&bytes),
        Some("goth") => {
            // TODO: implement parser
            Err(SerError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Text parsing not yet implemented"
            )))
        }
        _ => from_json_bytes(&bytes), // try JSON by default
    }
}

// ============ Size Estimation ============

/// Estimate binary size of a module
pub fn estimate_binary_size(module: &Module) -> usize {
    // Quick estimate based on serialization
    bincode::serialized_size(module).unwrap_or(0) as usize
}

/// Estimate JSON size of a module (compact)
pub fn estimate_json_size(module: &Module) -> usize {
    serde_json::to_string(module).map(|s| s.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    use crate::literal::Literal;
    use crate::op::BinOp;
    use crate::types::{Type, PrimType};
    use crate::decl::{Decl, LetDecl, FnDecl};
    use crate::effect::Effects;
    use crate::pattern::Pattern;
    use pretty_assertions::assert_eq;
    
    // ============ Expression-level JSON Tests ============
    
    #[test]
    fn test_json_literal_int() {
        let expr = Expr::Lit(Literal::Int(42));
        let json = expr_to_json(&expr).unwrap();
        
        assert!(json.contains("Lit"));
        assert!(json.contains("Int"));
        assert!(json.contains("42"));
        
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_literal_float() {
        let expr = Expr::Lit(Literal::Float(3.14));
        let json = expr_to_json(&expr).unwrap();
        
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_literal_bool() {
        let expr = Expr::Lit(Literal::True);
        let json = expr_to_json(&expr).unwrap();
        
        assert!(json.contains("True"));
        
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_binop() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        
        let json = expr_to_json(&expr).unwrap();
        assert!(json.contains("BinOp"));
        assert!(json.contains("Add"));
        
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_nested_binop() {
        // (1 + 2) * 3
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Lit(Literal::Int(1))),
                Box::new(Expr::Lit(Literal::Int(2))),
            )),
            Box::new(Expr::Lit(Literal::Int(3))),
        );
        
        let json = expr_to_json(&expr).unwrap();
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_lambda() {
        // λ→ ₀
        let expr = Expr::Lam(Box::new(Expr::Idx(0)));
        
        let json = expr_to_json(&expr).unwrap();
        assert!(json.contains("Lam"));
        assert!(json.contains("Idx"));
        
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_application() {
        // f x
        let expr = Expr::App(
            Box::new(Expr::Name("f".into())),
            Box::new(Expr::Name("x".into())),
        );
        
        let json = expr_to_json(&expr).unwrap();
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_let_binding() {
        use crate::pattern::Pattern;
        
        // let x = 5 in x + 1
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(1))),
            )),
        };
        
        let json = expr_to_json(&expr).unwrap();
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_tuple() {
        // ⟨1, true, 3.14⟩
        let expr = Expr::Tuple(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::True),
            Expr::Lit(Literal::Float(3.14)),
        ]);
        
        let json = expr_to_json(&expr).unwrap();
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_array() {
        // [1, 2, 3]
        let expr = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
        ]);
        
        let json = expr_to_json(&expr).unwrap();
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_json_if() {
        // if true then 1 else 2
        let expr = Expr::If {
            cond: Box::new(Expr::Lit(Literal::True)),
            then_: Box::new(Expr::Lit(Literal::Int(1))),
            else_: Box::new(Expr::Lit(Literal::Int(2))),
        };
        
        let json = expr_to_json(&expr).unwrap();
        let parsed = expr_from_json(&json).unwrap();
        assert_eq!(expr, parsed);
    }
    
    // ============ Module-level JSON Tests ============
    
    #[test]
    fn test_json_module_empty() {
        let module = Module {
            name: Some("test".into()),
            decls: vec![],
        };
        
        let json = to_json(&module).unwrap();
        let parsed = from_json(&json).unwrap();
        assert_eq!(module, parsed);
    }
    
    #[test]
    fn test_json_module_with_let() {
        let module = Module {
            name: Some("test".into()),
            decls: vec![
                Decl::Let(LetDecl {
                    name: "x".into(),
                    type_: Some(Type::Prim(PrimType::I64)),
                    value: Expr::Lit(Literal::Int(42)),
                }),
            ],
        };
        
        let json = to_json(&module).unwrap();
        let parsed = from_json(&json).unwrap();
        assert_eq!(module, parsed);
    }
    
    #[test]
    fn test_json_module_with_function() {
        let module = Module {
            name: Some("test".into()),
            decls: vec![
                Decl::Fn(FnDecl {
                    name: "add".into(),
                    signature: Type::func(
                        Type::Prim(PrimType::I64),
                        Type::Prim(PrimType::I64),
                    ),
                    effects: Effects::pure(),
                    body: Expr::BinOp(
                        BinOp::Add,
                        Box::new(Expr::Idx(0)),
                        Box::new(Expr::Lit(Literal::Int(1))),
                    ),
                    preconditions: vec![],
                    postconditions: vec![],
                    constraints: vec![],
                    type_params: vec![],
                }),
            ],
        };
        
        let json = to_json(&module).unwrap();
        let parsed = from_json(&json).unwrap();
        assert_eq!(module, parsed);
    }
    
    // ============ Binary Tests ============
    
    #[test]
    fn test_binary_literal() {
        let expr = Expr::Lit(Literal::Int(42));
        let binary = expr_to_binary(&expr).unwrap();
        let parsed = expr_from_binary(&binary).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_binary_binop() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        
        let binary = expr_to_binary(&expr).unwrap();
        let parsed = expr_from_binary(&binary).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_binary_complex() {
        // let x = 5 in λ→ ₀ + x
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::Lam(Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Idx(1)),
            )))),
        };
        
        let binary = expr_to_binary(&expr).unwrap();
        let parsed = expr_from_binary(&binary).unwrap();
        assert_eq!(expr, parsed);
    }
    
    #[test]
    fn test_binary_module() {
        let module = Module {
            name: Some("test".into()),
            decls: vec![
                Decl::Let(LetDecl {
                    name: "x".into(),
                    type_: None,
                    value: Expr::Lit(Literal::Int(42)),
                }),
            ],
        };
        
        let binary = to_binary(&module).unwrap();
        let parsed = from_binary(&binary).unwrap();
        assert_eq!(module, parsed);
    }
    
    // ============ Comparison Tests ============
    
    #[test]
    fn test_json_vs_binary_size() {
        let module = Module {
            name: Some("test".into()),
            decls: vec![
                Decl::Let(LetDecl {
                    name: "x".into(),
                    type_: Some(Type::Prim(PrimType::I64)),
                    value: Expr::Lit(Literal::Int(42)),
                }),
            ],
        };
        
        let json_size = to_json(&module).unwrap().len();
        let binary_size = to_binary(&module).unwrap().len();
        
        // Binary should be smaller
        assert!(binary_size < json_size);
    }
    
    #[test]
    fn test_json_compact_vs_pretty() {
        let expr = Expr::Tuple(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
        ]);
        
        let _pretty = expr_to_json(&expr).unwrap();
        
        // Compact version should be smaller
        let module = Module {
            name: None,
            decls: vec![],
        };
        let compact = to_json_compact(&module).unwrap();
        let pretty_mod = to_json(&module).unwrap();
        
        assert!(compact.len() <= pretty_mod.len());
    }
    
    // ============ Roundtrip Tests ============
    
    #[test]
    fn test_json_roundtrip_complex_expr() {
        // Complex nested expression
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("f".into())),
            value: Box::new(Expr::Lam(Box::new(Expr::Lam(Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Idx(1)),
            )))))),
            body: Box::new(Expr::App(
                Box::new(Expr::App(
                    Box::new(Expr::Idx(0)),
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
                Box::new(Expr::Lit(Literal::Int(3))),
            )),
        };
        
        // JSON roundtrip
        let json = expr_to_json(&expr).unwrap();
        let from_json = expr_from_json(&json).unwrap();
        assert_eq!(expr, from_json);
        
        // Binary roundtrip
        let binary = expr_to_binary(&expr).unwrap();
        let from_binary = expr_from_binary(&binary).unwrap();
        assert_eq!(expr, from_binary);
    }
    
    #[test]
    fn test_json_all_operators() {
        // Test all binary operators serialize/deserialize
        let ops = vec![
            BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div,
            BinOp::Mod, BinOp::And, BinOp::Or,
            BinOp::Lt, BinOp::Gt, BinOp::Leq, BinOp::Geq,
            BinOp::Eq, BinOp::Neq, BinOp::PlusMinus,
        ];
        
        for op in ops {
            let expr = Expr::BinOp(
                op.clone(),
                Box::new(Expr::Lit(Literal::Int(1))),
                Box::new(Expr::Lit(Literal::Int(2))),
            );
            
            let json = expr_to_json(&expr).unwrap();
            let parsed = expr_from_json(&json).unwrap();
            assert_eq!(expr, parsed, "Failed for operator {:?}", op);
        }
    }
    
    // ============ Size Estimation Tests ============
    
    #[test]
    fn test_estimate_sizes() {
        let module = Module {
            name: Some("test".into()),
            decls: vec![
                Decl::Let(LetDecl {
                    name: "x".into(),
                    type_: Some(Type::Prim(PrimType::I64)),
                    value: Expr::Lit(Literal::Int(42)),
                }),
            ],
        };
        
        let json_estimate = estimate_json_size(&module);
        let binary_estimate = estimate_binary_size(&module);
        
        let json_actual = to_json_compact(&module).unwrap().len();
        let binary_actual = to_binary(&module).unwrap().len();
        
        // Estimates should be close to actual
        assert_eq!(json_estimate, json_actual);
        assert_eq!(binary_estimate, binary_actual);
    }
    
    // ============ Error Cases ============
    
    #[test]
    fn test_invalid_json() {
        let result = expr_from_json("not valid json");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_invalid_binary() {
        let result = expr_from_binary(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_wrong_json_structure() {
        // Valid JSON but not a valid Expr
        let result = expr_from_json(r#"{"wrong": "structure"}"#);
        assert!(result.is_err());
    }
}

