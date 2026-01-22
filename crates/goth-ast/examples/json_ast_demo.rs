//! JSON AST Demonstration
//!
//! Shows how to use the JSON serialization features

use goth_ast::expr::Expr;
use goth_ast::literal::Literal;
use goth_ast::op::BinOp;
use goth_ast::ser::*;

fn main() {
    println!("=== Goth JSON AST Demonstration ===\n");
    
    // Example 1: Simple literal
    demo_simple_literal();
    
    // Example 2: Binary operation
    demo_binop();
    
    // Example 3: Lambda with capture
    demo_lambda();
    
    // Example 4: Complex nested expression
    demo_complex();
    
    // Example 5: Binary vs JSON size comparison
    demo_size_comparison();
}

fn demo_simple_literal() {
    println!("## Example 1: Simple Literal\n");
    
    let expr = Expr::Lit(Literal::Int(42));
    
    println!("Goth: 42");
    println!("\nJSON AST:");
    let json = expr_to_json(&expr).unwrap();
    println!("{}", json);
    
    // Roundtrip
    let parsed = expr_from_json(&json).unwrap();
    assert_eq!(expr, parsed);
    println!("\n✓ Roundtrip successful\n");
    println!("---\n");
}

fn demo_binop() {
    println!("## Example 2: Binary Operation\n");
    
    // 1 + 2
    let expr = Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Lit(Literal::Int(1))),
        Box::new(Expr::Lit(Literal::Int(2))),
    );
    
    println!("Goth: 1 + 2");
    println!("\nJSON AST:");
    let json = expr_to_json(&expr).unwrap();
    println!("{}", json);
    
    // Roundtrip
    let parsed = expr_from_json(&json).unwrap();
    assert_eq!(expr, parsed);
    println!("\n✓ Roundtrip successful\n");
    println!("---\n");
}

fn demo_lambda() {
    use goth_ast::pattern::Pattern;
    
    println!("## Example 3: Lambda with Capture\n");
    
    // let x = 10 in λ→ ₀ + x
    let expr = Expr::Let {
        pattern: Pattern::Var(Some("x".into())),
        type_: None,
        value: Box::new(Expr::Lit(Literal::Int(10))),
        body: Box::new(Expr::Lam(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),  // Lambda parameter
            Box::new(Expr::Idx(1)),  // x (captured)
        )))),
    };
    
    println!("Goth: let x = 10 in λ→ ₀ + x");
    println!("\nJSON AST (pretty):");
    let json = expr_to_json(&expr).unwrap();
    println!("{}", json);
    
    // Show binary size
    let binary = expr_to_binary(&expr).unwrap();
    println!("\nJSON size: {} bytes", json.len());
    println!("Binary size: {} bytes", binary.len());
    println!("Compression: {:.1}%", 
        (1.0 - binary.len() as f64 / json.len() as f64) * 100.0);
    
    // Roundtrip both
    let from_json = expr_from_json(&json).unwrap();
    let from_binary = expr_from_binary(&binary).unwrap();
    assert_eq!(expr, from_json);
    assert_eq!(expr, from_binary);
    println!("\n✓ Both roundtrips successful\n");
    println!("---\n");
}

fn demo_complex() {
    use goth_ast::pattern::Pattern;
    
    println!("## Example 4: Complex Nested Expression\n");
    
    // let f = λ→ λ→ ₀ + ₁ in f 5 3
    let expr = Expr::Let {
        pattern: Pattern::Var(Some("f".into())),
        type_: None,
        value: Box::new(Expr::Lam(Box::new(Expr::Lam(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),
            Box::new(Expr::Idx(1)),
        )))))),
        body: Box::new(Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::Idx(0)),  // f
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Box::new(Expr::Lit(Literal::Int(3))),
        )),
    };
    
    println!("Goth: let f = λ→ λ→ ₀ + ₁ in f 5 3");
    println!("\nJSON AST (first 500 chars):");
    let json = expr_to_json(&expr).unwrap();
    println!("{}...\n", &json[..json.len().min(500)]);
    
    println!("Full JSON size: {} bytes", json.len());
    
    // Roundtrip
    let parsed = expr_from_json(&json).unwrap();
    assert_eq!(expr, parsed);
    println!("✓ Roundtrip successful\n");
    println!("---\n");
}

fn demo_size_comparison() {
    use goth_ast::pattern::Pattern;
    use goth_ast::decl::{Module, Decl, LetDecl};
    use goth_ast::types::{Type, PrimType};
    
    println!("## Example 5: Size Comparison\n");
    
    let module = Module {
        name: Some("example".into()),
        decls: vec![
            Decl::Let(LetDecl {
                name: "x".into(),
                type_: Some(Type::Prim(PrimType::I64)),
                value: Expr::Lit(Literal::Int(42)),
            }),
            Decl::Let(LetDecl {
                name: "y".into(),
                type_: Some(Type::Prim(PrimType::F64)),
                value: Expr::Lit(Literal::Float(3.14)),
            }),
            Decl::Let(LetDecl {
                name: "z".into(),
                type_: None,
                value: Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Name("x".into())),
                    Box::new(Expr::Name("y".into())),
                ),
            }),
        ],
    };
    
    let json_pretty = to_json(&module).unwrap();
    let json_compact = to_json_compact(&module).unwrap();
    let binary = to_binary(&module).unwrap();
    
    println!("Module with 3 let declarations:");
    println!("\nJSON (pretty):  {} bytes", json_pretty.len());
    println!("JSON (compact): {} bytes", json_compact.len());
    println!("Binary:         {} bytes", binary.len());
    
    println!("\nCompression vs pretty JSON:");
    println!("  Compact: {:.1}%", 
        (1.0 - json_compact.len() as f64 / json_pretty.len() as f64) * 100.0);
    println!("  Binary:  {:.1}%", 
        (1.0 - binary.len() as f64 / json_pretty.len() as f64) * 100.0);
    
    // Roundtrips
    let from_json_pretty = from_json(&json_pretty).unwrap();
    let from_json_compact = from_json(&json_compact).unwrap();
    let from_binary = from_binary(&binary).unwrap();
    
    assert_eq!(module, from_json_pretty);
    assert_eq!(module, from_json_compact);
    assert_eq!(module, from_binary);
    println!("\n✓ All roundtrips successful\n");
}
