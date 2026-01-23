//! Roundtrip Validation Example
//!
//! Tests that AST → JSON → AST and AST → Binary → AST preserve equality

use goth_ast::expr::Expr;
use goth_ast::literal::Literal;
use goth_ast::op::BinOp;
use goth_ast::pattern::Pattern;
use goth_ast::ser::*;

fn main() {
    println!("=== AST Roundtrip Validation ===\n");
    
    let test_cases = vec![
        ("Literal int", Expr::Lit(Literal::Int(42))),
        ("Literal float", Expr::Lit(Literal::Float(3.14))),
        ("Literal bool", Expr::Lit(Literal::True)),
        ("Binary operation", binop_expr()),
        ("Lambda", lambda_expr()),
        ("Let binding", let_expr()),
        ("Nested let", nested_let_expr()),
        ("If expression", if_expr()),
        ("Tuple", tuple_expr()),
        ("Array", array_expr()),
        ("Complex expression", complex_expr()),
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (name, expr) in test_cases {
        print!("Testing {:.<40} ", name);
        
        // JSON roundtrip
        match expr_to_json(&expr) {
            Ok(json) => {
                match expr_from_json(&json) {
                    Ok(parsed) => {
                        if expr == parsed {
                            // Binary roundtrip
                            match expr_to_binary(&expr) {
                                Ok(binary) => {
                                    match expr_from_binary(&binary) {
                                        Ok(from_bin) => {
                                            if expr == from_bin {
                                                println!("✓ PASS (JSON: {} B, Binary: {} B)", 
                                                    json.len(), binary.len());
                                                passed += 1;
                                            } else {
                                                println!("✗ FAIL (binary mismatch)");
                                                failed += 1;
                                            }
                                        }
                                        Err(e) => {
                                            println!("✗ FAIL (binary deserialize: {})", e);
                                            failed += 1;
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("✗ FAIL (binary serialize: {})", e);
                                    failed += 1;
                                }
                            }
                        } else {
                            println!("✗ FAIL (JSON mismatch)");
                            failed += 1;
                        }
                    }
                    Err(e) => {
                        println!("✗ FAIL (JSON deserialize: {})", e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                println!("✗ FAIL (JSON serialize: {})", e);
                failed += 1;
            }
        }
    }
    
    println!("\n{} / {} tests passed", passed, passed + failed);
    
    if failed == 0 {
        println!("\n🎉 All roundtrips successful!");
    } else {
        println!("\n⚠️  {} test(s) failed", failed);
        std::process::exit(1);
    }
}

fn binop_expr() -> Expr {
    Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Lit(Literal::Int(1))),
        Box::new(Expr::Lit(Literal::Int(2))),
    )
}

fn lambda_expr() -> Expr {
    // λ→ ₀ + 1
    Expr::Lam(Box::new(Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Idx(0)),
        Box::new(Expr::Lit(Literal::Int(1))),
    )))
}

fn let_expr() -> Expr {
    // let x = 5 in x + 1
    Expr::Let {
        pattern: Pattern::Var(Some("x".into())),
        type_: None,
        value: Box::new(Expr::Lit(Literal::Int(5))),
        body: Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),
            Box::new(Expr::Lit(Literal::Int(1))),
        )),
    }
}

fn nested_let_expr() -> Expr {
    // let x = 5 in let y = x + 3 in y * 2
    Expr::Let {
        pattern: Pattern::Var(Some("x".into())),
        type_: None,
        value: Box::new(Expr::Lit(Literal::Int(5))),
        body: Box::new(Expr::Let {
            pattern: Pattern::Var(Some("y".into())),
            type_: None,
            value: Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(3))),
            )),
            body: Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(2))),
            )),
        }),
    }
}

fn if_expr() -> Expr {
    // if true then 1 else 2
    Expr::If {
        cond: Box::new(Expr::Lit(Literal::True)),
        then_: Box::new(Expr::Lit(Literal::Int(1))),
        else_: Box::new(Expr::Lit(Literal::Int(2))),
    }
}

fn tuple_expr() -> Expr {
    // ⟨1, true, 3.14⟩
    Expr::Tuple(vec![
        Expr::Lit(Literal::Int(1)),
        Expr::Lit(Literal::True),
        Expr::Lit(Literal::Float(3.14)),
    ])
}

fn array_expr() -> Expr {
    // [1, 2, 3]
    Expr::Array(vec![
        Expr::Lit(Literal::Int(1)),
        Expr::Lit(Literal::Int(2)),
        Expr::Lit(Literal::Int(3)),
    ])
}

fn complex_expr() -> Expr {
    // let f = λ→ λ→ ₀ + ₁ in f 5 3
    Expr::Let {
        pattern: Pattern::Var(Some("f".into())),
        type_: None,
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
    }
}
