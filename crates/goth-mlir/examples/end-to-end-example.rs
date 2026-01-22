//! End-to-End Goth Compilation Example
//!
//! Demonstrates the complete pipeline:
//! Goth source → AST → MIR → MLIR

use goth_ast::expr::Expr;
use goth_ast::literal::Literal;
use goth_ast::op::BinOp;
use goth_mir::lower_expr;
use goth_mlir::emit_program;

fn main() {
    println!("=== GOTH COMPILER: END-TO-END PIPELINE ===\n");
    
    // Example 1: Simple arithmetic
    example_1();
    
    // Example 2: Let binding
    example_2();
    
    // Example 3: Lambda
    example_3();
    
    println!("\n🔥 FULL COMPILATION PIPELINE WORKING! 🔥");
}

fn example_1() {
    println!("## Example 1: Simple Arithmetic\n");
    println!("Goth: 1 + 2\n");
    
    // Build AST
    let expr = Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Lit(Literal::Int(1))),
        Box::new(Expr::Lit(Literal::Int(2))),
    );
    
    // Lower to MIR
    let mir = lower_expr(&expr).unwrap();
    println!("MIR Functions: {}", mir.functions.len());
    println!("MIR Statements: {}", mir.functions[0].body.stmts.len());
    
    // Emit MLIR
    let mlir = emit_program(&mir).unwrap();
    println!("\nMLIR Output:");
    println!("{}", mlir);
    
    println!("---\n");
}

fn example_2() {
    use goth_ast::pattern::Pattern;
    
    println!("## Example 2: Let Binding\n");
    println!("Goth: let x = 5 in x + 1\n");
    
    // Build AST: let x = 5 in x + 1
    let expr = Expr::Let {
        pattern: Pattern::Var(Some("x".into())),
        type_: None,
        value: Box::new(Expr::Lit(Literal::Int(5))),
        body: Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),  // x
            Box::new(Expr::Lit(Literal::Int(1))),
        )),
    };
    
    // Lower to MIR
    let mir = lower_expr(&expr).unwrap();
    println!("MIR Statements: {}", mir.functions[0].body.stmts.len());
    
    // Emit MLIR
    let mlir = emit_program(&mir).unwrap();
    println!("\nMLIR Output:");
    println!("{}", mlir);
    
    println!("---\n");
}

fn example_3() {
    println!("## Example 3: Lambda Expression\n");
    println!("Goth: λ→ ₀ + 1\n");
    
    // Build AST: λ→ ₀ + 1
    let expr = Expr::Lam(Box::new(Expr::BinOp(
        BinOp::Add,
        Box::new(Expr::Idx(0)),  // parameter
        Box::new(Expr::Lit(Literal::Int(1))),
    )));
    
    // Lower to MIR
    let mir = lower_expr(&expr).unwrap();
    println!("MIR Functions generated: {}", mir.functions.len());
    println!("  - Lambda function: {}", mir.functions[0].name);
    println!("  - Main function: {}", mir.functions[1].name);
    
    // Emit MLIR
    let mlir = emit_program(&mir).unwrap();
    println!("\nMLIR Output:");
    println!("{}", mlir);
    
    println!("---\n");
}
