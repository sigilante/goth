//! Integration tests for the MIR/MLIR compilation pipeline
//!
//! These tests verify the full pipeline: source → parse → resolve → MIR → MLIR

use goth_parse::prelude::*;
use goth_mir::{lower_expr, lower_module};
use goth_mlir::emit_program;

/// Helper to test the full pipeline from source to MLIR
fn compile_expr_to_mlir(source: &str) -> Result<String, String> {
    // Parse
    let parsed = parse_expr(source).map_err(|e| format!("Parse error: {}", e))?;

    // Resolve (de Bruijn indices)
    let resolved = resolve_expr(parsed);

    // Lower to MIR
    let mir = lower_expr(&resolved).map_err(|e| format!("MIR error: {:?}", e))?;

    // Emit MLIR
    let mlir = emit_program(&mir).map_err(|e| format!("MLIR error: {:?}", e))?;

    Ok(mlir)
}

/// Helper to test module compilation
fn compile_module_to_mlir(source: &str) -> Result<String, String> {
    let parsed = parse_module(source, "test").map_err(|e| format!("Parse error: {}", e))?;
    let resolved = resolve_module(parsed);
    let mir = lower_module(&resolved).map_err(|e| format!("MIR error: {:?}", e))?;
    let mlir = emit_program(&mir).map_err(|e| format!("MLIR error: {:?}", e))?;
    Ok(mlir)
}

// ============ Literal Tests ============

#[test]
fn test_literal_int() {
    let mlir = compile_expr_to_mlir("42").unwrap();
    assert!(mlir.contains("arith.constant 42"));
}

#[test]
fn test_literal_float() {
    let mlir = compile_expr_to_mlir("3.14").unwrap();
    assert!(mlir.contains("arith.constant 3.14"));
}

#[test]
fn test_literal_bool_true() {
    let mlir = compile_expr_to_mlir("true").unwrap();
    assert!(mlir.contains("arith.constant true") || mlir.contains("arith.constant 1"));
}

#[test]
fn test_literal_bool_false() {
    let mlir = compile_expr_to_mlir("false").unwrap();
    assert!(mlir.contains("arith.constant false") || mlir.contains("arith.constant 0"));
}

// ============ Arithmetic Tests ============

#[test]
fn test_add() {
    let mlir = compile_expr_to_mlir("1 + 2").unwrap();
    assert!(mlir.contains("arith.addi") || mlir.contains("arith.add"));
}

#[test]
fn test_sub() {
    let mlir = compile_expr_to_mlir("5 - 3").unwrap();
    assert!(mlir.contains("arith.subi") || mlir.contains("arith.sub"));
}

#[test]
fn test_mul() {
    let mlir = compile_expr_to_mlir("4 × 5").unwrap();
    assert!(mlir.contains("arith.muli") || mlir.contains("arith.mul"));
}

#[test]
fn test_div() {
    let mlir = compile_expr_to_mlir("10 / 2").unwrap();
    assert!(mlir.contains("arith.divi") || mlir.contains("arith.div"));
}

#[test]
fn test_mod() {
    let mlir = compile_expr_to_mlir("10 % 3").unwrap();
    assert!(mlir.contains("arith.remsi") || mlir.contains("arith.rem"));
}

// ============ Let Binding Tests ============

#[test]
fn test_let_binding() {
    let mlir = compile_expr_to_mlir("let x ← 5 in x + 1").unwrap();
    assert!(mlir.contains("arith.constant 5"));
    assert!(mlir.contains("arith.addi") || mlir.contains("arith.add"));
}

#[test]
fn test_nested_let() {
    let mlir = compile_expr_to_mlir("let x ← 5 in let y ← x × 2 in y + 1").unwrap();
    assert!(mlir.contains("arith.constant 5"));
    assert!(mlir.contains("arith.muli") || mlir.contains("arith.mul"));
    assert!(mlir.contains("arith.addi") || mlir.contains("arith.add"));
}

// ============ Control Flow Tests ============

#[test]
fn test_if_expression() {
    let mlir = compile_expr_to_mlir("if true then 1 else 2").unwrap();
    // Should have conditional branch
    assert!(mlir.contains("cf.cond_br") || mlir.contains("scf.if"));
}

#[test]
fn test_if_with_comparison() {
    let mlir = compile_expr_to_mlir("if 5 > 3 then 100 else 0").unwrap();
    assert!(mlir.contains("arith.cmpi") || mlir.contains("arith.cmp"));
}

// ============ Primitive Tests ============

#[test]
fn test_iota() {
    let mlir = compile_expr_to_mlir("iota 5").unwrap();
    assert!(mlir.contains("goth.iota"));
}

#[test]
fn test_iota_unicode() {
    let mlir = compile_expr_to_mlir("⍳ 10").unwrap();
    assert!(mlir.contains("goth.iota"));
}

#[test]
fn test_range() {
    let mlir = compile_expr_to_mlir("range 1 10").unwrap();
    assert!(mlir.contains("goth.range"));
}

#[test]
fn test_sum() {
    let mlir = compile_expr_to_mlir("sum (iota 5)").unwrap();
    assert!(mlir.contains("goth.iota"));
    assert!(mlir.contains("goth.reduce_sum"));
}

#[test]
fn test_sum_unicode() {
    let mlir = compile_expr_to_mlir("Σ (⍳ 5)").unwrap();
    assert!(mlir.contains("goth.iota"));
    assert!(mlir.contains("goth.reduce_sum"));
}

// ============ Tuple Tests ============

#[test]
fn test_tuple_construction() {
    let mlir = compile_expr_to_mlir("⟨1, 2, 3⟩").unwrap();
    assert!(mlir.contains("arith.constant 1"));
    assert!(mlir.contains("arith.constant 2"));
    assert!(mlir.contains("arith.constant 3"));
}

#[test]
fn test_tuple_field_access() {
    let mlir = compile_expr_to_mlir("⟨1, 2⟩.0").unwrap();
    // TupleField emits as builtin.unrealized_conversion_cast with index
    assert!(mlir.contains("[0]") || mlir.contains("unrealized_conversion_cast"));
}

// ============ Array Tests ============

#[test]
fn test_array_literal() {
    let mlir = compile_expr_to_mlir("[1, 2, 3]").unwrap();
    assert!(mlir.contains("tensor.from_elements") || mlir.contains("Array"));
}

// ============ Lambda Tests ============

#[test]
fn test_lambda_simple() {
    let mlir = compile_expr_to_mlir("λ→ ₀ + 1").unwrap();
    // Should generate a lifted function
    assert!(mlir.contains("func.func @lambda"));
}

#[test]
fn test_lambda_application() {
    let mlir = compile_expr_to_mlir("(λ→ ₀ + 1) 5").unwrap();
    assert!(mlir.contains("arith.constant 5"));
}

// ============ Module Tests ============

#[test]
fn test_simple_function() {
    let source = r#"
╭─ double : I → I
╰─ ₀ × 2
"#;
    let mlir = compile_module_to_mlir(source).unwrap();
    assert!(mlir.contains("func.func @double"));
}

#[test]
fn test_function_with_if() {
    let source = r#"
╭─ absolute : I → I
╰─ if ₀ < 0 then 0 - ₀ else ₀
"#;
    let mlir = compile_module_to_mlir(source).unwrap();
    assert!(mlir.contains("func.func @absolute"));
    assert!(mlir.contains("cf.cond_br"));
}

#[test]
fn test_multiple_functions() {
    let source = r#"
╭─ inc : I → I
╰─ ₀ + 1

╭─ dec : I → I
╰─ ₀ - 1
"#;
    let mlir = compile_module_to_mlir(source).unwrap();
    assert!(mlir.contains("func.func @inc"));
    assert!(mlir.contains("func.func @dec"));
}

#[test]
fn test_main_function() {
    let source = r#"
╭─ main : I → I
╰─ ₀ × 2
"#;
    let mlir = compile_module_to_mlir(source).unwrap();
    assert!(mlir.contains("func.func @main"));
}

// ============ Complex Expression Tests ============

#[test]
fn test_triangular_number() {
    // (n × (n + 1)) / 2
    let mlir = compile_expr_to_mlir("let n ← 10 in (n × (n + 1)) / 2").unwrap();
    assert!(mlir.contains("arith.constant 10"));
    assert!(mlir.contains("arith.muli") || mlir.contains("arith.mul"));
    assert!(mlir.contains("arith.divi") || mlir.contains("arith.div"));
}

#[test]
fn test_digital_root_formula() {
    // 1 + ((n - 1) % (base - 1))
    let source = "let n ← 123 in let base ← 10 in 1 + ((n - 1) % (base - 1))";
    let mlir = compile_expr_to_mlir(source).unwrap();
    assert!(mlir.contains("arith.constant 123"));
    assert!(mlir.contains("arith.constant 10"));
}

// ============ Error Cases ============

#[test]
fn test_unbound_variable_error() {
    // This should fail - x is not defined
    let result = compile_expr_to_mlir("x + 1");
    assert!(result.is_err());
}

#[test]
fn test_parse_error() {
    let result = compile_expr_to_mlir("1 + + 2");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Parse error"));
}

// ============ MIR Structure Tests ============

#[test]
fn test_mir_if_blocks() {
    let parsed = parse_expr("if true then 1 else 2").unwrap();
    let resolved = resolve_expr(parsed);
    let mir = lower_expr(&resolved).unwrap();

    // Main function should have blocks
    let main = mir.functions.iter().find(|f| f.name == "main").unwrap();
    assert!(!main.blocks.is_empty(), "If expression should generate blocks");
    assert_eq!(main.blocks.len(), 3, "Should have then, else, and join blocks");
}

#[test]
fn test_mir_lambda_lifting() {
    let parsed = parse_expr("let f ← λ→ ₀ + 1 in f 5").unwrap();
    let resolved = resolve_expr(parsed);
    let mir = lower_expr(&resolved).unwrap();

    // Should have at least 2 functions: lambda_0 and main
    assert!(mir.functions.len() >= 2, "Lambda should be lifted to separate function");
    assert!(mir.functions.iter().any(|f| f.name.starts_with("lambda")));
}
