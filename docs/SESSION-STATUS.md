# Goth Compiler - Session Status

**Last Updated:** 2026-01-23
**Current Branch:** `claude/goth-language-overview-fOWhO`

## Summary

Working on MIR/LLVM lowering to get `tui_demo.goth` compiling. Made significant progress on multiple issues.

---

## Completed Work

### 1. Non-variant Match Lowering in MIR
- **File:** `goth-mir/src/lower.rs`
- **Issue:** Match expressions on literals/wildcards weren't supported
- **Fix:** Implemented if-else chain approach for literal patterns, variable patterns, and wildcard patterns

### 2. De Bruijn Index Convention Fix
- **File:** `goth-mir/src/lower.rs` (line ~2460)
- **Issue:** MIR was using reverse push order for params, but evaluator uses standard de Bruijn (most recent = 0)
- **Fix:** Changed from reverse to forward push order:
```rust
// Push in FORWARD order so that:
// ₀ = last param (standard de Bruijn, matching evaluator)
for i in 0..param_types.len() {
    fn_ctx.push_local(LocalId::new(i as u32), param_types[i].clone());
}
```

### 3. Named Local Variables (Let Bindings)
- **File:** `goth-mir/src/lower.rs`
- **Issue:** Let bindings with named patterns (`let row = ...`) weren't resolving correctly
- **Fix:** Added `named_locals: HashMap<String, (LocalId, Type)>` to LoweringContext to track name→local mappings

### 4. ArrayFill Support in LLVM Backend
- **Files:** `goth-mir/src/mir.rs`, `goth-mir/src/print.rs`, `goth-llvm/src/emit.rs`
- **Issue:** ArrayFill (`[n]⊢v`) wasn't implemented in LLVM emission
- **Fix:** Added Rhs::ArrayFill to MIR, implemented LLVM emission with loop for filling

### 5. Void Return Type Handling
- **File:** `goth-llvm/src/emit.rs`
- **Issue:** `ret void 0` was generated for void functions, `%result = call void ...` for void calls
- **Fix:**
  - Terminator::Return now checks for void and emits `ret void` without value
  - emit_c_main skips capturing/printing result for void functions

### 6. Sum/Prod Reduce Type Inference
- **File:** `goth-mir/src/lower.rs`
- **Issue:** `UnaryOp(Sum, tensor)` returned tensor type instead of element type
- **Fix:** Added special case for Sum/Prod to extract element type from tensor:
```rust
goth_ast::op::UnaryOp::Sum | goth_ast::op::UnaryOp::Prod => {
    match &op_ty {
        Type::Tensor(_, elem_ty) => (**elem_ty).clone(),
        _ => op_ty.clone(),
    }
}
```

### 7. Type Conversion Primitives
- **File:** `goth-llvm/src/emit.rs`
- **Issue:** `toFloat`, `toInt`, `floor`, `ceil`, `sqrt` called nonexistent runtime functions
- **Fix:** Added inline LLVM emission for these primitives:
  - `toFloat` → `sitofp i64 to double`
  - `toInt` → `fptosi double to i64`
  - Math functions use LLVM intrinsics

### 8. Print Primitive Registration
- **File:** `goth-llvm/src/emit.rs`
- **Issue:** `print` primitive returned early without registering the local, causing UndefinedLocal errors
- **Fix:** Removed early return, falls through to register the local

### 9. Extended Type Support in LLVM
- **File:** `goth-llvm/src/emit.rs`
- Added support for all Type variants in `emit_type`:
  - All PrimType variants (I64, I32, Int, Nat, F64, F32, etc.)
  - Type::Option, Type::Effectful, Type::Interval
  - Type::Forall, Type::Exists, Type::App
  - Type::Variant, Type::Refinement, Type::Uncertain, Type::Hole

---

## Current Issue

### ~~De Bruijn Index Resolution~~ ✅ FIXED (commit ee8a0bb)
The wildcard pattern issue is resolved. vline MIR now correctly shows:
```
_14: I64 = BinOp(Add, _4, Const(1))   // row + 1 ✓
_16: I64 = BinOp(Sub, _6, Const(1))   // len - 1 ✓
```

### ~~Boolean Op Type Inference~~ ✅ FIXED (commit d251d2a)
- Comparison ops (Lt, Gt, Leq, Geq, Eq, Neq) now return Bool type
- Logical ops (And, Or) now return Bool type
- MLIR/LLVM backends updated to handle Bool result types
- Test: `if 3 < 5 then 1 else 0` compiles and runs correctly

### UndefinedLocal in tui_demo.goth
- **Error:** `UndefinedLocal("LocalId(54)")`
- **Location:** LLVM emit during tui_demo compilation
- **Root Cause:** Complex control flow with many locals across blocks
- **Investigation needed:** Track how locals are defined/used across basic blocks

---

## Test Results

### Passing
- Simple expressions: `let x = 5 in x + 1` → 6 ✓
- Float operations: `0.7 × (toFloat 30)` → 21 ✓
- Basic functions with I64 return types ✓
- vline MIR generation ✓
- Wildcard pattern handling ✓

### Failing
- `tui_demo.goth` - Bool type inference for And/Or ops

---

## Key Files Modified

| File | Changes |
|------|---------|
| `goth-mir/src/lower.rs` | Match lowering, de Bruijn fix, named_locals, Sum/Prod types |
| `goth-mir/src/mir.rs` | Added ArrayFill variant |
| `goth-mir/src/print.rs` | ArrayFill display |
| `goth-llvm/src/emit.rs` | ArrayFill, void handling, type conversions, all type variants |
| `goth-mlir/src/emit.rs` | ArrayFill emission |
| `goth-mlir/src/builder.rs` | ArrayFill handling |

---

## Next Steps

1. **Fix UndefinedLocal in tui_demo** - Debug LLVM emit local tracking across blocks
2. **Add more primitive support** - toString, write, and TUI primitives
3. **MLIR Phase 5** - CLI integration, comprehensive testing, error handling

---

## Useful Commands

```bash
# Emit MIR to debug lowering
cargo run -q --bin gothic -- --emit-mir examples/tui_demo.goth

# Emit LLVM IR to check codegen
cargo run -q --bin gothic -- --emit-llvm /tmp/test.goth

# Compile and run
cargo run -q --bin gothic -- /tmp/test.goth -o /tmp/out && /tmp/out
```

---

## Architecture Notes

### De Bruijn Index Convention
- Standard de Bruijn: ₀ = most recently bound
- Function params pushed in order: param0, param1, ..., paramN
- After push, ₀ = last param, ₁ = second-to-last, etc.
- Each let binding shifts all indices by 1

### Named Locals Tracking
- `named_locals` HashMap tracks name→(LocalId, Type)
- Inserted when entering let scope with named pattern
- Removed when exiting scope
- Checked first in Expr::Name lowering before falling back to de Bruijn lookup
