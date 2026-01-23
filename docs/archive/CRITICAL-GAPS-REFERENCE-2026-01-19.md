# Goth Compiler - Critical Gaps Quick Reference

**TL;DR: 2 things stand between you and a working compiler:**

---

## ✅ ~~GAP 1: Shape Checking~~ (RESOLVED)

**Status:** Shape checking now works! Use `--check` flag or REPL.

**Example (now correctly caught):**
```goth
# This now correctly fails:
╭─ bad : [3]F64 → [5]F64
╰─ ₀
# → Type error: Shape mismatch: expected [5], found [3]

# Typed let expressions also check shapes:
let x : [5]F64 = [1.0, 2.0, 3.0] in x
# → Type error: Shape mismatch: expected [5], found [3]
```

**What Works:**
- ✅ Shape unification with variables (`[n]F64 → [n]F64`)
- ✅ Concrete shape checking (`[3] + [5]` fails)
- ✅ Type annotations in let expressions
- ✅ Function signature shape checking
- ✅ 69 type checker tests passing

---

## 🔴 GAP 2: MLIR Backend (CRITICAL)

**Problem:** Hand-crafted text generation, not real MLIR

**Current Hack:**
```rust
// This is bad:
write!(f, "%{} = arith.addi %{}, %{}", dst, src1, src2)
```

**Should Be:**
```rust
// Use proper MLIR:
use melior::dialect::arith;

let add_op = arith::AddIOp::new(
    context,
    src1,
    src2,
    location,
);
```

**What's Missing:**
- Proper MLIR C API bindings (use `melior` crate)
- Linalg dialect for tensor ops
- SCF dialect for control flow
- Memory management (bufferization)
- Type lowering

**Fix Estimate:** 3-4 weeks  
**Impact:** HIGH - Can't compile without it  
**Priority:** #2

---

## 🔴 GAP 3: Standard Library (CRITICAL)

**Problem:** Zero stdlib functions, can't write real programs

**What's Missing:**
```goth
# These don't exist yet:

# Tensor ops
map    : [n]A → (A → B) → [n]B
fold   : [n]A → (A → A → A) → A → A
scan   : [n]A → (A → A → A) → A → [n]A
zip    : [n]A → [n]B → [n]⟨A, B⟩

# Math
sin, cos, tan, exp, log, sqrt
abs, min, max, floor, ceil

# String
length, concat, split, join

# I/O  
print, read_line, read_file, write_file
```

**Structure Needed:**
```
goth-std/
  src/
    tensor.goth      # Array operations
    math.goth        # Numeric functions
    string.goth      # String ops
    io.goth          # Input/output
    prelude.goth     # Auto-imported
```

**Fix Estimate:** 2-3 weeks  
**Impact:** HIGH - Can't write programs without it  
**Priority:** #3

---

## Quick Comparison: What Works vs What Doesn't

### ✅ WORKS RIGHT NOW:

```goth
# Simple expressions
1 + 2
let x ← 5 in x + 1
λ→ ₀ + 1

# Pattern matching
match x of
  Just y → y
  Nothing → 0

# Type checking (basic)
let f : I64 → I64
f ← λ→ ₀ + 1
```

**Pipeline:**
1. Parse ✅
2. Type check ✅ (basic)
3. Interpret ✅
4. Emit MLIR ✅ (hacky)

---

### ❌ DOESN'T WORK:

```goth
# Shape checking
╭─ matmul : [m n]F64 → [n p]F64 → [m p]F64
╰─ ₀ @ ₁  # ❌ Shapes not checked!

# Compile to native
$ goth program.goth -o program
# ❌ No LLVM backend

# Use stdlib
let xs ← [1 2 3]
xs ↦ sqrt  # ❌ sqrt doesn't exist
```

**Missing:**
1. Shape checking ❌
2. Native compilation ❌
3. Standard library ❌

---

## The Fix (Prioritized)

### Fix #1: Shape Checking (Week 1-2)

**Create:**
- `goth-check/src/shapes.rs`
- `goth-check/src/shapes/unify.rs`
- `goth-check/src/shapes/infer.rs`

**Implement:**
1. Shape type representation
2. Shape unification
3. Shape inference
4. Error messages

**Test with:**
```goth
# Should pass:
╭─ id : [n]F64 → [n]F64
╰─ ₀

# Should fail:
╭─ bad : [3]F64 → [4]F64
╰─ ₀  # Error: Shape mismatch [3] vs [4]
```

---

### Fix #2: MLIR Backend (Week 3-4)

**Replace:**
- All of `goth-mlir/src/emit.rs`

**Use:**
```toml
[dependencies]
melior = "0.16"
```

**Implement:**
```rust
use melior::Context;
use melior::dialect::{arith, linalg, func};

pub struct MlirBuilder {
    context: Context,
    // ...
}

impl MlirBuilder {
    pub fn emit_add(&mut self, a: Value, b: Value) -> Value {
        arith::AddIOp::new(
            &self.context,
            a,
            b,
            self.location(),
        ).result(0)
    }
    
    pub fn emit_matmul(&mut self, a: Value, b: Value) -> Value {
        linalg::MatmulOp::new(
            a,
            b,
            // ...
        )
    }
}
```

---

### Fix #3: Standard Library (Week 5-6)

**Create:**
```
goth-std/
  Cargo.toml
  src/
    lib.rs
    tensor.goth       # Start here
    math.goth
    prelude.goth
  primitives/
    tensor.rs         # Rust impls
```

**Implement (Phase 1 - 10 functions):**
```goth
# tensor.goth
╭─ map : [n]A → (A → B) → [n]B
╰─ ₀ ↦ ₁

╭─ fold : [n]A → (A → A → A) → A → A
╰─ ₀ Σ  # For now, just sum

# math.goth (link to C math lib)
primitive sqrt : F64 → F64
primitive sin  : F64 → F64
primitive cos  : F64 → F64
primitive exp  : F64 → F64
primitive log  : F64 → F64
```

---

## Success Metrics

### After Fix #1 (Shape Checking):
```bash
$ goth -e "let bad : [3]F64 → [4]F64 in λ→ ₀"
Error: Shape mismatch
  Expected: [4]
  Got: [3]
```
**Can type-check tensor programs!** ✅

---

### After Fix #2 (MLIR):
```bash
$ goth matmul.goth -o matmul
$ ./matmul
[[14, 32], [32, 77]]
```
**Can compile to native!** ✅

---

### After Fix #3 (Stdlib):
```goth
let xs ← [1.0 2.0 3.0 4.0]
let ys ← xs ↦ sqrt
print ys  # [1.0, 1.414..., 1.732..., 2.0]
```
**Can write real programs!** ✅

---

## Minimal Working Example (Target)

**Goal:** Make this work end-to-end:

```goth
# matmul.goth
╭─ matmul : [2 3]F64 → [3 2]F64 → [2 2]F64
│  ⊨ shape result = [2 2]
╰─ ₀ @ ₁

╭─ main : ◇IO ()
╰─ 
  let a ← [[1.0 2.0 3.0]
           [4.0 5.0 6.0]]
  let b ← [[1.0 2.0]
           [3.0 4.0]
           [5.0 6.0]]
  let c ← matmul a b
  print c
```

**Compile and run:**
```bash
$ goth matmul.goth -o matmul
✓ Parsed successfully
✓ Type checked (shapes verified!)
✓ Lowered to MIR
✓ Emitted MLIR
✓ Compiled with LLVM
✓ Linked successfully

$ ./matmul
[[22.0, 28.0], [49.0, 64.0]]
```

---

## Effort Summary

| Gap                | Effort      | Priority | Impact |
|--------------------|-------------|----------|--------|
| Shape Checking     | 1-2 weeks   | #1       | HIGH   |
| MLIR Backend       | 3-4 weeks   | #2       | HIGH   |
| Standard Library   | 2-3 weeks   | #3       | HIGH   |
| **TOTAL**          | **6-9 weeks** | -      | -      |

**With focused work: ~2 months to MVP**

---

## One-Sentence Summary

**Fix shape checking (2 weeks), replace MLIR text with proper bindings (4 weeks), add 50 stdlib functions (3 weeks) = working compiler (9 weeks).**

---

## Next Command to Run

```bash
cd goth/crates/goth-check
cargo new --lib src/shapes
```

Then create `src/shapes.rs`:
```rust
//! Tensor shape checking and inference

use goth_ast::types::Type;

pub struct ShapeChecker {
    // Start here!
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_shape_equality() {
        // First test!
    }
}
```

**Let's go!** 🚀🖤
