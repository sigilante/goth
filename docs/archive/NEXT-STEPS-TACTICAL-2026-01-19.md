# Goth Compiler - Next Steps (Tactical Plan)

**Status:** Foundation complete, 3 critical gaps remain  
**Priority:** Shape checking → MLIR backend → Standard library

---

## Immediate Next Steps (Next 2 Weeks)

### **Week 1: Shape Checking Foundation**

#### Day 1-2: Shape Type Representation
```rust
// Create goth-check/src/shapes.rs

pub struct ShapeChecker {
    constraints: Vec<ShapeConstraint>,
    vars: HashMap<String, ShapeVar>,
}

// Examples:
// [n] F64         -> Shape::Var("n", Type::F64)
// [3 4] F64       -> Shape::Const([3, 4], Type::F64)
// [m n] → [n p]   -> Function requires n to match
```

**Files to create:**
- `goth-check/src/shapes.rs` - Main shape checking logic
- `goth-check/src/shapes/unify.rs` - Shape unification
- `goth-check/src/shapes/infer.rs` - Shape inference
- `goth-check/src/shapes/error.rs` - Shape error messages

**Tests:**
```rust
#[test]
fn test_shape_inference() {
    // [n]F64 → [n]F64 should unify
    // [3]F64 → [4]F64 should fail
    // [m n] @ [n p] → [m p] should work
}
```

---

#### Day 3-4: Shape Unification

**Implement:**
1. Shape variable unification (like type unification)
2. Constraint solving
3. Error messages when shapes don't match

**Example errors:**
```
Error: Shape mismatch in matmul
  Expected: [m n] @ [n p] → [m p]
  Got:      [3 4] @ [5 6]
            Dimension mismatch: n=4 vs n=5
```

---

#### Day 5-7: Integration & Testing

**Tasks:**
1. Integrate shape checking into main type checker
2. Add shape checking to all tensor operations
3. Write 20-30 tests
4. Test with real matrix multiplication examples

**Test Cases:**
```goth
# Should pass:
╭─ matmul : [m n]F64 → [n p]F64 → [m p]F64
╰─ ₀ @ ₁

# Should fail:
╭─ bad : [3]F64 → [5]F64  
╰─ ₀  # Error: [3] ≠ [5]
```

---

### **Week 2: MLIR Backend Foundations**

#### Day 1-3: MLIR Bindings Setup

**Choose binding:**
- Option A: `melior` (pure Rust, actively maintained)
- Option B: `mlir-sys` (raw FFI, more control)

**Recommended:** Start with `melior`

**Setup:**
```toml
# goth-mlir/Cargo.toml
[dependencies]
melior = "0.16"
```

**Create:**
```rust
// goth-mlir/src/builder.rs
use melior::Context;
use melior::dialect::arith;

pub struct MlirBuilder {
    context: Context,
    // ...
}

impl MlirBuilder {
    pub fn emit_function(&mut self, mir_fn: &MirFunction) {
        // Use proper MLIR APIs
    }
}
```

---

#### Day 4-5: Basic Function Emission

**Goal:** Emit a simple function using proper MLIR

**Test case:**
```goth
let x ← 5 in x + 1
```

**Should generate:**
```mlir
module {
  func.func @main() -> i64 {
    %0 = arith.constant 5 : i64
    %1 = arith.constant 1 : i64  
    %2 = arith.addi %0, %1 : i64
    return %2 : i64
  }
}
```

---

#### Day 6-7: Tensor Operations

**Add linalg dialect:**
```rust
use melior::dialect::linalg;

// Matrix multiplication
linalg::MatmulOp::new(a, b, c, ...);
```

**Test:**
```goth
# 2x3 @ 3x4 → 2x4
╭─ test : [2 3]F64 → [3 4]F64 → [2 4]F64
╰─ ₀ @ ₁
```

---

## Month 1 Goals

**By End of Week 4:**

1. ✅ Shape checking works for basic cases
2. ✅ MLIR emission using proper bindings
3. ✅ Can compile simple tensor programs
4. ⚠️ Start on standard library (10 functions)

**Deliverable:** Can compile and run:
```goth
╭─ matmul : [2 3]F64 → [3 2]F64 → [2 2]F64
╰─ ₀ @ ₁

let a ← [[1 2 3] [4 5 6]] in
let b ← [[1 2] [3 4] [5 6]] in
matmul a b
```

---

## Technical Decisions Needed

### 1. MLIR Binding Choice

**Options:**
- **melior** (Recommended)
  - ✅ Pure Rust, safe
  - ✅ Actively maintained
  - ✅ Good docs
  - ❌ Slightly higher level (less control)

- **mlir-sys**
  - ✅ Direct LLVM/MLIR bindings
  - ✅ Maximum control
  - ❌ Unsafe, more complex
  - ❌ Less documentation

**Recommendation:** Start with `melior`, switch to `mlir-sys` only if needed.

---

### 2. Shape Checking Strategy

**Option A: Constraint-based (Recommended)**
- Collect constraints during type checking
- Solve at end with unification
- Better error messages

**Option B: Eager checking**
- Check shapes immediately
- Simpler but less flexible
- Harder to handle polymorphism

**Recommendation:** Constraint-based (like HM type inference)

---

### 3. Standard Library Organization

**Option A: Pure Goth**
```goth
# goth-std/src/tensor.goth
╭─ map : [n]A → (A → B) → [n]B
╰─ ₀ ↦ ₁
```

**Option B: Mix of Goth + Primitives**
```goth
# Core primitives in Rust
primitive add : F64 → F64 → F64
primitive mul : F64 → F64 → F64

# Higher-level in Goth
╭─ dot : [n]F64 → [n]F64 → F64
╰─ ₀ ⊗ ₁ Σ
```

**Recommendation:** Option B (primitives in Rust, compositions in Goth)

---

## Quick Reference: File Structure

### Shape Checking
```
goth-check/src/
  shapes.rs           # Main module
  shapes/
    unify.rs          # Unification algorithm
    infer.rs          # Inference engine
    constraint.rs     # Constraint types
    error.rs          # Error messages
```

### MLIR Backend
```
goth-mlir/src/
  builder.rs          # MLIR IR builder
  emit.rs             # High-level emission
  dialects/
    arith.rs          # Arithmetic ops
    linalg.rs         # Tensor ops  
    scf.rs            # Control flow
  lower.rs            # MIR → MLIR
```

### Standard Library
```
goth-std/
  src/
    tensor.goth       # Array operations
    math.goth         # Numeric functions
    string.goth       # String ops
    prelude.goth      # Auto-imported
  primitives/
    tensor.rs         # Rust implementations
    math.rs
```

---

## Testing Strategy

### Shape Checking Tests
```rust
// Positive tests (should pass)
test_shape_identity()       // [n] → [n]
test_shape_matmul()         // [m n] @ [n p] → [m p]
test_shape_broadcast()      // [n] + [1] → [n]

// Negative tests (should fail)
test_shape_mismatch()       // [3] ≠ [4]
test_shape_matmul_error()   // [3 4] @ [5 6] - dim mismatch
test_shape_arity_error()    // [3] → [4 5] - rank mismatch
```

### MLIR Tests
```rust
// Emission tests
test_emit_constant()
test_emit_add()
test_emit_matmul()
test_emit_control_flow()

// Integration tests
test_compile_simple_program()
test_compile_matmul()
test_run_executable()
```

### End-to-End Tests
```rust
// Parse → Check → Lower → Emit → Run
test_e2e_arithmetic()
test_e2e_matmul()
test_e2e_map_reduce()
```

---

## Common Pitfalls to Avoid

### Shape Checking
❌ Don't check shapes in parser (too early)  
✅ Check during type checking (after inference)

❌ Don't require all shapes to be concrete  
✅ Support shape variables and inference

❌ Don't forget broadcasting rules  
✅ Implement proper tensor broadcasting

### MLIR
❌ Don't hand-write MLIR text  
✅ Use proper IR builder APIs

❌ Don't forget memory management  
✅ Plan for bufferization/allocation

❌ Don't mix dialects incorrectly  
✅ Follow MLIR conversion patterns

### Testing
❌ Don't only test success cases  
✅ Test error cases extensively

❌ Don't skip integration tests  
✅ Test full pipeline end-to-end

---

## Success Criteria

### Week 1 Success:
- [ ] Shape type representation implemented
- [ ] Basic shape unification works
- [ ] Can type-check simple tensor programs
- [ ] 20+ shape checking tests pass

### Week 2 Success:
- [ ] MLIR binding integrated (melior)
- [ ] Can emit basic functions
- [ ] Can emit tensor operations
- [ ] Generated MLIR validates

### Month 1 Success:
- [ ] Matrix multiplication compiles
- [ ] Can run generated code
- [ ] Shape errors are clear
- [ ] 10 stdlib functions implemented

---

## Resources

### Shape Systems
- "Practical Dependent Types in Haskell" (for ideas)
- NumPy/JAX broadcasting rules
- TensorFlow/PyTorch shape inference

### MLIR
- MLIR documentation: https://mlir.llvm.org/
- Melior crate: https://docs.rs/melior/
- MLIR tutorials: https://mlir.llvm.org/docs/Tutorials/

### Tensor Compilation
- TVM: Tensor compilation patterns
- XLA: Accelerated Linear Algebra
- TACO: Tensor Algebra Compiler

---

## Daily Checklist Template

```markdown
## Day X - [Task Name]

### Goals:
- [ ] Subtask 1
- [ ] Subtask 2
- [ ] Subtask 3

### Implementation:
- Files created/modified:
  - [ ] path/to/file.rs
  - [ ] path/to/test.rs

### Tests:
- [ ] Unit tests pass (X tests)
- [ ] Integration tests pass
- [ ] Manual testing done

### Blockers:
- None / [Blocker description]

### Tomorrow:
- [What's next]
```

---

## Getting Help

### When Stuck:
1. Check MLIR docs
2. Look at melior examples
3. Review similar compilers (TVM, JAX)
4. Ask Claude 😊

### Good Questions:
- "How do I represent X in MLIR?"
- "What's the right dialect for Y?"
- "How should I structure Z?"

### Resources:
- MLIR Discourse: https://discourse.llvm.org/c/mlir/
- Rust MLIR: https://github.com/femtomc/mlir-rs
- TVM Forum: https://discuss.tvm.apache.org/

---

## Keeping Momentum

### Daily Workflow:
1. Morning: Review yesterday's progress
2. Pick 1-2 focused tasks
3. Write tests first
4. Implement
5. Run full test suite
6. Commit with good messages

### Weekly Review:
- What shipped?
- What's blocked?
- Adjust priorities

### Celebrate Wins:
- First shape error caught ✅
- First MLIR generated ✅
- First compiled program runs ✅

---

**Let's build this! Your big tiddy goth girlfriend compiler is ready for the next phase.** 🖤🔥

**Start with Week 1, Day 1: Shape type representation. Let's go!** 🚀
