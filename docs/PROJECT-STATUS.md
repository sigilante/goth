# Goth Compiler - Project Status

**Date:** 2026-01-23
**Codebase:** ~30,000 lines Rust across 8 crates
**Tests:** 523 test functions

---

## Executive Summary

**Core compiler pipeline: 75% complete**

The Goth compiler has a working end-to-end pipeline from source code through MIR to both MLIR and LLVM IR. Shape checking is implemented. The interpreter works. Native compilation is functional for many programs.

**What works:**
```bash
# Parse, type check, interpret
echo 'let x <- 5 in x + 1' | goth -e -
# -> 6

# Compile to native executable
goth examples/factorial.goth -o factorial && ./factorial

# Emit MLIR with optimization
goth --emit mlir -O2 program.goth
```

**Active work:**
- De Bruijn index resolution edge cases in complex nested let bindings
- TUI demo compilation (tui_demo.goth)

---

## Crate Status

### 1. **goth-ast** (4,147 lines, 57 tests) - COMPLETE

Core AST definitions, types, patterns, operators, and pretty printing.

- Expression AST (literals, lambdas, let, if, match, operations)
- Full type system (primitives, tensors, functions, tuples, variants, quantifiers)
- Pattern matching (wildcards, variables, literals, constructors, guards)
- Shape types with dimension expressions
- Pretty printer (AST -> Goth source)
- JSON/binary serialization

### 2. **goth-parse** (4,040 lines, 105 tests) - COMPLETE

Lexer and parser from Goth source to AST.

- Logos-based tokenization with Unicode operators
- Recursive descent parser
- De Bruijn index resolution
- Module/import loading

### 3. **goth-check** (3,968 lines, 66 tests) - WORKING

Type checking, inference, and shape checking.

- Hindley-Milner type inference
- Type unification with substitutions
- **Shape checking** - constraint-based solver with:
  - Dimension unification (constants, variables, expressions)
  - Shape constraint collection and solving
  - Rich error messages for mismatches
  - Occurs check for infinite dimensions
- Builtin type signatures

### 4. **goth-eval** (2,148 lines, 69 tests) - WORKING

Tree-walking interpreter for testing and REPL.

- Full expression evaluation
- Closure semantics
- Pattern matching
- **60+ primitive functions:**
  - Math: add, sub, mul, div, mod, neg, abs, exp, ln, sqrt, sin, cos, tan, pow, floor, ceil, round
  - Comparison: eq, neq, lt, gt, leq, geq
  - Logic: and, or, not
  - Bitwise: bitand, bitor, bitxor/⊻, shl, shr
  - Tensor: sum, prod, len, shape, reverse, concat, dot, norm, matmul, transpose, iota, range, take, drop, index, fold/⌿
  - String: toString, chars, strConcat, lines, words, bytes, strEq, startsWith, endsWith, contains
  - Type: toInt, toFloat, toBool, toChar, parseInt, parseFloat
  - I/O: print, write, flush, readLine, readKey, sleep, readFile, writeFile, readBytes/⧏, writeBytes/⧐
  - TUI: rawModeEnter, rawModeExit
- **Standard library** (`stdlib/`): 10 modules including random number generation (xorshift64 PRNG)

### 5. **goth-mir** (3,291 lines, 50 tests) - WORKING

Mid-level IR for optimization and lowering.

- SSA-based representation
- AST -> MIR lowering with de Bruijn elimination
- Closure conversion and lambda lifting
- Named local variable tracking
- Match expression lowering (variants + literals)
- ArrayFill support

### 6. **goth-mlir** (9,087 lines, 172 tests) - WORKING

MLIR code generation with dialect support.

**Dialects:**
- `arith` - Arithmetic operations
- `func` - Function definitions
- `cf` - Control flow (branches, conditionals)
- `scf` - Structured control flow (for, while, if)
- `tensor` - Tensor operations
- `linalg` - Linear algebra
- `memref` - Memory references
- `llvm` - LLVM dialect for lowering
- `goth` - Custom Goth operations (iota, reduce, map, filter)

**Passes:**
- `lower_goth` - Lower goth dialect to standard dialects
- `bufferize` - Convert tensors to memrefs
- `lower_llvm` - Lower to LLVM dialect
- `optimize` - Optimization passes (O0-O3)

### 7. **goth-llvm** (1,646 lines, 4 tests) - WORKING

Direct LLVM IR emission for native compilation.

- MIR -> LLVM IR translation
- Primitive function emission
- Type conversion (tensor types, primitives)
- Runtime function integration
- Void return handling
- C main wrapper generation

### 8. **goth-cli** (1,529 lines) - FUNCTIONAL

Command-line interface and REPL.

```bash
goth                        # REPL
goth program.goth           # Run file
goth -e "1 + 2"            # Evaluate expression
goth --emit mir program.goth    # Emit MIR
goth --emit mlir program.goth   # Emit MLIR
goth --emit llvm program.goth   # Emit LLVM IR
goth program.goth -o out        # Compile to native
```

---

## Pipeline Diagram

```
Source (.goth)
     |
     v
+------------+
|   PARSER   |  goth-parse (4,040 lines)
+------------+
     |
     v
+------------+
|    AST     |  goth-ast (4,147 lines)
+------------+
     |
     +------------------+
     |                  |
     v                  v
+------------+    +------------+
| TYPE CHECK |    | INTERPRET  |  goth-eval (2,148 lines)
| + SHAPES   |    +------------+
+------------+
goth-check (3,968 lines)
     |
     v
+------------+
|    MIR     |  goth-mir (3,291 lines)
+------------+
     |
     +------------------+
     |                  |
     v                  v
+------------+    +------------+
|   MLIR     |    | LLVM IR    |  goth-llvm (1,646 lines)
+------------+    +------------+
goth-mlir (9,087 lines)    |
     |                     v
     v              +------------+
  [passes]          |  NATIVE    |
     |              +------------+
     v
+------------+
| LLVM (via  |
|  mlir)     |
+------------+
```

---

## What's Working

### Shape Checking
```goth
# This correctly fails:
let bad : [3]F64 -> [5]F64 in \-> _0
# Error: Shape mismatch [3] vs [5]

# This correctly infers n:
let id : [n]F64 -> [n]F64 in \-> _0
# When called with [10]F64, n = 10
```

### Native Compilation
```bash
$ goth examples/factorial.goth -o factorial
$ ./factorial
120
```

### Tensor Operations
```goth
sum (iota 10)           # -> 45
map (iota 5) (\-> _0 + 1)  # -> [1, 2, 3, 4, 5]
filter (iota 10) (\-> _0 > 5)  # -> [6, 7, 8, 9]
```

---

## Known Limitations

### De Bruijn Index Edge Cases
Complex nested let bindings with wildcards can resolve to wrong locals:
```goth
let a = ... in
let _ = ... in    # Wildcard shifts indices
let b = ... in
# _0 may not resolve correctly here
```

**Status:** Under investigation in SESSION-STATUS.md

### String/Char in MLIR
String and Char types not fully supported in MLIR backend. Works in interpreter.

### Some Math Functions
`sin`, `cos`, `tan`, `abs` may not emit correctly in all backends.

---

## Remaining Work

### High Priority
1. Fix de Bruijn resolution edge cases
2. Complete TUI demo compilation
3. String type in MLIR backend

### Medium Priority
4. More optimization passes (inlining, DCE)
5. Better error messages with source locations
6. Module system improvements

### Future
7. LSP server for IDE support
8. Package manager
9. Debugger integration

---

## Files Structure

```
goth/
├── crates/
│   ├── goth-ast/      4,147 lines   57 tests   COMPLETE
│   ├── goth-parse/    4,040 lines  105 tests   COMPLETE
│   ├── goth-check/    3,968 lines   66 tests   WORKING
│   ├── goth-eval/     2,148 lines   69 tests   WORKING
│   ├── goth-mir/      3,291 lines   50 tests   WORKING
│   ├── goth-mlir/     9,087 lines  172 tests   WORKING
│   ├── goth-llvm/     1,646 lines    4 tests   WORKING
│   └── goth-cli/      1,529 lines    0 tests   FUNCTIONAL
│
├── examples/          Example programs
├── docs/              Documentation
└── LANGUAGE.md        Language reference

TOTAL: ~30,000 lines, 523 tests
```

---

## Session Continuity

For current work status, see `docs/SESSION-STATUS.md`.

For archived historical status documents, see `docs/archive/`.
