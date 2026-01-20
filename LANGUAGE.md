# Goth Language Specification

**Goth** is a functional programming language with first-class tensor operations, algebraic data types, and effect tracking. It uses Unicode glyphs for a compact, mathematical syntax.

## Contents

1. [Literals](#literals)
2. [De Bruijn Indices](#de-bruijn-indices)
3. [Types](#types)
4. [Tensor Shapes](#tensor-shapes)
5. [Functions](#functions)
6. [Operators](#operators)
7. [Enums and Pattern Matching](#enums-and-pattern-matching)
8. [Effects](#effects)
9. [Modules](#modules)
10. [Primitives Reference](#primitives-reference)
11. [Examples](#examples)

---

## Literals

| Type | Example | Notes |
|------|---------|-------|
| Integer | `42`, `-7` | 64-bit signed |
| Float | `3.14`, `2.718` | 64-bit |
| Boolean | `true`, `false` or `⊤`, `⊥` | |
| Char | `'a'`, `'\n'` | Unicode |
| String | `"hello"` | UTF-8 |
| Unit | `()` or `⟨⟩` | |
| Array | `[1, 2, 3]` | Homogeneous |
| Tuple | `⟨1, "hi", true⟩` | Heterogeneous |
| Record | `⟨x: 10, y: 20⟩` | Named fields |

Comments use `--` or `#`:
```goth
-- This is a comment
# So is this
```

---

## De Bruijn Indices

Goth uses de Bruijn indices instead of named variables. This eliminates shadowing ambiguity.

| Syntax | Meaning |
|--------|---------|
| `₀` or `_0` | Innermost binding |
| `₁` or `_1` | One level out |
| `₂` or `_2` | Two levels out |

```goth
λ→ ₀              -- identity: returns its argument
λ→ λ→ ₁ + ₀      -- add: ₁ is first arg, ₀ is second
```

In function declarations, arguments bind left-to-right:
```goth
╭─ add : I64 → I64 → I64
╰─ ₀ + ₁            -- ₀ = first arg, ₁ = second arg
```

Let bindings shift indices:
```goth
╭─ example : I64 → I64
╰─ let x ← ₀ × 2 in    -- ₀ = argument
   let y ← ₁ + 1 in    -- ₀ = x, ₁ = argument
   ₀ + ₁               -- ₀ = y, ₁ = x
```

---

## Types

### Primitive Types

| Type | Description |
|------|-------------|
| `I64` | 64-bit signed integer |
| `I32` | 32-bit signed integer |
| `F64` | 64-bit float |
| `F32` | 32-bit float |
| `Bool` | Boolean |
| `Char` | Unicode character |
| `()` | Unit type |

### Composite Types

```goth
I64 → I64                -- Function type
I64 → I64 → I64          -- Curried function
[n]I64                   -- Vector of n integers
[m n]F64                 -- m×n matrix of floats
⟨I64, Bool⟩              -- Tuple (product)
⟨x: I64, y: I64⟩         -- Record
```

### Type Variables

Lowercase names are type variables:
```goth
α → α                    -- Polymorphic identity
[n]α → [n]α              -- Shape-preserving map
```

---

## Tensor Shapes

Shapes are part of the type system and checked at compile time.

```goth
[3]I64                   -- Exactly 3 integers
[2 3]F64                 -- 2×3 matrix
[n]I64                   -- Vector of unknown length n
[m n]F64                 -- m×n matrix (shape variables)
```

Shape checking catches dimension mismatches:
```goth
-- Error: Dimension mismatch at position 0: expected 5, found 3
╭─ bad : [3]I64 → [5]I64
╰─ ₀
```

Shape variables unify:
```goth
-- OK: input and output have same shape
╭─ double : [n]I64 → [n]I64
╰─ ₀ ↦ (λ→ ₀ × 2)
```

---

## Functions

### Declaration Syntax

```goth
╭─ name : Type
╰─ body
```

ASCII alternative: `/-` for `╭─`, `\-` for `╰─`

### Examples

```goth
╭─ square : I64 → I64
╰─ ₀ × ₀

╭─ add : I64 → I64 → I64
╰─ ₀ + ₁

╭─ sumSquares : [n]I64 → I64
╰─ Σ (₀ ↦ (λ→ ₀ × ₀))
```

### Lambda Expressions

```goth
λ→ body                  -- Single argument
λ→ λ→ body              -- Two arguments (curried)
```

ASCII: `\->` for `λ→`

### Recursive Functions

Functions can call themselves by name:
```goth
╭─ factorial : I64 → I64
╰─ if ₀ ≤ 1 then 1 else ₀ × factorial (₀ - 1)
```

### Let Bindings

```goth
let x ← expr in body     -- Bind x, use in body
let x ← a; y ← b in c    -- Sequential bindings
```

---

## Operators

### Arithmetic

| Unicode | ASCII | Operation |
|---------|-------|-----------|
| `+` | `+` | Addition |
| `-` | `-` | Subtraction |
| `×` | `*` | Multiplication |
| `/` | `/` | Division |
| `%` | `%` | Modulo |
| `^` | `^` | Power |

### Comparison

| Unicode | ASCII | Operation |
|---------|-------|-----------|
| `=` or `==` | `==` | Equality |
| `≠` | `/=` | Inequality |
| `<` | `<` | Less than |
| `>` | `>` | Greater than |
| `≤` | `<=` | Less or equal |
| `≥` | `>=` | Greater or equal |

### Logical

| Unicode | ASCII | Operation |
|---------|-------|-----------|
| `∧` | `&&` | And |
| `∨` | `\|\|` | Or |
| `¬` | `!` | Not |

### Array Operations

| Unicode | ASCII | Operation | Example |
|---------|-------|-----------|---------|
| `↦` | `-:` | Map | `arr ↦ (λ→ ₀ × 2)` |
| `▸` | `\|>` | Filter | `arr ▸ (λ→ ₀ > 5)` |
| `Σ` | `+/` | Sum | `Σ [1, 2, 3]` |
| `Π` | `*/` | Product | `Π [1, 2, 3]` |

### Indexing

```goth
arr[0]                   -- First element
arr[i]                   -- Element at index i
tuple.0                  -- First tuple field
record.x                 -- Named field x
```

---

## Enums and Pattern Matching

### Enum Declarations

```goth
enum Color where Red | Green | Blue

enum Option α where
  | Some α
  | None

enum Either α β where
  | Left α
  | Right β
```

### Pattern Matching

```goth
match expr {
  Pattern1 → result1
  Pattern2 → result2
  _ → default
}
```

Patterns:
- `_` — wildcard
- `x` — variable binding
- `42`, `true` — literals
- `⟨a, b⟩` — tuple destructuring
- `[h | t]` — list head/tail
- `Some x` — enum variant
- `None` — nullary variant

### Example

```goth
enum Option α where Some α | None

╭─ getOrDefault : I64 → I64
╰─ match (Some ₀) {
     Some x → x
     None → 0
   }
```

---

## Effects

Effects are declared with `◇` annotations:

| Effect | Description |
|--------|-------------|
| `◇io` | Input/output |
| `◇mut` | Mutation |
| `◇rand` | Randomness |

```goth
╭─ greet : () → ()
│  ◇io
╰─ print "Hello!"
```

Pure functions (no effects) need no annotation.

---

## Modules

### Imports

```goth
use stdlib.prelude
use stdlib.option
use mylib.utils
```

### Module Files

Each `.goth` file is a module. The module path corresponds to the file path:
- `stdlib/prelude.goth` → `stdlib.prelude`
- `stdlib/option.goth` → `stdlib.option`

---

## Primitives Reference

### Sequence Generation

| Name | Signature | Description |
|------|-----------|-------------|
| `iota` | `I64 → [n]I64` | `[0, 1, ..., n-1]` |
| `range` | `I64 → I64 → [m]I64` | `[start, ..., end-1]` |

### Reductions

| Name | Signature | Description |
|------|-----------|-------------|
| `sum`, `Σ` | `[n]I64 → I64` | Sum elements |
| `prod`, `Π` | `[n]I64 → I64` | Product elements |
| `min` | `[n]I64 → I64` | Minimum |
| `max` | `[n]I64 → I64` | Maximum |
| `length` | `[n]α → I64` | Array length |

### Transformations

| Name | Signature | Description |
|------|-----------|-------------|
| `map`, `↦` | `[n]α → (α → β) → [n]β` | Apply to each |
| `filter`, `▸` | `[n]α → (α → Bool) → [m]α` | Keep matching |
| `reverse` | `[n]α → [n]α` | Reverse order |

### Linear Algebra

| Name | Signature | Description |
|------|-----------|-------------|
| `dot`, `·` | `[n]F64 → [n]F64 → F64` | Dot product |
| `norm` | `[n]F64 → F64` | Euclidean norm |
| `matmul` | `[m n]F64 → [n p]F64 → [m p]F64` | Matrix multiply |
| `transpose`, `⍉` | `[m n]α → [n m]α` | Transpose |

### Math Functions

| Name | Signature |
|------|-----------|
| `sqrt`, `√` | `F64 → F64` |
| `exp` | `F64 → F64` |
| `ln` | `F64 → F64` |
| `sin`, `cos`, `tan` | `F64 → F64` |
| `floor`, `ceil`, `round` | `F64 → F64` |
| `abs` | `I64 → I64` |

### I/O

| Name | Signature | Description |
|------|-----------|-------------|
| `print` | `α → ()` | Print to stdout |
| `read_line` | `() → String` | Read line from stdin |

---

## Examples

### Factorial

```goth
╭─ factorial : I64 → I64
╰─ if ₀ ≤ 1 then 1 else ₀ × factorial (₀ - 1)

╭─ main : () → I64
╰─ factorial 10
```

### Sum of Squares

```goth
╭─ main : () → I64
╰─ Σ ((iota 10) ↦ (λ→ ₀ × ₀))
```

### Filter Even Numbers

```goth
╭─ main : () → I64
╰─ Σ ((iota 20) ▸ (λ→ (₀ % 2) == 0))
```

### Cross-Function Calls

```goth
╭─ square : I64 → I64
╰─ ₀ × ₀

╭─ main : () → I64
╰─ square 9
```

### Enum Pattern Match

```goth
enum Option α where Some α | None

╭─ main : () → I64
╰─ match (Some 42) {
     Some x → x × 2
     None → 0
   }
```

---

## Compilation

The `gothic` compiler produces native executables via LLVM:

```sh
gothic program.goth -o program
./program
```

The `goth` interpreter runs programs directly:

```sh
goth program.goth
goth -e "Σ [1, 2, 3]"
```
