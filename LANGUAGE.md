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
| Integer | `42`, `-7` | Arbitrary precision (`ℤ`), backed by i128 |
| Float | `3.14`, `2.718` | 64-bit (`F64`) |
| Boolean | `true`, `false` or `⊤`, `⊥` | |
| Char | `'a'`, `'\n'` | Unicode |
| String | `"hello"` | UTF-8 |
| Unit | `()` or `⟨⟩` | |
| Array | `[1, 2, 3]` | Homogeneous |
| Tuple | `⟨1, "hi", true⟩` | Heterogeneous |
| Record | `⟨x: 10, y: 20⟩` | Named fields |

Comments use `#`:
```goth
# This is a comment
x + y  # Inline comment
```

---

## De Bruijn Indices

Goth uses de Bruijn indices instead of named variables. This eliminates shadowing ambiguity.

| Syntax | Meaning |
|--------|---------|
| `₀` or `_0` | First parameter / most recent binding |
| `₁` or `_1` | Second parameter / one binding out |
| `₂` or `_2` | Third parameter / two bindings out |

```goth
λ→ ₀              # identity: returns its argument
λ→ λ→ ₀ + ₁      # add: ₀ is second (inner) arg, ₁ is first (outer) arg
```

In multi-arg function declarations, `₀` = **last** argument (most recently bound):
```goth
╭─ sub : ℤ → ℤ → ℤ
╰─ ₁ - ₀            # ₁ = first arg, ₀ = second arg (sub 10 3 = 7)
```

For single-arg functions, `₀` is the sole argument:
```goth
╭─ square : ℤ → ℤ
╰─ ₀ × ₀            # ₀ = the argument
```

Let bindings shift indices:
```goth
╭─ example : ℤ → ℤ
╰─ let x = ₀ × 2 in    # ₀ = argument
   let y = ₀ + 1 in    # ₀ = x, ₁ = argument
   ₀ + ₁               # ₀ = y, ₁ = x, ₂ = argument
```

---

## Types

### Primitive Types

| Type | Description |
|------|-------------|
| `ℤ` | Integer (backed by i128 at runtime) |
| `ℕ` | Natural number (non-negative integer, i128 at runtime) |
| `F` | Generic float (resolves to F64) |
| `I64` | 64-bit signed integer (i128 at runtime) |
| `I32` | 32-bit signed integer (i128 at runtime) |
| `F64` | 64-bit float |
| `F32` | 32-bit float |
| `Bool` | Boolean |
| `Char` | Unicode character |
| `String` | UTF-8 string (tensor of Char) |
| `()` | Unit type |

All integer types (`ℤ`, `ℕ`, `I64`, `I32`, etc.) are stored as i128 at runtime. The type distinctions exist for documentation and type checking but are not enforced by the interpreter. The standard library primarily uses `ℤ`, `F`, and `Bool` for polymorphic signatures.

### Composite Types

```goth
ℤ → ℤ                    # Function type
ℤ → ℤ → ℤ               # Curried function
[n]ℤ                     # Vector of n integers
[m n]F64                 # m×n matrix of floats
⟨ℤ, Bool⟩               # Tuple (product)
⟨x: ℤ, y: ℤ⟩            # Record
```

### Type Variables

Lowercase names are type variables:
```goth
α → α                    # Polymorphic identity
[n]α → [n]α              # Shape-preserving map
```

---

## Tensor Shapes

Shapes are part of the type system and checked at compile time.

```goth
[3]ℤ                     # Exactly 3 integers
[2 3]F64                 # 2×3 matrix
[n]ℤ                     # Vector of unknown length n
[m n]F64                 # m×n matrix (shape variables)
```

Shape checking catches dimension mismatches:
```goth
# Error: Dimension mismatch at position 0: expected 5, found 3
╭─ bad : [3]ℤ → [5]ℤ
╰─ ₀
```

Shape variables unify:
```goth
# OK: input and output have same shape
╭─ double : [n]ℤ → [n]ℤ
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
╭─ square : ℤ → ℤ
╰─ ₀ × ₀

╭─ add : ℤ → ℤ → ℤ
╰─ ₀ + ₁

╭─ sumSquares : [n]ℤ → ℤ
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
╭─ factorial : ℤ → ℤ
╰─ if ₀ ≤ 1 then 1 else ₀ × factorial (₀ - 1)
```

### Let Bindings

```goth
let x = expr in body          # Bind x, use in body (= or ← accepted)
let x : Type = expr in body   # With type annotation
```

Type annotations enable compile-time shape checking:
```goth
let v : [3]F64 = [1.0, 2.0, 3.0] in v    # OK
let v : [5]F64 = [1.0, 2.0, 3.0] in v    # Error: shape mismatch
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
| `±` | `+-` | Uncertainty |

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

### Bitwise

| Unicode | ASCII | Operation | Example |
|---------|-------|-----------|---------|
| | `bitand` | Bitwise AND | `bitand 255 15` → `15` |
| | `bitor` | Bitwise OR | `bitor 240 15` → `255` |
| `⊻` | `bitxor` | Bitwise XOR | `⊻ 255 170` → `85` |
| | `shl` | Shift left | `shl 1 8` → `256` |
| | `shr` | Shift right | `shr 256 4` → `16` |

All bitwise operations are curried functions on integer (`ℤ`) values.

### Structural Equality

| Unicode | ASCII | Operation | Example |
|---------|-------|-----------|---------|
| `≡` | `===` | Structural equality | `[1,2] ≡ [1,2]` → `⊤` |

### Array Operations

| Unicode | ASCII | Operation | Example |
|---------|-------|-----------|---------|
| `↦` | `-:` | Map | `arr ↦ (λ→ ₀ × 2)` |
| `▸` | `\|>` | Filter | `arr ▸ (λ→ ₀ > 5)` |
| `⤇` | `>>=` | Bind (flatmap) | `[1,2] ⤇ (λ→ [₀, ₀×2])` → `[1,2,2,4]` |
| `⌿` | `fold` | Fold/reduce | `⌿ (λ→ λ→ ₁ + ₀) 0 [1,2,3]` → `6` |
| `⍀` | `scan` | Scan (prefix sums) | `⍀ (λ→ λ→ ₁ + ₀) 0 [1,2,3]` → `[1,3,6]` |
| `Σ` | `+/` | Sum | `Σ [1, 2, 3]` |
| `Π` | `*/` | Product | `Π [1, 2, 3]` |
| `⊕` | `++` | Concatenate | `[1,2] ⊕ [3,4]` → `[1,2,3,4]` |
| `⊗` | `zip` | Zip (pair elements) | `[1,2] ⊗ [3,4]` → `[⟨1,3⟩, ⟨2,4⟩]` |

### Function Operations

| Unicode | ASCII | Operation | Example |
|---------|-------|-----------|---------|
| `∘` | `.:` | Compose | `(f ∘ g) x` = `f (g x)` |

### I/O Operations

| Unicode | ASCII | Operation | Example |
|---------|-------|-----------|---------|
| `▷` | `>>` | Write | `"hello" ▷ "file.txt"` |
| `◁` | `<<` | Read | `◁ "file.txt"` |

### Indexing

```goth
arr[0]                   -- First element (bracket must be adjacent, no space)
arr[i]                   -- Element at index i
tuple.0                  -- First tuple field
record.x                 -- Named field x
```

**Note:** The `[` must be directly adjacent to the expression (no space) to be parsed as indexing. With a space, `f [1,2]` is function application (passing the array `[1,2]` to `f`).

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

╭─ getOrDefault : ℤ → ℤ
╰─ match (Some ₀) {
     Some x → x
     None → 0
   }
```

---

## Effects

> **Note:** Effect annotations are parsed into the AST but not yet enforced. Functions with I/O side effects work with or without annotations. See `docs/EFFECT-SYSTEM-ROADMAP.md` for the enforcement plan.

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

Pure functions (no effects) need no annotation. Currently, effect annotations serve as documentation — the runtime does not reject effectful operations in unannotated functions.

---

## Modules

### Imports

```goth
use "prelude.goth"
use "../stdlib/option.goth"
```

### Module Files

Each `.goth` file is a module. The `use` declaration takes a string path (relative to the importing file) and inlines all declarations from that file into the current namespace.

---

## Primitives Reference

### Sequence Generation

| Name | Signature | Description |
|------|-----------|-------------|
| `ι`, `iota` | `ℤ → [n]ℤ` | `[0, 1, ..., n-1]` |
| `range` | `ℤ → ℤ → [m]ℤ` | `[start, ..., end-1]` |

### Reductions

| Name | Signature | Description |
|------|-----------|-------------|
| `Σ`, `sum` | `[n]α → α` | Sum elements |
| `Π`, `prod` | `[n]α → α` | Product elements |
| `len` | `[n]α → ℤ` | Array length |

### Transformations

| Name | Signature | Description |
|------|-----------|-------------|
| `↦` (map) | `[n]α → (α → β) → [n]β` | Apply to each |
| `▸` (filter) | `[n]α → (α → Bool) → [m]α` | Keep matching |
| `⌿`, `fold` | `(α → β → α) → α → [n]β → α` | Left fold with accumulator |
| `reverse` | `[n]α → [n]α` | Reverse order |
| `take` | `ℤ → [n]α → [m]α` | Take first k elements |
| `drop` | `ℤ → [n]α → [m]α` | Drop first k elements |
| `⧺` | `String → String → String` | Concatenate strings |
| `⊕` | `[n]α → [m]α → [p]α` | Concatenate arrays |

### Linear Algebra

| Name | Signature | Description |
|------|-----------|-------------|
| `·`, `dot` | `[n]F64 → [n]F64 → F64` | Dot product |
| `norm` | `[n]F64 → F64` | Euclidean norm |
| `matmul` | `[m n]F64 → [n p]F64 → [m p]F64` | Matrix multiply |
| `⍉`, `transpose` | `[m n]α → [n m]α` | Transpose |

### Math Functions

| Name | Signature |
|------|-----------|
| `√`, `sqrt` | `F64 → F64` |
| `exp` | `F64 → F64` |
| `ln` | `F64 → F64` |
| `sin`, `cos`, `tan` | `F64 → F64` |
| `floor`, `ceil`, `round` | `F64 → F64` |
| `abs` | Numeric → Numeric |

### Uncertainty Propagation

Goth supports first-class uncertain values using the `±` operator. When uncertain values flow through arithmetic and math functions, uncertainty propagates automatically.

**Creating uncertain values:**
```goth
10.5 ± 0.3              -- 10.5 with uncertainty 0.3
```

**Automatic propagation through operations:**
```goth
(10.0 ± 0.3) + (20.0 ± 0.4)    -- 30 ± 0.5  (additive: δ = √(δa² + δb²))
(5.0 ± 0.1) × (3.0 ± 0.2)     -- 15 ± 1.04  (relative: quadrature sum)
√(9.0 ± 0.3)                   -- 3 ± 0.05   (derivative: δ/(2√x))
sin (1.0 ± 0.1)                -- 0.841 ± 0.054  (|cos x| × δx)
```

Supported functions: `+`, `-`, `×`, `/`, `^`, `√`, `exp`, `ln`, `log10`, `log2`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `abs`, `floor`, `ceil`, `round`, `Γ`.

### Type Conversions

| Name | Signature | Description |
|------|-----------|-------------|
| `toInt` | `α → ℤ` | Convert to integer |
| `toFloat` | `α → F64` | Convert to float |
| `toChar` | `ℤ → Char` | Integer to character |
| `parseInt` | `String → ℤ` | Parse string as integer |
| `parseFloat` | `String → F64` | Parse string as float |

### Bitwise Operations

| Name | Signature | Description |
|------|-----------|-------------|
| `bitand` | `ℤ → ℤ → ℤ` | Bitwise AND |
| `bitor` | `ℤ → ℤ → ℤ` | Bitwise OR |
| `⊻`, `bitxor` | `ℤ → ℤ → ℤ` | Bitwise XOR |
| `shl` | `ℤ → ℤ → ℤ` | Shift left (0..127) |
| `shr` | `ℤ → ℤ → ℤ` | Shift right (0..127) |

### I/O

| Name | Signature | Description |
|------|-----------|-------------|
| `print` | `α → ()` | Print to stdout (with newline) |
| `readLine` | `() → String` | Read line from stdin |
| `readFile` | `String → String` | Read file contents |
| `writeFile` | `String → String → ()` | Write content to file path |
| `⧏`, `readBytes` | `ℤ → String → [n]ℤ` | Read n bytes from file as byte array |
| `⧐`, `writeBytes` | `[n]ℤ → String → ()` | Write byte array to file |
| `▷` | `String → String → ()` | Write operator: `"content" ▷ "path"` |
| `toString` | `α → String` | Convert value to string |
| `strConcat`, `⧺` | `String → String → String` | Concatenate strings |

---

## Examples

### Factorial

```goth
╭─ factorial : ℤ → ℤ
╰─ if ₀ ≤ 1 then 1 else ₀ × factorial (₀ - 1)

╭─ main : () → ℤ
╰─ factorial 10
```

### Sum of Squares

```goth
╭─ main : () → ℤ
╰─ Σ ((ι 10) ↦ λ→ ₀ × ₀)
```

### Filter Even Numbers

```goth
╭─ main : () → ℤ
╰─ Σ ((ι 20) ▸ λ→ (₀ % 2) = 0)
```

### Cross-Function Calls

```goth
╭─ square : ℤ → ℤ
╰─ ₀ × ₀

╭─ main : () → ℤ
╰─ square 9
```

### Enum Pattern Match

```goth
enum Option α where Some α | None

╭─ main : () → ℤ
╰─ match (Some 42) {
     Some x → x × 2
     None → 0
   }
```

---

## Standard Library

The `stdlib/` directory contains reusable modules imported via `use`:

```goth
use "stdlib/random.goth"
use "stdlib/math.goth"
```

### Modules

| Module | Description |
|--------|-------------|
| `prelude.goth` | Core combinators (`id`, `const`, `flip`, `compose`) |
| `list.goth` | List/array operations |
| `math.goth` | Math functions with uncertainty propagation |
| `option.goth` | Generic Option type (`some`/`none`, `mapOpt`, `flatMapOpt`, ...) |
| `result.goth` | Generic Result type (`ok`/`err`, `mapRes`, `flatMapRes`, ...) |
| `string.goth` | String manipulation |
| `io.goth` | I/O utilities |
| `canvas.goth` | Canvas/graphics operations |
| `tui.goth` | Terminal UI operations |
| `random.goth` | Seeded PRNG (xorshift64) with state-passing pattern |
| `crypto.goth` | SHA-256, MD5, BLAKE3, Base64 encode/decode |
| `json.goth` | JSON parser/serializer (`parseJson`, `toJson`, accessors) |

### JSON

The `json.goth` module provides a pure Goth JSON parser and serializer. JSON values are represented as tagged 2-tuples `⟨tag, payload⟩` where the tag discriminates the type:

```goth
use "stdlib/json.goth"

╭─ main : () → String
╰─ let r = parseJson "{\"name\":\"Goth\",\"version\":1}"
   in if r.0 then toJson r.1 else "error: " ⧺ r.2
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `jsonNull` | `() → Json` | Construct null (tag 0) |
| `jsonBool` | `Bool → Json` | Construct boolean (tag 1) |
| `jsonNum` | `F64 → Json` | Construct number (tag 2) |
| `jsonStr` | `String → Json` | Construct string (tag 3) |
| `jsonArr` | `[n]Json → Json` | Construct array (tag 4) |
| `jsonObj` | `[n]⟨String, Json⟩ → Json` | Construct object (tag 5) |
| `parseJson` | `String → ⟨Bool, Json, String⟩` | Parse JSON string; `⟨⊤, val, ""⟩` or `⟨⊥, 0, errMsg⟩` |
| `toJson` | `Json → String` | Serialize to compact JSON |
| `jsonGet` | `String → Json → ⟨Bool, Json⟩` | Lookup key in object |
| `jsonIndex` | `ℤ → Json → ⟨Bool, Json⟩` | Index into array |
| `jsonKeys` | `Json → [n]String` | Object keys |
| `jsonValues` | `Json → [n]Json` | Object values |
| `jsonLen` | `Json → ℤ` | Array/object length |
| `jsonType` | `Json → String` | Type name string |
| `isNull`, `isBool`, `isNum`, `isStr`, `isArr`, `isObj` | `Json → Bool` | Type predicates |
| `asBool`, `asNum`, `asStr`, `asArr`, `asObj` | `Json → α` | Payload extractors (unsafe) |

### Random Number Generation

The `random.goth` module provides a seeded PRNG using xorshift64. All RNG functions return `⟨value, nextSeed⟩` tuples for explicit state threading:

```goth
use "stdlib/random.goth"

╭─ main : ℤ → [n]F64
╰─ let seed = entropy ⟨⟩
   in let ⟨vals, _⟩ = randFloats ₁ seed
   in vals
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `entropy` | `() → I64` | Seed from `/dev/urandom` |
| `bytesToSeed` | `[8]I64 → I64` | Pack 8 bytes into seed |
| `xorshift64` | `I64 → I64` | Raw state transition |
| `randFloat` | `I64 → ⟨F64, I64⟩` | Uniform float in [0, 1) |
| `randFloatRange` | `F64 → F64 → I64 → ⟨F64, I64⟩` | Uniform float in [lo, hi) |
| `randInt` | `I64 → I64 → I64 → ⟨I64, I64⟩` | Uniform integer in [lo, hi] |
| `randBool` | `F64 → I64 → ⟨Bool, I64⟩` | Boolean with probability p |
| `randNormal` | `I64 → ⟨F64, I64⟩` | Standard normal (Box-Muller) |
| `randGaussian` | `F64 → F64 → I64 → ⟨F64, I64⟩` | Normal with mean and stddev |
| `randFloats` | `I64 → I64 → ⟨[n]F64, I64⟩` | Bulk uniform floats |
| `randInts` | `I64 → I64 → I64 → I64 → ⟨[n]I64, I64⟩` | Bulk uniform integers |
| `randNormals` | `I64 → I64 → ⟨[n]F64, I64⟩` | Bulk normal values |

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
