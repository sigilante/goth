# Goth Language Reference v0.1
## Comprehensive Documentation for the Interpreted Implementation

---

## Table of Contents

1. [Introduction](#introduction)
2. [Literals](#literals)
3. [Variables and Bindings](#variables-and-bindings)
4. [Functions](#functions)
5. [Operators](#operators)
6. [Data Structures](#data-structures)
7. [Pattern Matching](#pattern-matching)
8. [Type System](#type-system)
9. [Contracts](#contracts)
10. [Control Flow](#control-flow)
11. [Primitives](#primitives)
12. [Advanced Features](#advanced-features)
13. [REPL Commands](#repl-commands)
14. [Syntax Reference](#syntax-reference)

---

## Introduction

Goth is a functional, statically-typed (when compiled) programming language designed for tensor computation, formal verification, and elegant mathematical expression. This reference documents the interpreted implementation (v0.1).

**Key Features:**
- De Bruijn indices for variable binding
- Unicode mathematical notation
- Dependent types with shape variables
- Runtime contract checking
- Homoiconic representation (code as data)
- First-class tensors with shape tracking

**Current State:**
- ✅ Fully functional interpreter
- ✅ Complete parser with Unicode support
- ✅ Runtime contract checking
- 🔲 Type checker (in development)
- 🔲 Native compilation (planned)

---

## Literals

### Integers
```goth
42
-17
0
1_000_000
```

### Floats
```goth
3.14
-0.5
2.0
1.5e-10
```

### Booleans
```goth
⊤        # true
⊥        # false
true     # ASCII alternative
false    # ASCII alternative
```

### Characters
```goth
'a'
'π'
'∀'
'\n'     # Newline
```

### Strings
```goth
"hello world"
"Goth supports Unicode: αβγ"
"Escape sequences: \n \t \\"
```

---

## Variables and Bindings

### De Bruijn Indices

Goth uses **de Bruijn indices** for local variable references. Variables are accessed by their binding depth:

```goth
λ→ ₀           # Identity function: ₀ refers to the argument
λ→ λ→ ₀        # ₀ refers to inner lambda's argument
λ→ λ→ ₁        # ₁ refers to outer lambda's argument
λ→ λ→ ₀ + ₁    # ₀ + ₁ adds inner and outer arguments
```

**Subscript notation:**
- `₀` = most recently bound variable
- `₁` = second most recent
- `₂` = third most recent, etc.

**Typing subscripts:**
- Unicode: ₀₁₂₃₄₅₆₇₈₉
- ASCII fallback: `_0 _1 _2` etc.

### Named Variables

At the top level, variables use names:

```goth
let x = 5 in x                    # Simple let binding
let (a, b) = (10, 20) in a + b    # Pattern matching
```

### Let Bindings

**Simple let:**
```goth
let x = 5 in x × x
# Result: 25
```

**Sequential bindings (with semicolons):**
```goth
let x ← 5 ;
    y ← x × 2 ;
    z ← y + 1
in x + y + z
# Result: 5 + 10 + 11 = 26
```

**Alternative binding syntax:**
```goth
let x = 5 in x      # Using =
let x ← 5 in x      # Using ← (equivalent)
```

Both `=` and `←` work identically for bindings.

**Scoping:**
```goth
let x ← 10 ;
    y ← x + 5 ;      # x is in scope
    z ← x + y        # x and y are in scope
in x + y + z         # All three in scope
```

### Recursive Bindings

**let rec for mutually recursive definitions:**
```goth
let rec factorial ← λ→ match ₀
              0 → 1
              n → n × factorial(n - 1)
in factorial 5
# Result: 120
```

**Multiple recursive bindings:**
```goth
let rec even ← λ→ match ₀
                0 → ⊤
                n → odd(n - 1) ;
        odd ← λ→ match ₀
                0 → ⊥
                n → even(n - 1)
in even 10
# Result: ⊤
```

**With braces (optional):**
```goth
let rec {
  f ← λ→ g ₀ + 1 ;
  g ← λ→ f ₀ - 1
} in f 5
```

---

## Functions

### Lambda Expressions

**Single argument:**
```goth
λ→ ₀ + 1              # Increment function
λ→ ₀ × ₀              # Square function
λ→ if ₀ > 0 then ₀ else -₀   # Absolute value
```

**Multiple arguments:**
```goth
λ→ λ→ ₀ + ₁           # Two arguments: add them
λ→ λ→ λ→ ₀ + ₁ + ₂    # Three arguments: sum them
```

**Multi-arg syntax (3+ args):**
```goth
λ³→ ₀ + ₁ + ₂         # Three-argument lambda
λ⁴→ ₀ × ₁ × ₂ × ₃     # Four-argument lambda
```

### Function Application

**Simple application:**
```goth
(λ→ ₀ + 1) 5
# Result: 6
```

**Multiple arguments:**
```goth
(λ→ λ→ ₀ + ₁) 3 4
# Result: 7
```

**Partial application:**
```goth
let add = λ→ λ→ ₀ + ₁ in
let add5 = add 5 in
add5 10
# Result: 15
```

**Higher-order functions:**
```goth
let twice = λ→ λ→ ₁ (₁ ₀) in
twice (λ→ ₀ × 2) 5
# Result: 20 (applies function twice)
```

### Function Declarations

**Basic declaration:**
```goth
╭─ square : F → F
╰─ ₀ × ₀

square 5
# Result: 25
```

**Box drawing characters:**
- `╭─` = function start
- `│` = middle lines (for contracts)
- `╰─` = function end (body follows)

**ASCII alternatives:**
```goth
/- square : F -> F
\- _0 × _0
```

**Multi-line declarations:**
```goth
╭─ factorial : F → F
╰─ match ₀
     0 → 1
     n → n × factorial(n - 1)
```

**With preconditions:**
```goth
╭─ safe_div : F → F → F
│  ⊢ ₀ ≠ 0
╰─ ₁ / ₀
```

**With postconditions:**
```goth
╭─ double : F → F
│  ⊨ ₀ = ₁ × 2
╰─ ₀ × 2
```

**Complex example (z-score normalization):**
```goth
╭─ normalize : [n]F → [n]F
│  ⊢ len ₀ > 0
│  ⊨ abs(norm ₀ - sqrt(len ₁)) < 0.0001
╰─ let arr ← ₀ ;
       n ← len arr ;
       μ ← sum arr / n ;
       σ ← sqrt(sum ((arr ↦ (λ→ ₀ - μ)) ↦ (λ→ ₀ × ₀)) / n)
   in (arr ↦ (λ→ ₀ - μ)) ↦ (λ→ ₀ / σ)
```

---

## Operators

### Arithmetic

**Binary operators:**
```goth
5 + 3         # Addition: 8
10 - 4        # Subtraction: 6
6 × 7         # Multiplication: 42
15 / 3        # Division: 5.0
17 mod 5      # Modulo: 2
2 ^ 10        # Exponentiation: 1024
```

**Unary operators:**
```goth
-5            # Negation
abs(-10)      # Absolute value: 10
```

**Unicode alternatives:**
- `×` or `*` for multiplication
- `/` or `÷` for division
- `^` or `**` for exponentiation

### Comparison

```goth
5 = 5         # Equality: ⊤
5 ≠ 3         # Inequality: ⊤
10 > 5        # Greater than: ⊤
3 < 7         # Less than: ⊤
5 ≥ 5         # Greater or equal: ⊤
4 ≤ 10        # Less or equal: ⊤
```

**ASCII alternatives:**
- `≠` or `!=` or `/=` for inequality
- `≥` or `>=` for greater-or-equal
- `≤` or `<=` for less-or-equal

**Three levels of equality:**

| Level | Unicode | ASCII | Semantics |
|-------|---------|-------|-----------|
| Value equality | `=` | `=` | Compare values |
| Structural equality | `≡` | `==` | α-equivalent, ignoring sharing |
| Referential equality | `≣` | `===` | Same node in DAG (reserved) |

### Logical

```goth
⊤ ∧ ⊥         # AND: ⊥
⊤ ∨ ⊥         # OR: ⊤
¬⊤            # NOT: ⊥
```

**ASCII alternatives:**
- `∧` or `&&` or `and`
- `∨` or `||` or `or`
- `¬` or `!` or `not`

### Tensor Operations

**Map (apply function to each element):**
```goth
[1, 2, 3] ↦ (λ→ ₀ × 2)
# Result: [2, 4, 6]

[1, 2, 3, 4] ↦ (λ→ ₀ × ₀)
# Result: [1, 4, 9, 16]
```

**Unicode:** `↦`  
**ASCII:** `-:` or `map`

**Filter (select elements matching predicate):**
```goth
[1, 2, 3, 4, 5] ▸ (λ→ ₀ > 2)
# Result: [3, 4, 5]

[1, 2, 3, 4] ▸ (λ→ ₀ mod 2 = 0)
# Result: [2, 4]
```

**Unicode:** `▸`  
**ASCII:** `|>` or `filter`

**Zip (pair corresponding elements):**
```goth
[1, 2, 3] ⊗ [4, 5, 6]
# Result: [⟨1,4⟩, ⟨2,5⟩, ⟨3,6⟩]
```

**Unicode:** `⊗`  
**ASCII:** `*:` or `zip`

**Concat (join arrays):**
```goth
[1, 2, 3] ⊕ [4, 5, 6]
# Result: [1, 2, 3, 4, 5, 6]
```

**Unicode:** `⊕`  
**ASCII:** `+:` or `++` or `concat`

**Compose (function composition):**
```goth
let f = λ→ ₀ + 1 in
let g = λ→ ₀ × 2 in
let h = f ∘ g in
h 5
# Result: 11 (5 × 2 + 1)
```

**Unicode:** `∘`  
**ASCII:** `.:` or `.`

### Postfix Reduction Operators

**Sum (Σ):**
```goth
[1, 2, 3, 4, 5] Σ
# Result: 15

[10, 20, 30] Σ
# Result: 60
```

**Unicode:** `Σ`  
**ASCII:** `+/` or `sum`

**Product (Π):**
```goth
[1, 2, 3, 4] Π
# Result: 24

[2, 3, 4] Π
# Result: 24
```

**Unicode:** `Π`  
**ASCII:** `*/` or `prod`

**Scan (prefix sums, ⍀):**
```goth
[1, 2, 3, 4] ⍀
# Result: [1, 3, 6, 10]

[10, 20, 30] ⍀
# Result: [10, 30, 60]
```

**Unicode:** `⍀`  
**ASCII:** `\/` or `scan`

**Combining operations:**
```goth
[1, 2, 3] ⊗ [4, 5, 6] Σ
# Dot product: (1×4 + 2×5 + 3×6) = 32

[1, 2, 3] ⊗ [4, 5, 6] Π
# Product of pairs: (1×4 × 2×5 × 3×6) = 17280
```

**Precedence (low to high):**
1. Postfix reduction (Σ, Π, ⍀) - lowest
2. Function application
3. Infix operators (+, ×, etc.)
4. Field access (`.field`) - highest

---

## Data Structures

### Arrays/Tensors

**Literals:**
```goth
[1, 2, 3, 4, 5]
[3.14, 2.71, 1.41]
[⊤, ⊥, ⊤]
```

**Multi-dimensional:**
```goth
[[1, 2], [3, 4]]
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

**Array fill syntax:**
```goth
[3 4 ; 0]
# 3×4 array filled with 0
# Result: [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
```

**Indexing:**
```goth
let arr = [10, 20, 30, 40] in arr[2]
# Result: 30 (0-indexed)
```

**Note:** The `[` must be directly adjacent (no space) to be parsed as indexing. With a space, `f [1,2]` is function application, passing the array `[1,2]` as an argument to `f`.

**Multi-dimensional indexing:**
```goth
let matrix = [[1, 2, 3], [4, 5, 6]] in
matrix[1, 2]
# Result: 6
```

**Operations:**
```goth
len([1, 2, 3, 4])              # Length: 4
shape([[1, 2], [3, 4]])        # Shape: [2, 2]
reverse([1, 2, 3])             # Reverse: [3, 2, 1]
```

### Tuples

**Literals:**
```goth
⟨1, 2⟩
⟨3.14, 2.71, 1.41⟩
⟨⊤, 5, "hello"⟩
```

**ASCII alternative:**
```goth
(1, 2)
(3.14, 2.71, 1.41)
```

**Access by index:**
```goth
let pair = ⟨10, 20⟩ in pair.0
# Result: 10

let triple = ⟨1, 2, 3⟩ in triple.2
# Result: 3
```

**Pattern matching:**
```goth
let (x, y) = ⟨5, 10⟩ in x + y
# Result: 15

match ⟨1, 2, 3⟩
  (a, b, c) → a + b + c
# Result: 6
```

**Nested tuples:**
```goth
⟨⟨1, 2⟩, ⟨3, 4⟩⟩
⟨1, ⟨2, 3⟩, 4⟩
```

### Records

**Named fields:**
```goth
⟨x: 10, y: 20⟩
⟨name: "Alice", age: 30, active: ⊤⟩
```

**Field access:**
```goth
let point = ⟨x: 5.0, y: 10.0⟩ in point.x
# Result: 5.0

let person = ⟨name: "Bob", age: 25⟩ in person.age
# Result: 25
```

**Greek letters and superscripts in field names:**
```goth
let stats = ⟨μ: 10.0, σ: 2.0, σ²: 4.0, n: 100⟩ in stats.σ²
# Result: 4.0

let measurement = ⟨α: 0.5, β: 1.2⟩ in measurement.α
# Result: 0.5
```

**Pattern matching:**
```goth
match ⟨x: 5, y: 10⟩
  ⟨x, y⟩ → x + y
# Result: 15
```

### Variants (Sum Types)

**Construction:**
```goth
⟨Left 5⟩
⟨Right "error"⟩
⟨Some 42⟩
⟨None⟩
```

**Pattern matching:**
```goth
match ⟨Left 10⟩
  ⟨Left x⟩ → x × 2
  ⟨Right msg⟩ → 0

# Result: 20
```

**Option type example:**
```goth
let safe_head = λ→ match ₀
  [] → ⟨None⟩
  [x | rest] → ⟨Some x⟩
in safe_head [1, 2, 3]
# Result: ⟨Some 1⟩
```

---

## Pattern Matching

### Match Expression

**Basic syntax:**
```goth
match expr
  pattern₁ → result₁
  pattern₂ → result₂
  pattern₃ → result₃
```

### Pattern Types

**Literal patterns:**
```goth
match 5
  0 → "zero"
  1 → "one"
  5 → "five"
  _ → "other"
# Result: "five"
```

**Variable patterns:**
```goth
match 42
  x → x × 2
# Result: 84
```

**Wildcard pattern:**
```goth
match anything
  _ → "default"
```

**Tuple patterns:**
```goth
match ⟨10, 20⟩
  (0, 0) → "origin"
  (x, 0) → "x-axis"
  (0, y) → "y-axis"
  (x, y) → "general"
# Result: "general"
```

**Array patterns:**
```goth
match [1, 2, 3]
  [] → "empty"
  [x] → "single"
  [x, y] → "pair"
  [x, y, z] → "triple"
  _ → "many"
# Result: "triple"
```

**Array split patterns:**
```goth
match [1, 2, 3, 4, 5]
  [head | tail] → head
# Result: 1

match [1, 2, 3, 4]
  [x, y | rest] → x + y
# Result: 3
```

**Variant patterns:**
```goth
match ⟨Some 42⟩
  ⟨None⟩ → 0
  ⟨Some x⟩ → x
# Result: 42

match ⟨Left "error"⟩
  ⟨Left msg⟩ → "Error: " ++ msg
  ⟨Right val⟩ → "Success"
# Result: "Error: error"
```

**Record patterns:**
```goth
match ⟨x: 5, y: 10⟩
  ⟨x: 0, y: 0⟩ → "origin"
  ⟨x, y⟩ → x + y
# Result: 15
```

### Examples

**Fibonacci:**
```goth
╭─ fib : F → F
╰─ match ₀
     0 → 0
     1 → 1
     n → fib(n - 1) + fib(n - 2)
```

**List length:**
```goth
╭─ length : [n]α → F
╰─ match ₀
     [] → 0
     [_ | rest] → 1 + length rest
```

**Option unwrapping:**
```goth
╭─ unwrap_or : α? → α → α
╰─ match ₀
     ⟨None⟩ → ₁
     ⟨Some x⟩ → x
```

---

## Type System

### Primitive Types

**Numeric types:**
```goth
I8, I16, I32, I64, I128     # Signed integers
U8, U16, U32, U64, U128     # Unsigned integers
F32, F64                     # Floating point
Int                          # Arbitrary precision integer
Float                        # Arbitrary precision float
```

**Other primitives:**
```goth
Bool                         # Boolean
Char                         # Character
String                       # String
Unit                         # Unit type (no value)
```

**Shorthands:**
```goth
I   # I64
U   # U64
F   # F64
```

### Function Types

**Simple function:**
```goth
F → F                        # Float to Float
I → I → I                    # Curried: Int to Int to Int
(I, I) → I                   # Uncurried: pair of Ints to Int
```

**Unicode arrow:** `→`  
**ASCII arrow:** `->`

**Higher-order:**
```goth
(F → F) → F → F             # Takes function and value, returns value
(α → β) → [n]α → [n]β       # Map type signature
```

### Tensor Types

**Fixed-size tensors:**
```goth
[3]F                         # Vector of 3 floats
[3 4]F                       # 3×4 matrix
[2 3 4]I                     # 2×3×4 tensor of ints
```

**Variable-size (shape variables):**
```goth
[n]F                         # Vector of n floats
[n m]F                       # n×m matrix
[n n]F                       # Square matrix
```

### Tuple Types

```goth
⟨I, F⟩                       # Pair of int and float
⟨F, F, F⟩                    # Triple of floats
⟨α, β, γ⟩                    # Generic triple
```

### Record Types

```goth
⟨x: F, y: F⟩                 # Point record
⟨name: String, age: I⟩       # Person record
⟨μ: F, σ: F, n: I⟩          # Statistics record
```

### Variant Types

```goth
⟨Left α | Right β⟩           # Either type
⟨Some α | None⟩              # Option type
⟨Ok α | Err String⟩          # Result type
```

### Polymorphic Types

**Type variables:**
```goth
α, β, γ                      # Type variables
```

**Forall (universal quantification):**
```goth
∀α. α → α                    # Identity function
∀α β. α → β → α              # Const function
∀n α. [n]α → I               # Length function
```

**ASCII:** `forall α. α → α`

### Option Types

```goth
F?                           # Optional float
[n]I?                        # Optional vector
(α → β)?                     # Optional function
```

### Uncertain Types

**Interval types:**
```goth
F⊢[0..1]                     # Float in range [0, 1]
I⊢[1..100]                   # Int between 1 and 100
```

**Uncertain types (value ± uncertainty):**
```goth
F ± F                        # Float with uncertainty
I ± I                        # Int with uncertainty
```

**Creating uncertain values at runtime:**
```goth
10.5 ± 0.3                  # Value 10.5 with uncertainty 0.3
```

**Automatic uncertainty propagation:**

When uncertain values flow through arithmetic operators and math functions, uncertainty propagates automatically using standard error propagation rules:

| Operation | Propagation Rule |
|-----------|-----------------|
| `(a±δa) + (b±δb)` | δ = √(δa² + δb²) |
| `(a±δa) - (b±δb)` | δ = √(δa² + δb²) |
| `(a±δa) × (b±δb)` | δ = \|a×b\| × √((δa/a)² + (δb/b)²) |
| `(a±δa) / (b±δb)` | δ = \|a/b\| × √((δa/a)² + (δb/b)²) |
| `√(x±δx)` | δ = δx / (2√x) |
| `sin(x±δx)` | δ = \|cos(x)\| × δx |
| `cos(x±δx)` | δ = \|sin(x)\| × δx |
| `exp(x±δx)` | δ = exp(x) × δx |
| `ln(x±δx)` | δ = δx / \|x\| |

**Supported functions:** `+`, `-`, `×`, `/`, `^`, `√`, `exp`, `ln`, `log10`, `log2`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `abs`, `floor`, `ceil`, `round`, `Γ`.

**Example — chained propagation:**
```goth
╭─ main : F64 → F64 → F64 → F64 → (F64 ± F64)
╰─ sin (√(₃ ± ₂) + (₁ ± ₀))
# With inputs 4.0 0.2 1.0 0.1 → 0.1411±0.1107
```

### Refinement Types

**Constrained types:**
```goth
{x : F | x > 0}              # Positive floats
{x : I | x mod 2 = 0}        # Even integers
{arr : [n]F | n > 0}         # Non-empty arrays
```

**Syntax:**
```goth
{variable : BaseType | predicate}
```

### Effect Types

> **Aspirational:** Effect annotations are parsed and stored in the AST but not enforced by the type checker or evaluator. They currently serve as documentation. See `docs/EFFECT-SYSTEM-ROADMAP.md`.

```goth
□                            # Pure (no effects)
◇io                          # I/O effects
◇mut                         # Mutable state
◇exn                         # Exceptions
□ ∪ ◇io                      # Pure or I/O
```

### Type Ascription

**Annotating expressions:**
```goth
5 : I                        # 5 as integer
3.14 : F                     # 3.14 as float
[1, 2, 3] : [3]I            # Array with type annotation
```

**In function declarations:**
```goth
╭─ add : F → F → F
╰─ ₀ + ₁

╭─ map : ∀α β. (α → β) → [n]α → [n]β
╰─ ₁ ↦ ₀
```

**In let bindings:**
```goth
let x : F = 5.0 in x × 2
let arr : [5]I = [1, 2, 3, 4, 5] in len arr
```

---

## Contracts

### Preconditions (⊢)

**Checked before function execution:**
```goth
╭─ sqrt_safe : F → F
│  ⊢ ₀ ≥ 0
╰─ sqrt ₀
```

**Multiple preconditions:**
```goth
╭─ divide : F → F → F
│  ⊢ ₀ ≠ 0
│  ⊢ ₁ ≥ 0
╰─ ₁ / ₀
```

**In preconditions:**
- `₀` = last argument
- `₁` = second-to-last argument
- etc.

**Complex preconditions:**
```goth
╭─ bounded_divide : F → F → F
│  ⊢ ₀ ≠ 0
│  ⊢ abs ₁ < 1000
│  ⊢ abs ₀ > 0.001
╰─ ₁ / ₀
```

**Unicode:** `⊢`  
**ASCII:** `|-`

### Postconditions (⊨)

**Checked after function execution:**
```goth
╭─ double : F → F
│  ⊨ ₀ = ₁ × 2
╰─ ₀ × 2
```

**In postconditions:**
- `₀` = result
- `₁` = first argument (shifted)
- `₂` = second argument (shifted)
- etc.

**Multiple postconditions:**
```goth
╭─ abs_value : F → F
│  ⊨ ₀ ≥ 0
│  ⊨ ₀ = ₁ ∨ ₀ = -₁
╰─ if ₀ < 0 then -₀ else ₀
```

**With tolerance (for floating point):**
```goth
╭─ normalize : [n]F → [n]F
│  ⊢ len ₀ > 0
│  ⊨ abs(norm ₀ - sqrt(len ₁)) < 0.0001
╰─ ...
```

**Unicode:** `⊨`  
**ASCII:** `|=`

### Contract Violation

**Precondition violation:**
```goth
╭─ positive_only : F → F
│  ⊢ ₀ > 0
╰─ ₀

positive_only(-5)
# Error: Precondition violated: precondition #1 failed
```

**Postcondition violation:**
```goth
╭─ buggy : F → F
│  ⊨ ₀ > ₁
╰─ ₀ - 1

buggy(5)
# Error: Postcondition violated: postcondition #1 failed
```

### Examples

**Safe division:**
```goth
╭─ safe_div : F → F → F
│  ⊢ ₀ ≠ 0
│  ⊨ abs(₀ × ₁ - ₂) < 0.0001
╰─ ₁ / ₀
```

**Sorted array:**
```goth
╭─ sort : [n]F → [n]F
│  ⊨ len ₀ = len ₁
│  ⊨ is_sorted ₀
╰─ ...
```

**Contract inheritance:**
```goth
╭─ wrapper : F → F
│  ⊢ ₀ > 0
╰─ safe_div ₀ 2
# Inherits safe_div's contracts
```

---

## Control Flow

### If-Then-Else

**Basic syntax:**
```goth
if condition then true_branch else false_branch
```

**Examples:**
```goth
if 5 > 3 then "yes" else "no"
# Result: "yes"

if ⊤ ∧ ⊥ then 1 else 0
# Result: 0
```

**Nested:**
```goth
if x < 0 then
  "negative"
else if x = 0 then
  "zero"
else
  "positive"
```

**As expression:**
```goth
let abs = λ→ if ₀ < 0 then -₀ else ₀ in abs(-5)
# Result: 5
```

### Match (Pattern Matching)

See [Pattern Matching](#pattern-matching) section for comprehensive coverage.

### Recursion

**Direct recursion:**
```goth
╭─ factorial : F → F
╰─ match ₀
     0 → 1
     n → n × factorial(n - 1)
```

**Mutual recursion:**
```goth
let rec even = λ→ match ₀
                0 → ⊤
                n → odd(n - 1) ;
        odd = λ→ match ₀
                0 → ⊥
                n → even(n - 1)
in even 10
```

**Tail recursion:**
```goth
╭─ sum_tail : [n]F → F → F
╰─ match ₀
     [] → ₁
     [x | rest] → sum_tail rest (₁ + x)
```

---

## Primitives

### Arithmetic

```goth
add(5, 3)              # 8
sub(10, 4)             # 6
mul(6, 7)              # 42
div(15, 3)             # 5.0
mod(17, 5)             # 2
neg(-5)                # 5
abs(-10)               # 10
pow(2, 10)             # 1024
```

### Mathematical Functions

```goth
exp(1.0)               # e ≈ 2.718
ln(2.718)              # ≈ 1.0
sqrt(16.0)             # 4.0
sin(3.14159 / 2)       # ≈ 1.0
cos(0.0)               # 1.0
tan(0.785398)          # ≈ 1.0
floor(3.7)             # 3.0
ceil(3.2)              # 4.0
round(3.5)             # 4.0
```

**Unicode alternatives:**
```goth
√16.0                  # Same as sqrt(16.0)
⌊3.7⌋                  # Same as floor(3.7)
⌈3.2⌉                  # Same as ceil(3.2)
```

### Comparison

```goth
eq(5, 5)               # ⊤
neq(5, 3)              # ⊤
lt(3, 7)               # ⊤
gt(10, 5)              # ⊤
leq(5, 5)              # ⊤
geq(10, 5)             # ⊤
```

### Logical

```goth
and(⊤, ⊥)              # ⊥
or(⊤, ⊥)               # ⊤
not(⊤)                 # ⊥
```

### Bitwise Operations

```goth
bitand 255 15          # 15
bitor 240 15           # 255
bitxor 255 170         # 85     (also: ⊻ 255 170)
shl 1 8               # 256
shr 256 4             # 16
```

All bitwise operations are curried: `I64 → I64 → I64`.

### Array/Tensor Operations

```goth
sum([1, 2, 3, 4])              # 10
prod([2, 3, 4])                # 24
len([1, 2, 3, 4, 5])           # 5
shape([[1, 2], [3, 4]])        # [2, 2]
reverse([1, 2, 3])             # [3, 2, 1]
concat([1, 2], [3, 4])         # [1, 2, 3, 4]
⌿ (λ→ λ→ ₁ + ₀) 0 [1, 2, 3]  # 6  (fold/reduce)
```

**Linear algebra:**
```goth
dot([1, 2, 3], [4, 5, 6])              # 32
norm([3.0, 4.0])                        # 5.0
matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
transpose([[1, 2, 3], [4, 5, 6]])
```

### Type Conversions

```goth
toInt(3.14)            # 3
toFloat(42)            # 42.0
toBool(0)              # ⊥
toBool(1)              # ⊤
toChar(65)             # 'A'
```

### I/O

**Print (with newline):**
```goth
print("Hello, world!")
# Prints: Hello, world!\n
# Returns: ()
```

`print` appends a newline. It accepts any value.

**Write to stdout (no newline):**
```goth
"hello" ▷ stdout
# Prints: hello (no trailing newline)
# Returns: ()
```

**Write to stderr:**
```goth
"error message" ▷ stderr
# Prints to stderr: error message
# Returns: ()
```

**Write to file:**
```goth
"file contents" ▷ "/tmp/output.txt"
# Writes the string to the given file path
# Returns: ()
```

**Unicode:** `▷`
**ASCII:** `|>`

**Read bytes from file:**
```goth
⧏ 8 "/dev/urandom"            # read 8 bytes → [8]I64
readBytes 4 "/tmp/data.bin"    # ASCII fallback
```

**Write bytes to file:**
```goth
⧐ [72, 101, 108] "/tmp/out"   # write byte array to file
writeBytes [0, 255] "/tmp/bin" # ASCII fallback
```

`stdout` and `stderr` are built-in stream constants. The `▷` operator dispatches on the right-hand side: a stream value writes to that stream (without a newline), a string value writes to that file path.

**Read from file:**
```goth
◁ "/tmp/input.txt"
# Returns: file contents as String
```

---

## Advanced Features

### Do-Notation

**Monadic operations on arrays:**

```goth
do [1, 2, 3]
  ↦ λ→ ₀ × 2
  ▸ λ→ ₀ > 3
end
# Result: [4, 6]
```

**With let bindings:**
```goth
do [1, 2, 3, 4, 5]
  let x ← ₀ × 2
  ↦ λ→ ₀ + 1
end
# Result: [3, 5, 7, 9, 11]
```

**Operators in do-notation:**
```goth
do [10, 20, 30]
  + 5
  × 2
end
# Result: [30, 50, 70]
```

### Type Ascription (as!)

**Type coercion/assertion:**
```goth
5 as! F                          # Treat 5 as Float
[1, 2, 3] as! [3]I              # Assert array is [3]I
```

**In complex expressions:**
```goth
let x = 42 as! F in x / 2
# Result: 21.0
```

### Custom Operators

**Definition (parsed but not fully implemented):**
```goth
⊙ : (α → β) → (β → γ) → (α → γ)
f ⊙ g = λ→ g(f(₀))
```

### Holes

**Type holes for incomplete code:**
```goth
let incomplete = λ→ ?hole in incomplete
# Used during development
```

### Lazy Evaluation (Thunks)

**Delayed computation:**
```goth
# Implementation detail - transparent to user
# Expressions are evaluated when needed
```

---

## REPL Commands

### Help

```goth
:help
:h
:?
```

Shows available commands and usage.

### Type Information

```goth
:type expr
:t expr
```

Shows the inferred type of an expression (when type checker is available).

### AST Display

```goth
:ast expr
```

Shows the parsed abstract syntax tree.

**Example:**
```goth
goth[0]› :ast λ→ ₀ + 1
Lam(BinOp(Add, Idx(0), Lit(Int(1))))
```

### Load Files

```goth
:load filename.goth
:l filename.goth
```

Loads and executes a Goth source file.

### Clear Environment

```goth
:clear
:c
```

Clears all defined variables and functions.

### Quit

```goth
:quit
:q
```

Exits the REPL.

### Multi-line Input

The REPL supports multi-line input with continuation prompts:

```goth
goth[0]› let x ← 5 ;
.......    y ← x × 2
.......  in x + y
15
```

**Continuation triggers:**
- Unbalanced delimiters: `[`, `(`, `{`, etc.
- Trailing operators: `+`, `×`, etc.
- Keywords without completion: `let` without `in`, `if` without `else`

---

## Syntax Reference

### Keywords

```
let, in, rec, if, then, else, match, do, end
forall, exists, where
true, false, ⊤, ⊥
as
```

### Reserved Symbols

```
λ          # Lambda (not available as identifier)
→          # Arrow (function type, lambda, match arm)
←          # Back arrow (let binding alternative)
⊢          # Precondition
⊨          # Postcondition
```

### Delimiters

```
( )        # Parentheses
[ ]        # Brackets (arrays, tensor types)
{ }        # Braces (records, refinements)
⟨ ⟩        # Angle brackets (tuples, variants)
╭─ │ ╰─    # Box drawing (function declarations)
```

### Subscripts (De Bruijn Indices)

```
₀₁₂₃₄₅₆₇₈₉
```

**ASCII alternative:** `_0 _1 _2 _3 ...`

### Superscripts (Field Names)

```
⁰¹²³⁴⁵⁶⁷⁸⁹
αβγδεζηθικμνξοπρστυφχψω
```

Can be used in identifiers: `.σ²`, `.x⁰`

### Comments

**Not yet implemented!** Currently no comment syntax.

**Planned:**
```goth
# Line comment
{- Block comment -}
```

### Escape Sequences

**In strings:**
```
\n         # Newline
\t         # Tab
\\         # Backslash
\"         # Quote
\r         # Carriage return
```

**In characters:**
```
'\n'       # Newline character
'\t'       # Tab character
```

---

## Complete Examples

### Statistical Functions

**Mean:**
```goth
╭─ mean : [n]F → F
│  ⊢ len ₀ > 0
╰─ sum ₀ / len ₀

mean([1.0, 2.0, 3.0, 4.0, 5.0])
# Result: 3.0
```

**Variance:**
```goth
╭─ variance : [n]F → F
│  ⊢ len ₀ > 0
╰─ let arr ← ₀ ;
       μ ← mean arr ;
       deviations ← arr ↦ (λ→ ₀ - μ) ;
       squared ← deviations ↦ (λ→ ₀ × ₀)
   in sum squared / len arr

variance([1.0, 2.0, 3.0, 4.0, 5.0])
# Result: 2.0
```

**Standard Deviation:**
```goth
╭─ std_dev : [n]F → F
│  ⊢ len ₀ > 0
╰─ sqrt(variance ₀)

std_dev([1.0, 2.0, 3.0, 4.0, 5.0])
# Result: 1.414...
```

**Z-score Normalization:**
```goth
╭─ normalize : [n]F → [n]F
│  ⊢ len ₀ > 0
│  ⊨ abs(sum ₀) < 0.0001
╰─ let arr ← ₀ ;
       n ← len arr ;
       μ ← sum arr / n ;
       σ ← sqrt(sum ((arr ↦ (λ→ ₀ - μ)) ↦ (λ→ ₀ × ₀)) / n)
   in (arr ↦ (λ→ ₀ - μ)) ↦ (λ→ ₀ / σ)

normalize([1.0, 2.0, 3.0, 4.0, 5.0])
# Result: [-1.414..., -0.707..., 0, 0.707..., 1.414...]
```

### List Operations

**Map:**
```goth
╭─ map : ∀α β. (α → β) → [n]α → [n]β
╰─ ₁ ↦ ₀

map (λ→ ₀ × 2) [1, 2, 3, 4]
# Result: [2, 4, 6, 8]
```

**Filter:**
```goth
╭─ filter : ∀α. (α → Bool) → [n]α → [?]α
╰─ ₁ ▸ ₀

filter (λ→ ₀ > 2) [1, 2, 3, 4, 5]
# Result: [3, 4, 5]
```

**Fold (reduce):**
```goth
╭─ foldl : ∀α β. (β → α → β) → β → [n]α → β
╰─ match ₂
     [] → ₁
     [x | xs] → foldl ₀ (₀ ₁ x) xs

foldl (λ→ λ→ ₀ + ₁) 0 [1, 2, 3, 4, 5]
# Result: 15
```

**Reverse:**
```goth
╭─ reverse : [n]α → [n]α
╰─ match ₀
     [] → []
     [x | xs] → concat (reverse xs) [x]

reverse [1, 2, 3, 4]
# Result: [4, 3, 2, 1]
```

### Tree Operations

**Binary tree type:**
```goth
data Tree α = ⟨Leaf | Node α (Tree α) (Tree α)⟩
```

**Tree sum:**
```goth
╭─ tree_sum : Tree F → F
╰─ match ₀
     ⟨Leaf⟩ → 0
     ⟨Node val left right⟩ → val + tree_sum left + tree_sum right
```

**Tree map:**
```goth
╭─ tree_map : ∀α β. (α → β) → Tree α → Tree β
╰─ match ₁
     ⟨Leaf⟩ → ⟨Leaf⟩
     ⟨Node val left right⟩ → 
       ⟨Node (₀ val) (tree_map ₀ left) (tree_map ₀ right)⟩
```

### Matrix Operations

**Matrix addition:**
```goth
╭─ mat_add : [m n]F → [m n]F → [m n]F
╰─ ₀ ↦ (λ→ ₁ ↦ (λ→ ₀ + ₃))
```

**Dot product:**
```goth
╭─ dot : [n]F → [n]F → F
│  ⊢ len ₀ = len ₁
╰─ sum (₀ ⊗ ₁)

dot [1.0, 2.0, 3.0] [4.0, 5.0, 6.0]
# Result: 32.0
```

**Matrix-vector multiplication:**
```goth
╭─ matvec : [m n]F → [n]F → [m]F
╰─ ₁ ↦ (λ→ dot ₀ ₂)
```

### Quicksort

```goth
╭─ quicksort : [n]F → [n]F
╰─ match ₀
     [] → []
     [pivot | rest] →
       let smaller ← filter (λ→ ₀ < pivot) rest ;
           greater ← filter (λ→ ₀ ≥ pivot) rest
       in concat (concat (quicksort smaller) [pivot]) (quicksort greater)

quicksort [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
# Result: [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
```

### Fizzbuzz

```goth
╭─ fizzbuzz : I → String
╰─ match (₀ mod 15, ₀ mod 3, ₀ mod 5)
     (0, _, _) → "FizzBuzz"
     (_, 0, _) → "Fizz"
     (_, _, 0) → "Buzz"
     _ → toString ₀

[1..100] ↦ fizzbuzz
```

---

## Language Status Summary

### ✅ Fully Implemented

- Complete lexer with Unicode support
- Full parser with all syntax features
- Tree-walking interpreter
- De Bruijn index resolution
- Runtime contract checking (pre/postconditions)
- Pattern matching (all forms)
- Higher-order functions
- Recursive functions (let rec)
- Sequential let bindings (with `;`)
- Multi-line REPL
- Greek letters in identifiers
- Postfix reduction operators
- All primitive operations
- Array/tensor operations
- Tuple and record types
- Variant types
- Function declarations with box syntax
- Type annotations (parsed)

### 🔲 In Progress

- Type checker (Priority 7 - started by Opus)
- Static type inference
- Type error messages

### 🔲 Planned

- Refinement type solving (needs Z3)
- Effect type checking
- Dependent shape inference
- Polymorphism (let-generalization)
- Native code compilation (MLIR → LLVM)
- Comment syntax
- Module system
- Standard library
- Package manager
- Language server protocol (LSP)
- Debugger
- Profiler
- Optimizations

### 📝 Notes

- All syntax is parsed but not all features are type-checked
- Contracts are runtime-only (no static proving yet)
- Shape variables are tracked but not unified
- Effect annotations are parsed but not enforced
- Refinement types are parsed but predicates not solved

---

## Quick Start

### Installation

```bash
cd goth
cargo build --release
```

### Running the REPL

```bash
cargo run --bin goth
```

### Hello World

```goth
goth[0]› "Hello, Goth!"
"Hello, Goth!"

goth[1]› print("Hello, Goth!")
Hello, Goth!
()
```

### Simple Function

```goth
goth[2]› ╭─ greet : String → String
       . ╰─ "Hello, " ++ ₀ ++ "!"
fn greet : String → String

goth[3]› greet("World")
"Hello, World!"
```

### Working with Arrays

```goth
goth[4]› [1, 2, 3, 4, 5] ↦ (λ→ ₀ × ₀) Σ
55

goth[5]› let squares = [1, 2, 3, 4] ↦ (λ→ ₀ × ₀) in squares
[1, 4, 9, 16]
```

---

## Getting Help

### Documentation

- This reference document
- REPL `:help` command
- Example files in `examples/` directory

### Reporting Issues

For bugs, feature requests, or questions, please file an issue on the GitHub repository.

### Community

Join the Goth community to discuss language features, share examples, and get help.

---

## Version History

**v0.1 (Current) - Interpreted Implementation**
- Complete interpreter with all core features
- Full syntax support
- Runtime contract checking
- Multi-line REPL
- Greek letter support
- Sequential let bindings
- Fixed expression parser (operators after application)
- Fixed primitives in function bodies
- All documented features working

**Next: v0.2 - Type Checker**
- Static type checking
- Type inference
- Better error messages

**Future: v1.0 - Compiled Implementation**
- Native code generation
- Optimizations
- Full standard library
- Module system

---

## Acknowledgments

Goth is designed for elegant mathematical expression, formal verification, and tensor computation. Special thanks to all contributors and the functional programming community for inspiration.

---

**End of Goth Language Reference v0.1**

*Last updated: January 2026*
*For the latest version, see the official Goth repository*
