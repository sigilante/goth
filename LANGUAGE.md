# Goth Language Overview

**Goth** is a functional programming language with first-class support for tensor operations, dependent types, and effect tracking. It combines elements from APL/J (array programming), Haskell (type system), and ML (pattern matching) with a distinctive visual syntax using Unicode glyphs.

## Table of Contents

1. [Basic Syntax](#basic-syntax)
2. [De Bruijn Indices](#de-bruijn-indices)
3. [Types](#types)
4. [Tensor Shapes](#tensor-shapes)
5. [Functions](#functions)
6. [Operators](#operators)
7. [Effects](#effects)
8. [Pattern Matching](#pattern-matching)
9. [Primitives Reference](#primitives-reference)

---

## Basic Syntax

### Literals

| Type | Unicode | ASCII | Example |
|------|---------|-------|---------|
| Integer | - | - | `42`, `-7` |
| Float | - | - | `3.14`, `2.718` |
| Boolean | `⊤`, `⊥` | `true`, `false` | `⊤` |
| Char | - | - | `'a'`, `'\n'` |
| String | - | - | `"hello"` |
| Unit | `⟨⟩` | `<\|>` | `⟨⟩` |

### Arrays and Tuples

```goth
# Array (tensor) literal
[1, 2, 3, 4, 5]

# Tuple
⟨1, "hello", ⊤⟩    # or <|1, "hello", true|>

# Record (labeled tuple)
⟨x: 10, y: 20⟩
```

### Comments

```goth
# This is a comment (to end of line)
```

---

## De Bruijn Indices

Goth uses **de Bruijn indices** for variable binding instead of named variables. This eliminates issues with variable shadowing and alpha-equivalence.

| Unicode | ASCII | Meaning |
|---------|-------|---------|
| `₀` | `_0` | Innermost binding |
| `₁` | `_1` | One level out |
| `₂` | `_2` | Two levels out |
| ... | `_n` | n levels out |

```goth
# Lambda that returns its argument
λ→ ₀           # \-> _0

# Lambda that adds two arguments
λ→ λ→ ₁ + ₀   # outer arg is ₁, inner arg is ₀
```

### Index Binding Rules

**In function declarations** with multiple arguments, indices are bound in order:
```goth
╭─ f : A → B → C
╰─ ...          # ₀ = first arg (A), ₁ = second arg (B)
```

**In nested lambdas**, the innermost lambda's argument is ₀:
```goth
λ→ λ→ body    # ₀ = second/inner arg, ₁ = first/outer arg
```

**Important: Let bindings shift indices.** After `let x ← expr in body`, within `body`:
- `x` is bound at ₀
- All previous bindings shift up by 1

```goth
╭─ f : ⟨I, I⟩ → I
╰─ let a ← ₀.0 in   # ₀ = the tuple argument
   let b ← ₁.1 in   # After first let: ₀ = a, ₁ = original tuple
   a + b            # After second let: ₀ = b, ₁ = a, ₂ = tuple
```

---

## Types

### Primitive Types

| Type | Unicode | Description |
|------|---------|-------------|
| `F64` | - | 64-bit floating point |
| `F32` | - | 32-bit floating point |
| `I64` | - | 64-bit signed integer |
| `I32` | - | 32-bit signed integer |
| `Bool` | - | Boolean |
| `Char` | - | Unicode character |
| `Nat` | `ℕ` | Natural number (arbitrary precision) |
| `Int` | `ℤ` | Integer (arbitrary precision) |
| `I` | - | Alias for Int (common shorthand) |

### Composite Types

```goth
# Function type
I → I                    # Function from Int to Int
I → I → I               # Curried two-argument function

# Tensor type with shape
[n]I                    # Vector of n integers
[m n]F64                # m×n matrix of floats
[3]Bool                 # Vector of 3 booleans

# Tuple type
⟨I, Bool, Char⟩        # Product of three types

# Variant type (sum)
⟨None | Some I⟩        # Option type

# Optional type shorthand
I?                      # Same as ⟨None | Some I⟩
```

---

## Tensor Shapes

Shapes define the dimensions of tensors and are a core part of Goth's type system.

### Shape Syntax

```goth
[]                      # Scalar (rank 0)
[n]                     # Vector of length n
[m n]                   # Matrix m×n
[a b c]                 # 3D tensor

# Concrete shapes
[3]I                    # Exactly 3 integers
[2 3]F64                # 2×3 matrix

# Symbolic shapes (type variables)
[n]I                    # Vector of any length n
[m]I → [n]I → [m n]F64  # Function with shape polymorphism
```

### Shape Variables

Shape variables like `n`, `m` are inferred or declared in type signatures. They allow writing shape-polymorphic functions.

---

## Functions

### Function Declarations

Functions use a distinctive box-drawing syntax:

```goth
╭─ name : TypeSignature
│  ◇effect           # Optional effect annotation
│  where Constraints # Optional constraints
│  ⊢ Precondition    # Optional precondition
│  ⊨ Postcondition   # Optional postcondition
╰─ body
```

ASCII alternative: `/-` for `╭─`

### Examples

```goth
# Simple function: double an integer
╭─ double : I → I
╰─ ₀ × 2

# Function with IO effect
╭─ greet : () → ()
│  ◇io
╰─ print "Hello, World!"

# Identity function
╭─ id : α → α
╰─ ₀

# Map over array (using the map operator)
╭─ squares : [n]I → [n]I
╰─ ₀ ↦ (λ→ ₀ × ₀)
```

### Lambda Expressions

```goth
λ→ body              # Single argument lambda (arg is ₀)
λ→ λ→ body          # Two argument lambda (₁ is first, ₀ is second)
```

### Application

```goth
f x                   # Apply f to x
f x y                 # Apply f to x, then apply result to y (curried)
(f x)                # Parentheses for grouping
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
| `^` | `^` | Power |
| `%` | `%` | Modulo |
| `±` | `+-` | Uncertain value |

### Comparison

| Unicode | ASCII | Operation |
|---------|-------|-----------|
| `=` | `=` | Equality |
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
| `▸` | `\|>_` | Filter | `arr ▸ (λ→ ₀ > 0)` |
| `⤇` | `=>>` | Bind (flatMap) | `arr ⤇ f` |
| `⊗` | `*:` | ZipWith | `a ⊗ b` |
| `⊕` | `+:` | Concat | `a ⊕ b` |
| `∘` | `.:` | Compose | `f ∘ g` |

### Reduction Operators

| Unicode | ASCII | Operation |
|---------|-------|-----------|
| `Σ` | `+/` | Sum |
| `Π` | `*/` | Product |
| `⍀` | `\\/` | Scan (prefix sum) |
| `‖x‖` | `\|\|_x_\|\|` | Norm |

### Unary Operators

| Unicode | ASCII | Operation |
|---------|-------|-----------|
| `-` | `-` | Negation |
| `√` | `sqrt` | Square root |
| `⌊x⌋` | `floor` | Floor |
| `⌈x⌉` | `ceil` | Ceiling |

---

## Effects

Goth tracks computational effects in the type system. Pure functions (no effects) are the default.

### Effect Annotations

| Effect | Symbol | Description |
|--------|--------|-------------|
| Pure | `□` | No effects (default) |
| IO | `◇io` | Input/output |
| Mutation | `◇mut` | Local mutation |
| Random | `◇rand` | Randomness |
| Divergence | `◇div` | May not terminate |
| Exception | `◇exn⟨T⟩` | May throw exception |

### Declaring Effects

```goth
╭─ readNumber : () → I
│  ◇io
╰─ toInt (read_line ⟨⟩)

╭─ pureFunction : I → I
╰─ ₀ × 2                    # No effect annotation = pure
```

---

## Pattern Matching

### Match Expression

```goth
match expr with
  | pattern₁ → body₁
  | pattern₂ → body₂
  | _ → default
```

### Pattern Types

```goth
# Wildcard
_

# Variable binding (adds to environment)
x

# Literal
42
⊤

# Tuple
⟨a, b, c⟩

# Array (exact length)
[x, y, z]

# Array with rest
[head | tail]

# Variant
None
Some x

# Typed pattern
x : I

# Or pattern
pattern₁ | pattern₂

# Guard
pattern if condition
```

### Let Binding

```goth
let pattern ← value in body
```

---

## Primitives Reference

### Sequence Generation

| Name | Alias | Signature | Description |
|------|-------|-----------|-------------|
| `iota` | `ι`, `⍳` | `I → [n]I` | Generate `[0, 1, 2, ..., n-1]` |
| `range` | `…` | `I → I → [m]I` | Generate `[start, ..., end-1]` |

### Array Operations

| Name | Alias | Signature | Description |
|------|-------|-----------|-------------|
| `len` | - | `[n]α → I` | Array length |
| `sum` | - | `[n]I → I` | Sum elements |
| `prod` | - | `[n]I → I` | Product elements |
| `reverse` | `⌽` | `[n]α → [n]α` | Reverse array |
| `concat` | - | `[n]α → [m]α → [n+m]α` | Concatenate |
| `take` | `↑` | `I → [n]α → [m]α` | Take first k elements |
| `drop` | `↓` | `I → [n]α → [m]α` | Drop first k elements |
| `index` | - | `[n]α → I → α` | Get element at index |
| `shape` | `ρ` | `[...]α → [n]I` | Get tensor shape |

### Math Functions

| Name | Signature | Description |
|------|-----------|-------------|
| `abs` | `I → I` | Absolute value |
| `neg` | `I → I` | Negation |
| `sqrt` | `F64 → F64` | Square root |
| `exp` | `F64 → F64` | Exponential |
| `ln` | `F64 → F64` | Natural log |
| `sin`, `cos`, `tan` | `F64 → F64` | Trigonometric |
| `floor`, `ceil`, `round` | `F64 → I` | Rounding |
| `pow` | `I → I → I` | Power |

### Linear Algebra

| Name | Alias | Signature | Description |
|------|-------|-----------|-------------|
| `dot` | `·` | `[n]F64 → [n]F64 → F64` | Dot product |
| `norm` | - | `[n]F64 → F64` | Vector norm |
| `matmul` | - | `[m n]F64 → [n p]F64 → [m p]F64` | Matrix multiply |
| `transpose` | `⍉` | `[m n]α → [n m]α` | Transpose |

### Conversion

| Name | Alias | Signature | Description |
|------|-------|-----------|-------------|
| `toInt` | - | `α → I` | Convert to integer |
| `toFloat` | - | `α → F64` | Convert to float |
| `toBool` | - | `α → Bool` | Convert to boolean |
| `toChar` | - | `I → Char` | Integer to character |
| `toString` | `str` | `α → [n]Char` | Convert to string |
| `strConcat` | `⧺` | `[n]Char → [m]Char → [k]Char` | Concatenate strings |

### IO

| Name | Signature | Description |
|------|-----------|-------------|
| `print` | `α → ()` | Print value |
| `read_line` | `() → [n]Char` | Read line from stdin |

---

## Example Programs

### Hello World

```goth
╭─ main : () → ()
│  ◇io
╰─ print "Hello, World!"
```

### Echo Input

```goth
╭─ main : () → ()
│  ◇io
╰─ print (read_line ⟨⟩)
```

### Sum of Squares

```goth
╭─ main : I → I
╰─ Σ((iota ₀) ↦ (λ→ ₀ × ₀))
```

### Filter Even Numbers

```goth
╭─ main : I → [m]I
╰─ (iota ₀) ▸ (λ→ (₀ % 2) = 0)
```

### Find Syzygies

```goth
# Find pairs (i, n-1-i) for i < n/2
╭─ main : I → ()
│  ◇io
╰─ let count ← ₀ in
   let indices ← iota (count / 2) in
   indices ↦ (λ→ print ⟨₀, count - 1 - ₀⟩);
   ⟨⟩
```

---

## Running Goth Programs

```bash
# Run a file
goth program.goth

# Run with arguments (passed to main)
goth program.goth 10

# Evaluate an expression
goth -e "Σ(iota 10)"

# Parse only (show AST)
goth -a program.goth

# Enable trace output
goth -t program.goth
```
