# Goth Quick Reference for LLMs

Goth is a functional language with tensor operations and de Bruijn indices. This is a compact reference for code generation.

## Syntax Quick Reference

```
Comments:     # comment
Functions:    ╭─ name : Type → ReturnType
              ╰─ body
Lambda:       λ→ body          (or \-> body)
Let:          let x ← expr in body
If:           if cond then a else b
```

## De Bruijn Indices (Critical)

Variables are numbered by binding depth, not named:
- `₀` (or `_0`) = first parameter / most recent binding
- `₁` (or `_1`) = second parameter / one binding out
- `₂` (or `_2`) = third parameter / two bindings out

```goth
# Function args: ₀ = first arg, ₁ = second arg
╭─ sub : I64 → I64 → I64
╰─ ₀ - ₁              # sub 10 3 = 7

# Let shifts indices
╭─ example : I64 → I64
╰─ let x ← ₀ × 2 in   # ₀ = argument
   let y ← ₀ + 1 in   # ₀ = x, ₁ = argument
   ₀ + ₁              # ₀ = y, ₁ = x, ₂ = argument

# Lambdas also shift
[1, 2, 3] ↦ λ→ ₀ × 2  # Inside lambda: ₀ = array element
```

## Types

```
I64, I32            Integer
F64, F32            Float
Bool                Boolean (literals: true/false or ⊤/⊥)
[n]I64              Vector of n integers
[m n]F64            m×n matrix of floats
I64 → I64           Function
⟨I64, Bool⟩         Tuple
```

## Common Operators

```
Arithmetic:   +  -  ×  /  %  ^  ±  (% also written as mod, ± also +-)
Comparison:   =  ≠  <  >  ≤  ≥
Equality:     =  (value)  ≡/==  (structural)  ≣/===  (referential, reserved)
Logical:      ∧  ∨  ¬            (or &&  ||  !)
Map:          arr ↦ λ→ body      (or arr -: \-> body)
Filter:       arr ▸ λ→ body      (or arr |> \-> body)
Sum:          Σ arr              (or +/ arr)
Product:      Π arr              (or */ arr)
```

## Common Patterns

### Map over array
```goth
# Double each element
[1, 2, 3] ↦ λ→ ₀ × 2    # Result: [2, 4, 6]

# Square each element
arr ↦ λ→ ₀ × ₀
```

### Filter array
```goth
# Keep elements > 5
[1, 8, 3, 9, 2] ▸ λ→ ₀ > 5    # Result: [8, 9]

# Keep even numbers
arr ▸ λ→ (₀ % 2) = 0
```

### Reduce to single value
```goth
Σ [1, 2, 3, 4, 5]    # Sum = 15
Π [1, 2, 3, 4]       # Product = 24
```

### Conditional
```goth
if ₀ > 0 then ₀ else -₀     # Absolute value
if (₀ % 2) = 0 then ₀ / 2 else ₀ × 3 + 1
```

### Nested let
```goth
let a ← ₀ × 2 in
let b ← ₀ + 1 in    # ₀ = a (not ₁!)
₀ + ₁              # ₀ = b, ₁ = a
```

### Typed let (for shape checking)
```goth
let v : [3]F64 ← [1.0, 2.0, 3.0] in v    # OK - shapes match
let v : [5]F64 ← [1.0, 2.0, 3.0] in v    # Error: shape mismatch
```

## Complete Program Template

```goth
# Description comment

╭─ main : () → I64
╰─ expression

# Or with argument:
╭─ main : I64 → I64
╰─ let x ← ₀ in    # ₀ is the CLI argument
   x × x
```

## Working Examples

### Sum of squares
```goth
╭─ main : () → I64
╰─ Σ ([1, 2, 3, 4, 5] ↦ λ→ ₀ × ₀)
```

### Newton's sqrt
```goth
╭─ main : I64 → I64
╰─ let n ← ₀ in
   let x1 ← (n + n / n) / 2 in
   let x2 ← (₀ + ₂ / ₀) / 2 in    # ₀=x1, ₂=n
   let x3 ← (₀ + ₃ / ₀) / 2 in    # ₀=x2, ₃=n
   ₀
```

### Primality test
```goth
╭─ main : I64 → I64
╰─ if ₀ < 2 then 0
   else if ₀ = 2 then 1
   else if (₀ % 2) = 0 then 0
   else 1
```

### Character conversion
```goth
# Uppercase: a-z (97-122) → A-Z (65-90)
╭─ main : I64 → I64
╰─ if (₀ ≥ 97) ∧ (₀ ≤ 122) then ₀ - 32 else ₀
```

### Polynomial 1 + 2x + 3x²
```goth
╭─ main : I64 → I64
╰─ let x ← ₀ in
   1 + 2 × x + 3 × x × x
```

## Uncertainty Propagation

Goth has first-class uncertain values. Create with `±`, and uncertainty propagates automatically through arithmetic and math functions.

```goth
# Create uncertain value
╭─ main : F64 → F64 → (F64 ± F64)
╰─ let a ← ₁ ± ₀ in        # a = value ± uncertainty
   √₀ + (2.0 ± 0.1)        # propagates through √ and +
```

Propagation rules: additive (√(δa²+δb²)) for `+`/`-`, relative error for `×`/`/`, derivative-based for math functions (`√`, `sin`, `cos`, `exp`, `ln`, etc.).

## Common Mistakes

1. **Wrong index after let**: Each `let` shifts indices by 1
   ```goth
   # WRONG: trying to use argument after let
   let x ← 5 in ₀ + ₀     # ₀ is x, not argument!

   # RIGHT: argument shifted to ₁
   let x ← 5 in ₀ + ₁     # ₀=x, ₁=argument
   ```

2. **Lambda index scope**: Inside `λ→`, ₀ is the lambda parameter
   ```goth
   # In this example:
   let x ← 5 in [1,2,3] ↦ λ→ ₀ + ₁    # ₀=elem, ₁=x (not argument!)
   # Argument would be ₂ inside the lambda
   ```

3. **Type mismatch I vs I64**: Use `I64` in signatures
   ```goth
   # WRONG
   ╭─ main : () → I

   # RIGHT
   ╭─ main : () → I64
   ```

4. **Comments**: Use `#` for comments
   ```goth
   # Line comment
   x + y  # Inline comment
   ```

## ASCII Alternatives

| Unicode | ASCII |
|---------|-------|
| `λ→` | `\->` |
| `╭─` | `/-` |
| `╰─` | `\-` |
| `←` | `<-` |
| `↦` | `-:` |
| `▸` | `\|>` |
| `Σ` | `+/` |
| `Π` | `*/` |
| `×` | `*` |
| `≤` | `<=` |
| `≥` | `>=` |
| `≠` | `/=` |
| `≡` | `==` |
| `≣` | `===` |
| `%` | `mod` |
| `∧` | `&&` |
| `∨` | `\|\|` |
| `¬` | `!` |
| `₀` | `_0` |
| `⊤` | `true` |
| `⊥` | `false` |
| `ι` | `iota` |
| `⧺` | `++` |
| `⍉` | `transpose` |
| `·` | `dot` |

## Running Code

```sh
# Interpreter
goth file.goth
goth file.goth 42         # Pass argument to main

# REPL
goth
> Σ [1, 2, 3]
6

# Compiler
gothic file.goth -o program
./program
```
