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
- `₀` (or `_0`) = innermost/most recent binding
- `₁` (or `_1`) = one level out
- `₂` (or `_2`) = two levels out

```goth
# Function args: first arg binds to higher index
╭─ add : I64 → I64 → I64
╰─ ₁ + ₀              # ₁ = first arg, ₀ = second arg

# Let shifts indices
╭─ example : I64 → I64
╰─ let x ← ₀ × 2 in   # ₀ = argument
   let y ← ₁ + 1 in   # ₀ = x, ₁ = argument
   ₀ + ₁              # ₀ = y, ₁ = x

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
Arithmetic:   +  -  ×  /  %  ^
Comparison:   =  ≠  <  >  ≤  ≥
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
   # WRONG: ₀ outside lambda is not ₀ inside
   let x ← 5 in [1,2,3] ↦ λ→ ₀ + ₁    # ₀=elem, ₁=x (not argument)
   ```

3. **Type mismatch I vs I64**: Use `I64` in signatures
   ```goth
   # WRONG
   ╭─ main : () → I

   # RIGHT
   ╭─ main : () → I64
   ```

4. **Comments**: Use `#` not `--`
   ```goth
   # RIGHT: hash comment
   -- WRONG: double dash doesn't work
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
| `∧` | `&&` |
| `∨` | `\|\|` |
| `¬` | `!` |
| `₀` | `_0` |
| `⊤` | `true` |
| `⊥` | `false` |

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
