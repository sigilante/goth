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
- `₀` (or `_0`) = most recent binding / last parameter
- `₁` (or `_1`) = one binding out / second-to-last parameter
- `₂` (or `_2`) = two bindings out / third-to-last parameter

```goth
# Function args: ₀ = last arg, ₁ = second-to-last arg
╭─ sub : I64 → I64 → I64
╰─ ₁ - ₀              # sub 10 3 = 7  (₁=10, ₀=3)

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
Bitwise:      bitand  bitor  ⊻/bitxor  shl  shr
Map:          arr ↦ λ→ body      (or arr -: \-> body)
Filter:       arr ▸ λ→ body      (or arr |> \-> body)
Fold:         ⌿ (λ→ λ→ ₁+₀) 0 arr  (or fold f acc arr)
Sum:          Σ arr              (or +/ arr)
Product:      Π arr              (or */ arr)
Byte I/O:     ⧏ n "path"  (readBytes)    ⧐ bytes "path"  (writeBytes)
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

## Standard Libraries

### Random Numbers (`stdlib/random.goth`)

Seeded PRNG using xorshift64. All RNG functions return `⟨value, nextSeed⟩` tuples — thread the seed through sequential calls.

```goth
use "stdlib/random.goth"
╭─ main : I64 → F64
╰─ let seed = entropy ⟨⟩
   in let ⟨x, s1⟩ = randFloat seed
   in x
```

Key functions: `entropy` (OS seed), `randFloat` (uniform [0,1)), `randFloatRange` (uniform [lo,hi)), `randInt` (uniform [lo,hi]), `randNormal` (standard normal), `randFloats`/`randInts`/`randNormals` (bulk generation).

### Cryptography (`stdlib/crypto.goth`)

Pure-Goth cryptographic hashes and encoding using bitwise primitives. No FFI.

```goth
use "stdlib/crypto.goth"
╭─ main : () → String
╰─ sha256 "hello"
# → "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
```

Key functions: `sha256` (FIPS 180-4), `md5` (RFC 1321), `blake3` (≤ 64 bytes), `base64EncodeStr`/`base64Decode` (RFC 4648), `hexEncode`.

Implementation notes:
- All 32-bit operations must mask with `bitand X 4294967295` (Goth uses i128)
- SHA-256 = big-endian word packing; MD5, BLAKE3 = little-endian
- Use `fold` over `iota N` for round-based block processing

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

3. **Indexing vs application**: `arr[i]` (no space) is indexing; `f [1,2]` (with space) is function application
   ```goth
   let arr = [10, 20, 30] in arr[1]       # Indexing → 20
   let f = (λ→ Σ ₀) in f [10, 20, 30]    # Application → 60
   ```

4. **Type mismatch I vs I64**: Use `I64` in signatures
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
| `⧺` | `strConcat` |
| `⊕` | `concat` |
| `⍉` | `transpose` |
| `·` | `dot` |
| `±` | `+-` |
| `⌿` | `fold` |
| `⊻` | `bitxor` |
| `⧏` | `readBytes` |
| `⧐` | `writeBytes` |

## Enforcement Notes

- **Contracts** (`⊢` preconditions, `⊨` postconditions): enforced at runtime
- **Effect annotations** (`◇io`, `◇mut`, `◇rand`): parsed but **not enforced** — `print` works without `◇io`. Use effect annotations as documentation only; do not rely on them for correctness.
- **Type annotations**: parsed but type checking is partial
- **Refinement types** (`{x : F64 | x > 0}`): parsed but predicates not solved

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
