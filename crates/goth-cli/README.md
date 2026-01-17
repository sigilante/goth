# goth-cli

Command-line interface and REPL for the **Goth** programming language.

## Installation

```bash
cargo install --path .
```

## Usage

```bash
# Start REPL
goth

# Run a file
goth program.goth

# Evaluate expression
goth -e "1 + 2 * 3"

# Parse only (show result without evaluating)
goth -p -e "λ→ ₀ + 1"

# Show AST
goth -a -e "[1,2,3] ↦ λ→ ₀ * 2"

# Enable trace output
goth -t -e "let x = 5 in x * x"
```

## REPL

```
   ╔═══════════════════════════════════════╗
   ║            𝔊𝔬𝔱𝔥  v0.1.0              ║
   ║   Functional • Tensors • Refinements  ║
   ╚═══════════════════════════════════════╝

  Type :help for help, :quit to quit

goth[0]› 1 + 2 * 3
7

goth[1]› let x = 10
let x = 10

goth[2]› x * x
100

goth[3]› [1,2,3,4,5] ↦ λ→ ₀ * 2
[2 4 6 8 10]

goth[4]› Σ [1,2,3,4,5]
15

goth[5]› let double = λ→ ₀ * 2
let double = <λ/1>

goth[6]› double 21
42
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `:help`, `:h`, `:?` | Show help |
| `:quit`, `:q` | Exit REPL |
| `:ast <expr>` | Show AST for expression |
| `:type <expr>` | Show type of expression result |
| `:clear` | Clear environment |
| `:load <file>` | Load definitions from file |

## Syntax Quick Reference

| Syntax | ASCII | Description |
|--------|-------|-------------|
| `λ→ body` | `\-> body` | Lambda |
| `₀ ₁ ₂` | `_0 _1 _2` | De Bruijn indices |
| `⟨x, y⟩` | `(x, y)` | Tuple |
| `Σ xs` | `+/ xs` | Sum |
| `Π xs` | `*/ xs` | Product |
| `xs ↦ f` | `xs -: f` | Map |
| `xs ▸ p` | `xs \|>_ p` | Filter |
| `f ∘ g` | `f .: g` | Compose |
| `⊤ ⊥` | `true false` | Booleans |

## Examples

```goth
# Factorial
let factorial = λ→ match ₀ { 0 → 1; n → n * factorial (n - 1) }
factorial 5  # => 120

# Sum of squares of evens
[1,2,3,4,5,6,7,8,9,10] ▸ λ→ ₀ % 2 = 0 ↦ λ→ ₀ * ₀
# => [4 16 36 64 100]

# Dot product
let dot = λ→ λ→ Σ (₁ ⊗ ₀ ↦ λ→ ₀.0 * ₀.1)
dot [1,2,3] [4,5,6]  # => 32

# Function composition
let double = λ→ ₀ * 2
let inc = λ→ ₀ + 1
let f = double ∘ inc
f 5  # => 12
```

## License

MIT
