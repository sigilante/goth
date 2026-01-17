# goth-parse

Parser for the **Goth** programming language.

## Features

- Logos-based lexer with Unicode and ASCII fallback support
- Pratt parser for expressions with correct precedence/associativity
- Recursive descent for types, patterns, and declarations
- De Bruijn index parsing (₀, ₁, ... or _0, _1, ...)
- Full support for Goth syntax including:
  - Lambda expressions (λ→ or \->)
  - Functional operators (↦ ▸ ⤇ ∘ or ASCII equivalents)
  - Tensor types ([n m]F64)
  - Pattern matching with guards
  - Function box declarations (╭─ ... ╰─)

## Quick Example

```rust
use goth_parse::prelude::*;

// Parse expression
let expr = parse_expr("λ→ ₀ + 1").unwrap();

// Parse type
let ty = parse_type("[n]F64 → F64").unwrap();

// Parse pattern
let pat = parse_pattern("Some x").unwrap();

// Parse module
let module = parse_module("let x = 42", "main").unwrap();
```

## Syntax Overview

### Expressions

| Syntax | Description |
|--------|-------------|
| `42`, `3.14` | Numeric literals |
| `"hello"`, `'c'` | String/char literals |
| `⊤`, `⊥` | Boolean literals (or `true`, `false`) |
| `₀`, `₁`, ... | De Bruijn indices (or `_0`, `_1`, ...) |
| `foo` | Identifier |
| `λ→ body` | Lambda (or `\-> body`) |
| `f x` | Application |
| `x + y` | Binary operation |
| `[1, 2, 3]` | Array |
| `(x, y)` | Tuple |
| `⟨x, y⟩` | Angle tuple (or `<| x, y |>`) |
| `let x = e in body` | Let binding |
| `if c then t else e` | Conditional |
| `match e { p → b }` | Pattern matching |

### Operators (by precedence, low to high)

| Prec | Operators |
|------|-----------|
| 2 | `↦` `-:` (map), `▸` (filter), `⤇` `=>>` (bind) |
| 3 | `∨` `\|\|` (or) |
| 4 | `∧` `&&` (and) |
| 5 | `=` `≠` `<` `>` `≤` `≥` |
| 6 | `⊕` `+:` (concat) |
| 7 | `+` `-` |
| 8 | `×` `*` `/` `%` `⊗` `*:` |
| 9 | `^` (power) |
| 10 | `∘` `.:` (compose) |

### Unary Operators

| Syntax | Description |
|--------|-------------|
| `Σ` or `+/` | Sum |
| `Π` or `*/` | Product |
| `⍀` or `\\/` | Scan |
| `-` | Negate |
| `¬` or `!` | Not |

### Types

| Syntax | Description |
|--------|-------------|
| `F64`, `I32`, etc. | Primitive types |
| `[n]T` | Vector of T with length n |
| `[m n]T` | Matrix of T |
| `T → U` | Function type |
| `(T, U)` | Tuple type |
| `∀ α. T` | Universal type |

### Patterns

| Syntax | Description |
|--------|-------------|
| `_` | Wildcard |
| `x` | Variable binding |
| `42` | Literal |
| `(p, q)` | Tuple |
| `[p, q, r]` | Array |
| `[h \| t]` | Array split |
| `Some x` | Variant |
| `p \| q` | Or pattern |

## Unicode ↔ ASCII

The parser accepts both Unicode glyphs and ASCII fallbacks:

| Unicode | ASCII |
|---------|-------|
| `λ` | `\` |
| `→` | `->` |
| `↦` | `-:` |
| `⤇` | `=>>` |
| `∘` | `.:` |
| `⊗` | `*:` |
| `⊕` | `+:` |
| `Σ` | `+/` |
| `Π` | `*/` |
| `∀` | `forall` |
| `⊤` | `true` |
| `⊥` | `false` |
| `₀₁₂...` | `_0`, `_1`, `_2`, ... |
| `╭─` | `/-` |
| `╰─` | `\-` |
| `⟨` | `<\|` |
| `⟩` | `\|>` |

## License

MIT
