# Changelog

All notable changes to the Goth programming language are documented in this file.

## [0.2.0] - 2026-01-31

Bug fixes, type checker improvements, and quality-of-life cleanup.

### Type Checker
- Cast validation: `as?` (try cast) now returns `Option<T>` instead of bare `T`
- Record update: `Expr::Update` validates field names and types against base record
- LetRec: extracts type annotations from `Pattern::Typed` and `Expr::Annot` instead of using `Type::Hole` for all bindings
- Do blocks: full type threading through Map, Filter, Bind, Op, and Let operations
- Fresh type variables: Forall instantiation uses unique `_t0`, `_t1`... variables instead of `Type::Hole`
- Added `fromChars` type signature to builtins

### Interpreter
- `eval_filter` now propagates predicate errors instead of silently dropping elements
- Added `fromChars` primitive for string round-tripping (`fromChars (chars s) = s`)
- Closure equality now compares captured environment, not just code
- Terminal raw mode: replaced unsafe `static mut` with `Mutex`, added `atexit` handler and `catch_unwind` for panic-safe terminal restoration
- REPL `:quit` now restores terminal state before exiting

### Pretty Printer
- Implemented `OpDecl` pretty printing (was `todo!()` panic)
- Operator precedence now delegates to `BinOp::precedence()` — no more drift between parser and printer

### Documentation
- Added CHANGELOG.md
- LANGUAGE.md: documented 8 missing operators (⍀, ∘, ⊗, ⤇, ◁, ⊕, ▷, ≡)
- LANGUAGE.md: clarified all integer types are i128 at runtime
- Removed empty `examples/string/` and `examples/data-structures/` directories

### Dependencies
- thiserror: 1.0 → 2.0 (all crates)
- Added `license` and `repository` to goth-check Cargo.toml

## [0.1.0] - 2026-01-31

Initial public release.

### Language

- De Bruijn indexed lambda calculus with curried multi-argument functions
- Pattern matching with literal, tuple, array, variant, wildcard, guard, and or-patterns
- Recursive bindings (`let rec`)
- Do-notation for tensor pipelines (map, filter, bind, let, binary ops)
- Tensor (array) types with shape tracking and broadcasting
- Record types with labeled fields and update syntax
- Variant (sum) types with constructor syntax
- Uncertainty propagation: first-class `±` values with automatic error propagation through all math operations
- Cast expressions: `as` (static), `as!` (force), `as?` (try, returns Option)
- Unicode operator surface: `λ`, `→`, `↦`, `▸`, `⤇`, `∘`, `⊕`, `⊗`, `Σ`, `Π`, `⌿`, `⍀`, `▷`, `◁`, `≤`, `≥`, `≠`, `∧`, `∨`, `±`
- ASCII aliases for all Unicode operators
- String literals as character tensors
- Effect annotations (parsed and stored, not yet enforced)
- Type annotations with refinement types (parsed, partially checked)
- Module system with `use` imports and file-based resolution

### Standard Library

- `prelude.goth` — core utilities (identity, const, flip, compose, arithmetic helpers)
- `math.goth` — trigonometric, hyperbolic, logarithmic, and special functions
- `list.goth` — list operations (map, filter, fold, zip, take, drop, reverse, sort, minimum, maximum)
- `option.goth` — generic Option type (some, none, map, flatMap, getOrElse, etc.)
- `result.goth` — generic Result type (ok, err, map, flatMap, recover, etc.)
- `string.goth` — string manipulation (split, join, trim, contains, replace, etc.)
- `tui.goth` — terminal UI helpers (ANSI escapes, cursor control, raw mode)
- `crypto.goth` — pure-Goth SHA-256, SHA-1, MD5, HMAC, and PBKDF2

### Interpreter

- Tree-walking evaluator with global environment and closure capture
- Partial application for all functions
- 127 evaluator tests covering arithmetic, closures, patterns, tensors, uncertainty, I/O
- Built-in primitives: math functions, string operations, file I/O, byte I/O, terminal control
- Checked integer arithmetic (overflow returns error, not panic)
- Checked shift/power operations with bounds validation
- Thread-safe terminal state management with atexit and panic cleanup

### Type Checker

- Bidirectional type checking (inference + checking modes)
- Type unification with occurs check
- Shape unification for tensor dimensions
- 76 type checker tests
- Handles: let, let rec (with annotations), if/else, match, binary/unary ops, tuples, records, arrays, casts, do-blocks, field access, indexing, slicing, annotations
- Fresh type variable generation for polymorphic instantiation
- Type substitution across all type forms including Variant, Exists, App, Effectful, Interval

### Parser

- Full Goth syntax including Unicode and ASCII operator forms
- Proper error reporting on unrecognized tokens (no silent drops)
- Proper error reporting on unrecognized operators (no silent fallback)
- Module parsing with declarations, function definitions, and type definitions
- 111 parser tests

### Tooling

- CLI (`goth`): file execution, expression evaluation (`-e`), REPL, type checking (`-c`)
- REPL with syntax-aware multi-line input, history, and colored output
- AST-first LLM workflow: `--to-json` / `--from-json` for JSON AST round-tripping
- Pretty printer for Goth syntax from AST

### Known Limitations

- Type checker is advisory — programs run regardless of type errors
- Variant constructor validation requires ADT registry (not yet implemented)
- Effect system is parsed but not enforced
- No parametric polymorphism at runtime (evaluator is type-erased)
- Closure equality compares code only, not captured environment
