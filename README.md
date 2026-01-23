# 🦇 𝔤𝔬𝔱𝔥 🖤
## the `goth` language for machine spirits

𝔤𝔬𝔱𝔥 is an LLM-friendly functional programming language with Unicode syntax, dependent types, and tensor operations.

![](./img/header.png)

## Quick Start

```sh
cd crates
cargo build --release
```

### Example Program

```goth
╭─ factorial : I64 → I64
╰─ if ₀ ≤ 1 then 1 else ₀ × factorial (₀ - 1)

╭─ main : () → I64
╰─ factorial 10
```

### Interpreter

The interpreted version of the language is the canonical language.

```sh
# REPL
./target/release/goth

# Run a file
./target/release/goth ../examples/factorial.goth

# Evaluate expression
./target/release/goth -e "Σ [1, 2, 3, 4, 5]"
```

#### Examples

[`add_main.goth`](./examples/add_main.goth)
[`factorial.goth`](./examples/factorial.goth)
[`fibonacci.goth`](./examples/fibonacci.goth)
[`math_comprehensive.goth`](./examples/math_comprehensive.goth)
[`matmul.goth`](./examples/matmul.goth)
[`primes.goth`](./examples/primes.goth)
[`test_iota.goth`](./examples/test_iota.goth)
[`use_math.goth`](./examples/use_math.goth)

### Compiler

The compiler is based on LLVM.  It is still somewhat experimental and may not support all language features.  Feel free to file bug reports.

```sh
# Compile to native executable
./target/release/gothic ../examples/hello_main.goth -o hello
./hello

# Emit LLVM IR
./target/release/gothic program.goth --emit-llvm

# Emit MIR
./target/release/gothic program.goth --emit-mir
```

### Tests

```sh
# Unit tests
cargo test

# Integration tests (interpreter + compiler)
cd .. && bash tests/self_compile_test.sh
```

### Jupyter Kernel

```sh
cd jupyter && ./install.sh
```

## Status

`goth` is an experiment in language design and exotic syntax.  It targets certain desiderata for a language that is human readable but optimized for LLM characteristics.

`goth` was born on 2026-01-16.  The alpha version of the language and interpreter were completed on 2026-01-17.  Type checking was implemented from 2026-01-17 to 2026-01-18.  The compiler was implemented starting on 2026-01-18.  The 1.0 was finalized and released on 2026-01-20.

The language is not particularly efficient to execute, but it is designed to be easy for LLMs to read and write.  In particular, contracts (preconditions and postconditions) are checked at runtime for each evaluation.

### Fully Implemented

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

### In Progress

- Type checker (in place but not hooked up)
- Static type inference
- Type error messages

### Planned

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

### Notes

- All syntax is parsed but not all features are type-checked.
- Contracts are runtime-only (no static proving yet).
- Shape variables are tracked but not unified.
- Effect annotations are parsed but not enforced.
- Refinement types are parsed but predicates not solved.

## Design

* [Reference](./docs/GOTH-LANGUAGE-REFERENCE-v0.1.md)
* [Philosophy](./docs/PHILOSOPHY.md)

Among its unusual features, for `goth` the AST is the primary
representation of the program.  There is a one-to-one mapping
between source code and AST nodes (barring whitespace).  This
is more efficient for the LLM to generate and edit.

## Documentation

- [Language Specification](./LANGUAGE.md) — Full syntax and semantics
- [Philosophy](./docs/PHILOSOPHY.md) — Design rationale

## License

The 𝔤𝔬𝔱𝔥 language name is reserved as a trademark by Sigilante.

The 𝔤𝔬𝔱𝔥 source code is available under the MIT License, © 2026 Sigilante.
