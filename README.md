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
./target/release/goth ../examples/recursion/factorial.goth

# Evaluate expression
./target/release/goth -e "Σ [1, 2, 3, 4, 5]"
```

#### Examples

See [`examples/README.md`](./examples/README.md) for a full listing organized by category.

| Category | Examples |
|----------|---------|
| [`basic/`](./examples/basic/) | identity, double, square, abs, sign, is_even, max, min |
| [`recursion/`](./examples/recursion/) | factorial, fibonacci, gcd, ackermann, collatz, hyperop |
| [`higher-order/`](./examples/higher-order/) | map, filter, fold, compose, pipeline, zip |
| [`numeric/`](./examples/numeric/) | harmonic, pi, exp, sqrt, gamma |
| [`algorithms/`](./examples/algorithms/) | isPrime, count_primes, nth_prime, binary_search, modpow |
| [`contracts/`](./examples/contracts/) | abs_post, div_safe, factorial_contract, sqrt_safe |
| [`tco/`](./examples/tco/) | naive vs tail-call-optimized pairs |
| [`io/`](./examples/io/) | stdout, stderr, file write |
| [`uncertainty/`](./examples/uncertainty/) | measurement, error propagation through arithmetic |
| [`simulation/`](./examples/simulation/) | heat diffusion, wave equation, Laplace, power iteration |
| [`random/`](./examples/random/) | PRNG, Buffon's needle, Monte Carlo integration |
| [`crypto/`](./examples/crypto/) | SHA-256, MD5, BLAKE3, Base64, file hashing |

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
- Effect type enforcement (currently parsed but not enforced)
- Dependent shape inference
- Polymorphism (let-generalization)
- Bigint backing for ℕ/ℤ types
- Package manager
- Language server protocol (LSP)
- Debugger
- Profiler
- Optimizations

### Notes

- All syntax is parsed but not all features are type-checked.
- Contracts are runtime-only (no static proving yet).
- Effect annotations are parsed but not enforced.
- Refinement types are parsed but predicates not solved.
- Comment syntax (`#`), module system (`use`), and standard library (11 modules) are implemented.

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
