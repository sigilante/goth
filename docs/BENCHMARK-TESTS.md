# Goth Benchmark Test Battery

This document defines the test battery for validating Goth's thesis: that LLMs can generate correct, runnable code more reliably via AST-first generation than traditional text-based approaches.

## Current Test Coverage

### 1. Basic Operations (10 tests) - `examples/basic/`

Simple single-expression functions testing basic arithmetic and logic.

| Test | Description | Input → Output | Contract |
|------|-------------|----------------|----------|
| `identity` | Identity function | `5 → 5` | output = input |
| `add_one` | Increment by one | `5 → 6` | output = input + 1 |
| `double` | Multiply by two | `5 → 10` | output = 2 × input |
| `square` | Square a number | `5 → 25` | output = input² |
| `max_two` | Maximum of two | `3 7 → 7` | output ≥ both inputs |
| `min_two` | Minimum of two | `3 7 → 3` | output ≤ both inputs |
| `abs` | Absolute value | `-5 → 5` | output ≥ 0 |
| `sign` | Sign function | `-5 → -1` | returns -1, 0, or 1 |
| `is_even` | Even check | `4 → ⊤` | boolean |
| `is_positive` | Positive check | `-3 → ⊥` | boolean |

### 2. Recursion (14 tests) - `examples/recursion/`

Classic recursive algorithms with mathematical properties.

| Test | Description | Input → Output | Contract |
|------|-------------|----------------|----------|
| `factorial` | n! | `5 → 120` | n! = n × (n-1)! |
| `fibonacci` | Fib(n) | `10 → 55` | F(n) = F(n-1) + F(n-2) |
| `sum_to_n` | Σ(1..n) | `10 → 55` | = n(n+1)/2 |
| `power` | b^e | `2 10 → 1024` | integer exponentiation |
| `gcd` | GCD(a,b) | `48 18 → 6` | Euclidean algorithm |
| `lcm` | LCM(a,b) | `4 6 → 12` | = a×b/gcd(a,b) |
| `ackermann` | A(m,n) | `3 4 → 125` | non-primitive recursive |
| `sudan` | F_n(x,y) | `0 3 4 → 7` | non-primitive recursive |
| `collatz_len` | Steps to 1 | `7 → 16` | Collatz conjecture |
| `digit_sum` | Σ digits | `12345 → 15` | digit sum |
| `reverse_num` | Reverse | `1234 → 4321` | reverse digits |
| `hyperop` | H_n(a,b) | `2 3 4 → 12` | generalized operations |
| `tak` | Takeuchi | `18 12 6 → 7` | benchmark function |
| `mccarthy91` | M(n) | `99 → 91` | M(n) = 91 for n ≤ 100 |

### 3. Higher-Order Functions (10 tests) - `examples/higher-order/`

Functional programming patterns.

| Test | Description | Input → Output | Contract |
|------|-------------|----------------|----------|
| `map_double` | Double list | `5 → [2,4,6,8,10]` | length preserved |
| `filter_positive` | Keep positive | `5 → [1,2,3,4,5]` | all > 0 |
| `fold_sum` | Sum via fold | `10 → 55` | recursive sum |
| `fold_product` | Product via fold | `5 → 120` | = n! |
| `compose` | f(g(x)) | `3 → 36` | square(double(3)) |
| `apply_twice` | f(f(x)) | `3 → 12` | double(double(3)) |
| `all_positive` | ∀ > 0 | `5 → ⊤` | boolean |
| `any_negative` | ∃ < 0 | `5 → ⊤` | boolean |
| `count_if` | Count evens | `10 → 5` | count satisfying |
| `pipeline` | Composed ops | `6 → 56` | sum(squares(evens)) |

### 4. Numeric Algorithms (8 tests) - `examples/numeric/`

Mathematical functions and special values.

| Test | Description | Input → Output | Contract |
|------|-------------|----------------|----------|
| `gamma_fact` | Γ(n+1) = n! | `5.0 → 120.0` | Gamma function |
| `gamma_half` | Γ(x) | `0.5 → 1.77...` | Γ(1/2) = √π |
| `sum_squares` | Σ k² | `5 → 55` | = n(n+1)(2n+1)/6 |
| `product_range` | Π(1..n) | `5 → 120` | uses Π operator |
| `harmonic` | H(n) | `10 → 2.93...` | ≈ ln(n) + γ |
| `exp_taylor` | e^x | `1.0 → 2.718...` | Taylor series |
| `pi_leibniz` | π approx | `20 → 3.19...` | Leibniz formula |
| `sqrt_newton` | √x | `2.0 → 1.414...` | Newton-Raphson |

### 5. Algorithms (6 tests) - `examples/algorithms/`

Classic computer science algorithms.

| Test | Description | Input → Output | Contract |
|------|-------------|----------------|----------|
| `binary_search` | Find in range | `10 7 → 7` | O(log n) |
| `isPrime` | Primality | `17 → ⊤` | trial division |
| `count_primes` | π(n) | `20 → 8` | prime counting |
| `nth_prime` | p_n | `10 → 29` | nth prime |
| `isqrt` | ⌊√n⌋ | `50 → 7` | integer sqrt |
| `modpow` | b^e mod m | `2 10 1000 → 24` | fast exponentiation |

### 6. Uncertainty (6 tests) - `examples/uncertainty/`

Uncertain value creation and error propagation through operations.

| Test | Description | Input → Output | Contract |
|------|-------------|----------------|----------|
| `measure` | Create uncertain value | `10.5 0.3 → 10.5±0.3` | value ± uncertainty |
| `add_uncertain` | Additive propagation | `10 0.3 20 0.4 → 30±0.5` | δ = √(δa² + δb²) |
| `mul_uncertain` | Multiplicative propagation | `5 0.1 3 0.2 → 15±1.04` | relative error quadrature |
| `sqrt_uncertain` | Sqrt propagation | `9 0.3 → 3±0.05` | δ = δx / (2√x) |
| `sin_uncertain` | Trig propagation | `1.0 0.1 → 0.841±0.054` | δ = \|cos x\| × δx |
| `chained_uncertain` | Multi-step propagation | `4 0.2 1 0.1 → 0.141±0.111` | sin(√(a±δa)+(b±δb)) |

## Running Tests

```bash
# From the crates directory, use absolute paths:
cargo run --quiet --package goth-cli -- "$(pwd)/../examples/category/test.goth" args

# Examples:
cargo run --quiet --package goth-cli -- /path/to/examples/recursion/factorial.goth 5
# Output: 120

cargo run --quiet --package goth-cli -- /path/to/examples/algorithms/isPrime.goth 17
# Output: ⊤

cargo run --quiet --package goth-cli -- /path/to/examples/numeric/gamma_fact.goth 5.0
# Output: 120.00000000000021
```

Note: Use absolute paths due to a CLI path resolution issue with relative paths.

## Contract Documentation

Each `.goth` file includes contracts as comments:

```goth
# Factorial: n!
# Precondition: n >= 0
# Postcondition: result >= 1
# Property: factorial(n) = n * factorial(n-1) for n > 0
# Property: factorial(0) = 1
╭─ main : I64 → I64
╰─ if ₀ < 1 then 1
   else ₀ × main (₀ - 1)
```

## De Bruijn Index Convention

For multi-argument functions, indices count from most-recent binding:
- `f : A → B → C` - ₁ = first arg (A), ₀ = second arg (B)
- `f : A → B → C → D` - ₂ = first, ₁ = second, ₀ = third

## Known Limitations

1. **Stack depth**: Deep recursion may overflow (e.g., collatz(27) needs 111 steps)
2. **Type checker gaps**: Some primitives (fold, map operator) not fully type-checked
3. **Relative paths**: CLI doubles relative paths; use absolute paths

## Test Statistics

| Category | Count | Verified |
|----------|-------|----------|
| Basic | 10 | ✓ |
| Recursion | 14 | ✓ |
| Higher-Order | 10 | ✓ |
| Numeric | 8 | ✓ |
| Algorithms | 6 | ✓ |
| Uncertainty | 6 | ✓ |
| **Total** | **54** | |

## Future Additions

- String manipulation tests
- Data structure tests (lists, trees)
- I/O tests
- Multi-argument edge cases
- JSON AST generation tests
