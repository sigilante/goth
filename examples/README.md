# Goth Examples

A collection of examples demonstrating the Goth programming language, organized by concept. All examples run under the eval interpreter.

## Running Examples

From the `crates/` directory:

```sh
# Run an example with an integer argument
cargo run --quiet --package goth-cli -- ../examples/basic/square.goth 7
# Output: 49

# Run with no argument (for () → T signatures)
cargo run --quiet --package goth-cli -- ../examples/numeric/product_range.goth

# Evaluate an inline expression
cargo run --quiet --package goth-cli -- -e 'Σ [1, 2, 3, 4, 5]'
# Output: 15
```

Most examples define `main : I64 → ...` and expect a single integer argument. Multi-argument functions return a partial application when given fewer arguments than needed — pass additional arguments as space-separated values.

```sh
# Two-argument function
cargo run --quiet --package goth-cli -- ../examples/recursion/gcd.goth 12 8
# Output: 4
```

### Running All Examples

```sh
cd crates
for f in ../examples/basic/*.goth; do
  echo "$(basename $f): $(cargo run --quiet --package goth-cli -- "$f" 5 2>&1)"
done
```

---

## basic/

Elementary single-function programs. Each takes one integer and returns a result.

| File | Description | Example |
|------|-------------|---------|
| `identity.goth` | Returns its argument unchanged | `5 → 5` |
| `add_one.goth` | Adds one | `5 → 6` |
| `double.goth` | Doubles the input | `5 → 10` |
| `square.goth` | Squares the input | `5 → 25` |
| `abs.goth` | Absolute value | `-3 → 3` |
| `sign.goth` | Sign function (−1, 0, or 1) | `5 → 1` |
| `is_even.goth` | Evenness check | `4 → ⊤` |
| `is_positive.goth` | Positivity check | `5 → ⊤` |
| `max_two.goth` | Maximum of two integers (curried) | `3 7 → 7` |
| `min_two.goth` | Minimum of two integers (curried) | `3 7 → 3` |

**Demonstrates:** Function declarations, De Bruijn indices, conditionals, boolean returns.

---

## recursion/

Classic recursive algorithms. Demonstrates self-referencing function calls, base cases, and termination.

| File | Description | Example |
|------|-------------|---------|
| `factorial.goth` | n! | `5 → 120` |
| `fibonacci.goth` | Fibonacci numbers | `10 → 55` |
| `sum_to_n.goth` | Triangular numbers: 1 + 2 + ... + n | `5 → 15` |
| `power.goth` | base^exp (two args) | `2 10 → 1024` |
| `gcd.goth` | Euclidean GCD (two args) | `12 8 → 4` |
| `lcm.goth` | Least common multiple (two args) | `4 6 → 12` |
| `ackermann.goth` | Ackermann function (two args) — non-primitive recursive | `2 3 → 9` |
| `sudan.goth` | Sudan function (three args) — non-primitive recursive | `1 1 1 → 4` |
| `collatz_len.goth` | Length of Collatz (3n+1) sequence | `7 → 16` |
| `digit_sum.goth` | Sum of decimal digits | `1234 → 10` |
| `reverse_num.goth` | Reverse digits of a number | `1234 → 4321` |
| `hyperop.goth` | Hyperoperation / Knuth's up-arrow (three args) | `2 3 4 → 65536` |
| `tak.goth` | Takeuchi function (three args) — classic call-overhead benchmark | `7 4 2 → 4` |
| `mccarthy91.goth` | McCarthy 91 function — nested recursion | `85 → 91` |

**Demonstrates:** Recursive self-calls, multi-argument functions, De Bruijn index ordering in curried functions.

---

## higher-order/

Functions that take or return other functions. Demonstrates map, filter, fold, and function composition.

| File | Description | Example |
|------|-------------|---------|
| `map_double.goth` | Map: double each element of a range | `5 → [2 4 6 8 10]` |
| `filter_positive.goth` | Filter: keep only positive numbers | `5 → [1 2 3 4 5]` |
| `compose.goth` | Function composition: square(double(x)) | `5 → 100` |
| `apply_twice.goth` | Apply a function twice: f(f(x)) | `5 → 20` |
| `all_positive.goth` | Check if all elements satisfy a predicate | `5 → ⊤` |
| `any_negative.goth` | Check if any element satisfies a predicate | `5 → ⊤` |
| `count_if.goth` | Count elements matching a predicate | `5 → 2` |
| `fold_sum.goth` | Fold/reduce to compute sum | `5 → 15` |
| `fold_product.goth` | Fold/reduce to compute product (factorial) | `5 → 120` |
| `pipeline.goth` | Chained operations: range → filter → map → sum | `6 → 56` |
| `concat.goth` | Array concatenation with `⊕` | `5 → [1 2 3 4 5 1 2 3 4 5]` |
| `zip_sum.goth` | Zip two arrays into pairs | `5 → [⟨1,0⟩ ⟨2,1⟩ ...]` |

**Demonstrates:** `↦` (map), `▸` (filter), `Σ`/`Π` (reduce), lambda expressions, closures, `⊕` (concat).

---

## numeric/

Numerical computing: series, special functions, and iterative methods.

| File | Description | Example |
|------|-------------|---------|
| `sum_squares.goth` | Sum of squares: 1² + 2² + ... + n² | `5 → 55` |
| `product_range.goth` | Product of 1..n via `Π` (factorial) | `5 → 120` |
| `harmonic.goth` | Harmonic number H(n) = 1 + 1/2 + ... + 1/n | `5 → 2.28...` |
| `pi_leibniz.goth` | π via Leibniz series (slow convergence) | `20 → 2.976...` |
| `exp_taylor.goth` | e^x via Taylor series | `5.0 → 148.41...` |
| `sqrt_newton.goth` | √x via Newton-Raphson iteration | `5.0 → 2.236...` |
| `gamma_fact.goth` | Factorial via the Gamma function: Γ(n+1) = n! | `5.0 → 120.0` |
| `gamma_half.goth` | Gamma at half-integers: Γ(n+½) | `5.0 → 24.0` |

**Demonstrates:** `Σ`, `Π`, `ι` (iota), `↦` (map), floating-point arithmetic, built-in math functions (`Γ`, `√`, `exp`, `ln`).

---

## algorithms/

Classical algorithms implemented in a functional style.

| File | Description | Example |
|------|-------------|---------|
| `isPrime.goth` | Primality test by trial division | `7 → ⊤` |
| `count_primes.goth` | Count primes ≤ n (prime counting function π(n)) | `10 → 4` |
| `nth_prime.goth` | Find the nth prime number | `5 → 11` |
| `isqrt.goth` | Integer square root ⌊√n⌋ | `10 → 3` |
| `binary_search.goth` | Binary search in a range (three args) | `7 0 10 → 7` |
| `modpow.goth` | Modular exponentiation base^exp mod m (three args) | `2 10 1000 → 24` |

**Demonstrates:** Multi-argument curried functions, conditional recursion, helper functions.

---

## contracts/

Functions with runtime-checked preconditions (`⊢`) and postconditions (`⊨`).

| File | Description | Contract |
|------|-------------|----------|
| `abs_post.goth` | Absolute value | Postcondition: result ≥ 0 |
| `div_safe.goth` | Safe division (two args) | Precondition: divisor ≠ 0 |
| `factorial_contract.goth` | Factorial | Pre: n ≥ 0, Post: result ≥ 1 |
| `log_safe.goth` | Natural logarithm | Precondition: x > 0 |
| `sqrt_safe.goth` | Square root | Precondition: x ≥ 0 |

**Demonstrates:** `⊢` (preconditions), `⊨` (postconditions), `│` (contract lines in function boxes).

---

## tco/

Paired naive vs. tail-call-optimized versions of the same algorithm. Each pair shows the transformation from non-tail-recursive to accumulator-based tail recursion.

| Pair | Naive | TCO |
|------|-------|-----|
| Factorial | `factorial_naive.goth` | `factorial_tco.goth` |
| Fibonacci | `fibonacci_naive.goth` | `fibonacci_tco.goth` |
| Sum 1..n | `sum_naive.goth` | `sum_tco.goth` |
| Collatz length | `collatz_naive.goth` | `collatz_tco.goth` |
| List length | `length_naive.goth` | `length_tco.goth` |

Both versions produce identical results. The TCO versions use accumulator parameters and place the recursive call in tail position.

**Demonstrates:** Tail recursion, accumulator pattern, helper functions with extra parameters.

---

## io/

File and stream I/O using the `▷` (write) operator.

| File | Description |
|------|-------------|
| `write_stdout.goth` | Write `"hello"` to stdout (no newline) |
| `write_stderr.goth` | Write `"error"` to stderr |
| `write_file.goth` | Write `"hello world"` to `/tmp/goth_write_test.txt` |

**Demonstrates:** `▷ stdout`, `▷ stderr`, `▷ "filepath"` — the write operator dispatches on the right-hand side.

---

## uncertainty/

First-class uncertain values with automatic error propagation.

| File | Description | Example |
|------|-------------|---------|
| `measure.goth` | Create an uncertain value | `10.0 0.5 → 10 ± 0.5` |
| `add_uncertain.goth` | Addition with error propagation | `10 0.3 20 0.4 → 30 ± 0.5` |
| `mul_uncertain.goth` | Multiplication with relative error | `5 0.1 3 0.2 → 15 ± 1.04...` |
| `sqrt_uncertain.goth` | Square root with derivative propagation | `9 0.3 → 3 ± 0.05` |
| `sin_uncertain.goth` | Sine with derivative propagation | `1.0 0.1 → 0.841... ± 0.054...` |
| `chained_uncertain.goth` | Multi-step propagation: sin(√a + b) | `4 0.1 1 0.05 → ...` |

**Demonstrates:** The `±` operator, automatic propagation through `+`, `-`, `×`, `/`, `√`, `sin`, `cos`, `exp`, `ln`, etc.

---

## simulation/

Numerical simulations that produce SVG visualizations and CSV data files. These examples write output files to the working directory (typically `crates/`).

| File | Description | Output Files |
|------|-------------|-------------|
| `heat1d.goth` | 1D heat diffusion (20 pts, 50 steps, r=0.4) | `heat1d.svg` |
| `heat2d.goth` | 2D heat diffusion (15×15 grid, 30 steps, r=0.2) | `heat2d.svg` |
| `heat1d_tunable.goth` | 1D heat with imported config | `heat1d.svg` |
| `heat2d_tunable.goth` | 2D heat with imported config | `heat2d.svg` |
| `newton_raphson.goth` | Newton-Raphson cube root finder | `newton.svg`, `newton.csv` |
| `wave1d.goth` | 1D wave equation (leapfrog scheme) | `wave1d.svg`, `wave1d.csv` |
| `laplace2d.goth` | 2D Laplace equation (Jacobi iteration) | `laplace2d.svg`, `laplace2d.csv` |
| `power_iter.goth` | Eigenvalue power iteration | `power_iter.svg`, `power_iter.csv` |

Config files (`heat1d_config.goth`, `heat2d_config.goth`) are imported via `use "file.goth"` and contain bare `let` bindings for tunable parameters.

```sh
# Run a simulation (argument is unused, pass 0)
cargo run --quiet --package goth-cli -- ../examples/simulation/heat2d.goth 0

# Open the SVG output
open heat2d.svg
```

**Demonstrates:** Recursive time-stepping, flat 2D array indexing, string building for SVG/CSV, `writeFile`/`▷` for file output, `use` imports for configuration, `toString`, `strConcat`/`⧺`.

---

## Language Features Covered

| Feature | Examples |
|---------|---------|
| De Bruijn indices (`₀`, `₁`, `₂`) | All |
| Function declarations (`╭─`/`╰─`) | All |
| Lambda expressions (`λ→`) | higher-order, numeric, simulation |
| Let bindings | numeric, simulation, algorithms |
| Recursion | recursion, tco, algorithms, simulation |
| Map (`↦`) | higher-order, numeric, simulation |
| Filter (`▸`) | higher-order |
| Sum/Product (`Σ`/`Π`) | numeric, higher-order, simulation |
| Iota (`ι`) | numeric, simulation |
| Array concat (`⊕`) | higher-order |
| String concat (`⧺`) | simulation |
| Conditionals (`if/then/else`) | basic, recursion, algorithms |
| Contracts (`⊢`/`⊨`) | contracts |
| Uncertainty (`±`) | uncertainty |
| File I/O (`▷`, `writeFile`) | io, simulation |
| Module imports (`use`) | simulation (tunable variants) |
| Tail-call optimization | tco |
