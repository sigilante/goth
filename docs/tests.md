# Goth Quick Reference - Implemented Features

## Test Script

Copy-paste this into your Goth REPL to test all features:

```goth
# =============================================================================
# PRIORITY 1: Multi-line Input
# =============================================================================

# Simple multi-line let
let x =
  5
in x * x

# Multi-line function
╭─ square : F64 → F64
╰─ ₀ × ₀

# Multi-line if-then-else
if ⊤
then 42
else 0

# =============================================================================
# PRIORITY 2: Postfix Reduction Operators
# =============================================================================

# Sum
[1, 2, 3, 4, 5] Σ        # → 15

# Product
[1, 2, 3, 4] Π           # → 24

# Scan (prefix sums)
[1, 2, 3, 4] ⍀           # → [1, 3, 6, 10]

# Combined: zip then sum
[1, 2, 3] ⊗ [4, 5, 6] Σ  # → 32 (dot product)

# Works with expressions
(1 + 2) × (3 + 4) Σ      # Parse error (operators on scalars)
[1, 2] ⊗ [3, 4] Σ        # → 11

# Precedence: postfix after infix
[1, 2, 3] ↦ (λ→ ₀ × 2) Σ # → 12

# =============================================================================
# PRIORITY 3: Type Syntax (parsing only, no checking yet)
# =============================================================================

# Uncertain types (parse test with :ast)
:ast let temp : F64 ± F64 = ⟨μ: 20.0, σ: 0.5⟩

# Uncertainty propagation (runtime tests)
10.5 ± 0.3                              # → 10.5±0.3
(10.0 ± 0.3) + (20.0 ± 0.4)            # → 30±0.5
(5.0 ± 0.1) × (3.0 ± 0.2)             # → 15±1.044...
√(9.0 ± 0.3)                           # → 3±0.05
sin (1.0 ± 0.1)                        # → 0.841...±0.054...

# Refinement types (parse test)
:ast let positive : {x : F64 | x > 0} = 5.0

# Interval types
:ast let bounded : F64⊢[0..100] = 50.0

# =============================================================================
# PRIORITY 4: Field Access with Unicode
# =============================================================================

# Greek letters in field names
let stats = ⟨μ: 10.0, σ: 2.0, n: 100⟩
stats.μ                  # → 10.0
stats.σ                  # → 2.0

# Superscripts (variance)
let variance = ⟨σ²: 4.0⟩
variance.σ²              # → 4.0

# Numeric field access still works
stats.0                  # → 10.0
stats.1                  # → 2.0

# =============================================================================
# PRIORITY 5: Shape Variables (already working)
# =============================================================================

# These parse correctly (test with :ast)
:ast let map : ∀n. (F64 → F64) → [n]F64 → [n]F64 = ...
:ast let dot : ∀n. [n]F64 → [n]F64 → F64 = ...
:ast let outer : ∀m n. [m]F64 → [n]F64 → [m n]F64 = ...

# Shape variables preserved
:ast [n]F64 → [n]F64

# =============================================================================
# PRIORITY 6: Runtime Contracts
# =============================================================================

# Precondition: positive input
╭─ sqrt_safe : F64 → F64
│  ⊢ ₀ > 0
╰─ ⊥sqrt

sqrt_safe 9              # → 3.0 (works)
sqrt_safe (-1)           # → Error: Precondition violated

# Postcondition: verify result
╭─ double : F64 → F64
│  ⊨ ₀ = ₁ × 2
╰─ ₀ × 2

double 5                 # → 10 (postcondition passes)

# Multi-arg contracts
╭─ safe_div : F64 → F64 → F64
│  ⊢ ₀ ≠ 0
╰─ ₁ / ₀

safe_div 10 2            # → 5.0
safe_div 10 0            # → Error: Precondition violated

# Both pre and post
╭─ checked_inc : F64 → F64
│  ⊢ ₀ ≥ 0
│  ⊨ ₀ > ₁
╰─ ₀ + 1

checked_inc 5            # → 6 (all checks pass)
checked_inc (-1)         # → Error: Precondition violated

# =============================================================================
# EXISTING FEATURES (Pre-implementation)
# =============================================================================

# Lambdas with de Bruijn indices
(λ→ ₀ + 1) 5             # → 6

# Multi-arg lambdas
(λ→ λ→ ₀ + ₁) 3 4        # → 7

# Higher-order functions
let twice = λ→ λ→ ₁ (₁ ₀)
twice (λ→ ₀ × 2) 5       # → 20

# Pattern matching
match [1, 2, 3]
  [] → 0
  [x] → x
  [x, y | rest] → x + y

# Let bindings
let x = 5 in
let y = 10 in
x + y                    # → 15

# Arrays and tensors
[1, 2, 3, 4, 5]
[[1, 2], [3, 4]]

# Map and filter
[1, 2, 3, 4] ↦ (λ→ ₀ × 2)        # → [2, 4, 6, 8]
[1, 2, 3, 4] ▸ (λ→ ₀ > 2)       # → [3, 4]

# Composition
let f = λ→ ₀ + 1
let g = λ→ ₀ × 2
let h = f ∘ g
h 5                      # → 11 (5 × 2 + 1)

# Tuples and records
⟨1, 2, 3⟩
⟨x: 1, y: 2, z: 3⟩

# Variants
⟨Left 5 | Right "error"⟩

# =============================================================================
# REPL COMMANDS
# =============================================================================

:help                    # Show help
:type EXPR               # Show type (not implemented yet)
:ast EXPR                # Show parsed AST
:clear                   # Clear environment
:load FILE               # Load definitions from file
:quit                    # Exit REPL

# =============================================================================
# ASCII FALLBACKS (when Unicode not available)
# =============================================================================

# Operators
-:  instead of  ↦   (map)
+:  instead of  ⊕   (concat)
*:  instead of  ⊗   (zip)
+/  instead of  Σ   (sum)
*/  instead of  Π   (product)
\/  instead of  ⍀   (scan)

->  instead of  →   (arrow)
<-  instead of  ←   (back arrow)
=>  instead of  ⤇   (bind)
.:  instead of  ∘   (compose)
|>  instead of  ▸   (filter)

|-  instead of  ⊢   (precondition)
|=  instead of  ⊨   (postcondition)

\   instead of  λ   (lambda)

# Types
forall  instead of  ∀
exists  instead of  ∃

# Booleans
true    instead of  ⊤
false   instead of  ⊥

# Function box (for declarations)
/-  instead of  ╭─
|   for middle lines
\-  instead of  ╰─

# =============================================================================
# COMMON PATTERNS
# =============================================================================

# Define and use function
╭─ factorial : F64 → F64
│  ⊢ ₀ ≥ 0
╰─ match ₀
     0 → 1
     n → n × factorial (n - 1)

factorial 5              # → 120

# Recursive fibonacci
╭─ fib : F64 → F64
╰─ match ₀
     0 → 0
     1 → 1
     n → fib (n - 1) + fib (n - 2)

fib 10                   # → 55

# Array processing pipeline
let nums = [1, 2, 3, 4, 5, 6]
nums ▸ (λ→ ₀ > 2)        # Filter > 2
     ↦ (λ→ ₀ × ₀)        # Square
     Σ                   # Sum
# → 77 (3² + 4² + 5² + 6²)

# Dot product
let dot = λ→ λ→ ₀ ⊗ ₁ Σ
dot [1, 2, 3] [4, 5, 6]  # → 32

# Matrix-vector multiply (simplified)
let matvec = λ→ λ→ ₀ ↦ (λ→ ₀ ⊗ ₂ Σ)
# Usage: matvec matrix vector

# =============================================================================
# DEBUGGING TIPS
# =============================================================================

# 1. Use :ast to see parsed structure
:ast λ→ ₀ + 1

# 2. Check resolution with simple expressions
:ast let x = 5 in x

# 3. Test operators incrementally
[1, 2] Σ                 # First just sum
[1, 2] ⊗ [3, 4] Σ        # Then zip and sum

# 4. Contracts help debug
╭─ mystery : F64 → F64
│  ⊨ ₀ > ₁               # Assert output > input
╰─ ₀ × 2

mystery 5                # If this fails, postcondition tells you why

# =============================================================================
# GOTCHAS
# =============================================================================

# 1. De Bruijn indices count from innermost
λ→ λ→ ₀ + ₁              # ₀ = inner arg, ₁ = outer arg

# 2. Postfix operators bind after infix
[1, 2] + [3, 4] Σ        # Parse error (+ expects scalars)
([1, 2] + [3, 4]) Σ      # Would work if + supported arrays

# 3. Continuation prompt is part of input
# Don't try to edit the dots - they're prompt characters

# 4. Contracts checked at runtime only
# No static verification yet (Priority 7)

# 5. Type annotations parse but don't check yet
let x : F64 = "string"   # Parses fine, but :type doesn't work yet
```

## Expected Output Summary

| Test | Expected Result |
|------|-----------------|
| `[1,2,3] Σ` | `6` |
| `[1,2,3] ⊗ [4,5,6] Π` | `17280` |
| Multi-line let | `25` |
| `stats.μ` | `10.0` |
| `sqrt_safe (-1)` | `Error: Precondition violated` |
| `safe_div 10 0` | `Error: Precondition violated` |
| `double 5` | `10` (postcondition passes) |
| `:ast {x : F64 | x > 0}` | Parsed AST (no error) |

## What's NOT Tested Here

- Type checking (Priority 7) - parser only
- Static contract proving - runtime only
- Native compilation - interpreter only
- Effect checking - parsed but not enforced
- Refinement solving - no Z3 yet

## Performance Notes

Current interpreter performance (approximate):
- Simple arithmetic: ~1μs
- Function call: ~5μs
- Contract check: +10-50μs
- Array operation: O(n) time, no SIMD

These will improve dramatically with compilation (Priority 7+).
