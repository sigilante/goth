# 𝔊𝔬𝔱𝔥
### the `goth` language for machine spirits

`goth` is an LLM-native programming language designed for efficient code generation, editing, and comprehension by large language models.

This document records the design process and decisions for `goth`.  The code samples are not up-to-date with the latest release but do illustrate the design philosophy and history.

Future work:
* posits, bfloat16, TensorFloat-32
* GPU intrinsics
* SIMD intrinsics
* Memory layout
* terminal with Ani-like assistant

---

I've wondered idly for a while about what a language optimized for LLMs rather than humans would look like.  Now, the priests of the machine spirits still need to be able to read the language, so it needs a legible representation (as opposed to a blast of static).  But we can play with other parameters in order to make it legible.  In fact, a very few people have started to play this game (like [LMQL](https://lmql.ai/)).  So, what the heck, let's just build a new language from scratch over the weekend.

Our design constraints include prioritizing what LLMs are good at while avoiding what hurts their context and accuracy.  These things are favorable for LLMs:

- Pattern completion over structural templates
- Reasoning about explicit dataflow
- Type-level constraint satisfaction
- Compositional/algebraic thinking
- Dense semantic compression (token economy, efficient compression)

These things are "hard" in some sense for LLMs, since they have sliding finite context windows for memories rather than human-like deep recall.

- Boilerplate (imports, class scaffolding, semicolons) wastes tokens
- Implicit state and side effects requires simulating execution
- Distant syntactic dependencies, like matching braces 200 lines apart
- Naming inconsistencies, like `getString`, `get_string`, `GetString`
- Separation of spec from implementation

I had a great working session with Claude on this.  What we devised has echoes of APL and Plankalkül.  (Plankalkül is a strange duck:  in some way, it's the first programming language, having been devised by Konrad Zuse in the late 1930s for his machine.  It featured a two-dimensional syntax and only had a primitive data type of a single bit.)

Some other desiderata for the language:

1. Homoiconic.  Code is data.
2. Concatenative.  Stack-based computation.  (At this point, it's like a Forth-y Lisp-y APL.)
3. Shape-first types.  Types are tensor shapes + algebraic constraints.  `[3 4] → [4 5] → [3 5]`  is a type signature.
4. Spec–implementation unification.  There are not separate test files.  The contracts are the function.  Pre- and post-conditions generate tests automatically.
5. Unicode operators.  Single glyphs describe common operations.  An IDE can convert digraphs to glyphs seamlessly (following Mathematica).  Think of `∘` compose, `⊛` apply, `⊗` product, `⊕` sum, `↦` map, `⤇` bind, `⊢` assert/precondition, `⊨` satisfies/postcondition.  Operator overloading is supported for language extension.
6. Explicit effects.  The language is pure by default.  Effects are capabilities passed in for I/O, mutation, randomness handled in types.
7. Structural not textual.  The AST is the primary representation.  Syntax is a serialization format rather than the canonical source of truth.

In working with Nock and writing https://nock.is, I have thought extensively about combinators and their relationship to systems.  Essentially, a set of combinators is the fundamental toolkit for carving reality at the joints.  Thus one of the most important things to get right is the set of primitives.

So, ladies and gentlemen, please meet (for the first time in our material plane), the `goth` language.  `goth` is a homoiconic, statically-typed language targeting the LLVM IR (actually MLIR).

```
# Dot product of two double-precision vectors
╭─ dot : [n]F64 → [n]F64 → F64
╰─ ₀ ⊗ ₁ Σ
```

Here's a multi-line function with conditions:

```
goth[0]›
╭─ normalize : [n]F → [n]F
│  ⊢ len ₀ > 0
│  ⊨ abs(norm ₀ - sqrt(len ₁)) < 0.0001
╰─ let arr ← ₀ ;
       n ← len arr ;
       μ ← sum arr / n ;
       σ ← sqrt(sum ((arr ↦ (λ→ ₀ - μ)) ↦ (λ→ ₀ × ₀)) / n)
   in (arr ↦ (λ→ ₀ - μ)) ↦ (λ→ ₀ / σ)
goth[1]› normalize([1.0, 2.0, 3.0, 4.0, 5.0])
[-1.414213562373095 -0.7071067811865475 0 0.7071067811865475 1.414213562373095]
goth[2]› norm(normalize([1.0, 2.0, 3.0, 4.0, 5.0]))
2.2360679774997894
```

Here is a higher-order function:
```
let xs ← [1 2 3] in
  # Map a function over xs
  xs ↦ (λ→ ₀ × 2)
```

These are equivalents of each other, defining an outer product:
```
let matrix_op ← λ→ λ→ ₀ + ₁ × ₂ in
#                    ↑   ↑   ↑
#                    inner λ arg (row element)
#                        outer λ arg (row)  
#                            matrix_op's own captured... wait

# Clearer:
╭─ outer : A → (B → C)
╰─ λ→ ₀ + ₁
#     ↑   ↑
#     inner arg (B)
#         outer arg (A)
```

## Primitive Operators

| Glyph | Name          | Semantics           |
| ----- | ------------- | ------------------- |
| `↦`   | map           | functor lift        |
| `⤇`   | bind          | monadic chain       |
| `∘`   | compose       | f ∘ g = λx. f(g(x)) |
| `▸`   | filter        | predicate select    |
| `⊕`   | sum           | coproduct / concat  |
| `⊗`   | product       | tensor / zip        |
| `Σ`   | fold          | reduce with +       |
| `Π`   | fold          | reduce with ×       |
| `⊢`   | precondition  | must hold on entry  |
| `⊨`   | postcondition | must hold on exit   |
| `□`   | pure          | no effects          |
| `◇`   | effect        | capability required |

What do you get from this set?  Here's what Claude and I like about it:
- APL-like battle-tested array of tooling.
- Highly dense tokens.  Expressing an algorithm with fewer tokens means that the LLM can see more program in context.  Dense symbolic notation reduces what we could call “parse load”.
- Unambiguous parsing.  No lookahead is necessary and there are no context-sensitive grammar quirks.
- The spec is the test.  Correct code is generated by algebraically satisfying the postcondition.
- Shape errors will be caught statically:  `[3 4] → [5 5]` is a type error which is instantly legible.
- Compositional reasoning mirrors function pipelines naturally.

Efficient semantic compression takes place along three axes:
1. Elision, what can be inferred.  Implicit is better than explicit when inference can take place trivially.  Effects are pure unless annotated.  Arguments are positionally marked with de Bruijn indices rather than by names.
2. Factoring, shared structure (write-once code).  The AST is a DAG, not a tree.  Common subexpressions get names or indices once.  If multiple references occur we reference rather than redefining.  Macros can quote and unquote.
3. Density, more meaning per glyph.  In fact, we can even have expansions while still preserving semantic atoms.  `⊛Σ` is a single unit (a catamorphism)

| Pattern       | Glyph | Expansion      |
| ------------- | ----- | -------------- |
| map-reduce    | `⊛Σ`  | `↦ f ∘ Σ`      |
| filter-map    | `▸↦`  | `▸ p ∘ ↦ f`    |
| zip-with      | `⊗f`  | `zip ∘ ↦ f`    |
| scan          | `⍀`   | prefix fold    |
| outer product | `∘.`  | APL-style      |
| broadcast     | `⁺`   | shape coercion |

The thing to track isn't tokens per se, but "bits" per semantic unit.  The relevant semantic units are things like function application, binding introduction, type constraint, shape assertion, effect annotation, and so forth.  The sort-of goal, in that it's hyperreal, is to approach the Kolmogorov complexity for algorithms.  Most languages build quicksort in 50–100 semantic units, but if we can whittle that down to, say, 20 then we're really cooking with gas.

## Type System

Next, let's look at some data representations.  Tensors are the universal container:
```
[1 2 3]           # vector, shape [3]
[[1 2] [3 4]]     # matrix, shape [2 2]
[2 3 4]⊢0         # zeros, shape [2 3 4]
[2 3 4]⊢ρ         # random, shape [2 3 4], requires ◇rand
```

Records are positional tuples with optional labels for humans (since we're using Unicode we aren't limited to overloading square brackets):
```
⟨42, "alice", ⊤⟩           # tuple
⟨age: 42, name: "alice"⟩   # labeled (sugar, compiles to positional)
```

```
# Definition
Point ≡ ⟨x: F64, y: F64⟩

# Construction
p ← ⟨x: 3.0, y: 4.0⟩

# Access (by name for humans, compiles to positional)
p.x        # or p₀
p.y        # or p₁

# Update (functional)
p⊢{x: 5.0}   # returns new Point with x changed
```

Variants are sum types:
```
⟨L x | R y⟩                # either
⟨None | Some v⟩            # option
# Pattern match: ⊳ L → f | R → g
```

Typeclasses are behavioral interfaces:
```
class Num τ where
  (+) : τ → τ → τ
  (×) : τ → τ → τ
  zero : τ
  one : τ

impl Num F64 where
  (+) ← float_add
  (×) ← float_mul
  zero ← 0.0
  one ← 1.0

impl Num [n]F64 where
  (+) ← ⊕         # elementwise
  (×) ← ⊗         # elementwise
  zero ← [n]⊢0
  one ← [n]⊢1
```

(Why no objects?  A few reasons:
1. Method dispatch is implicit control flow (which is hard for LLMs to trace)
2. Inheritance creates non-local reasoning
3. Typeclasses are explicit: you see the constraint, you know the interface
4. Records + typeclasses cover the same ground with more clarity
Remember, we're building this for a context window.)

Finally, if the grammar is a value in the language, then we can type macros:
```
Grammar ← ⟨
  Expr  → ⟨Lam Expr | App Expr Expr | Idx ℕ | Lit Val | Op Prim⟩
  Val   → ⟨Tensor Shape Data | Tuple [Val] | ...⟩
  Shape → [ℕ]
  Prim  → ⟨+ | × | ↦ | Σ | ...⟩
⟩
```

Tensors are homogeneous containers with a shape. The element type is parametric:

```
[n]T        # vector of n elements of type T
[m n]T      # matrix
[a b c]T    # 3-tensor

T ∈ { F64, F32, I64, U8, Bool, Char, ... , ⟨UserType⟩ }
```

Primitive types include things like these:

| Type   | Repr                         | Notes            |
| ------ | ---------------------------- | ---------------- |
| `Fx`   | IEEE 754 (or something else) | x = 16,32,64,128 |
| `Ix`   | signed int                   | x = 8,16,32,64   |
| `Ux`   | unsigned                     | x = 8,16,32,64   |
| `Bool` | 1 bit                        | packed in arrays |
| `Char` | Unicode scalar               | 32-bit           |
| `Byte` | U8                           | raw bytes        |

Strings are `[n]Char`.  (`Char` is a 32-bit Unicode scalar (UCS-4).)  UTF-8 encoding is an explicit conversion to bytes, not the internal representation:
```
"hello" : [5]Char
"hello"⊢utf8 : [5]Byte   # 5 bytes

"héllo" : [5]Char        # 5 codepoints
"héllo"⊢utf8 : [6]Byte   # 6 bytes (é = 2 bytes)

"🔥" : [1]Char           # 1 codepoint  
"🔥"⊢utf8 : [4]Byte      # 4 bytes
```

Compound element types via tuples:
```
# Array of 3D points
[100]⟨F64, F64, F64⟩

# AoS ↔ SoA is a view transformation
[100]⟨F64, F64, F64⟩ ⊢soa ≡ ⟨[100]F64, [100]F64, [100]F64⟩
```

Byte arrays for foreign data:
```
[n]Byte              # raw buffer
[n]Byte ⊢as [m]F32   # reinterpret (requires n = 4m)
```

Some other bits and pieces:
```
= value equality (vanilla comparison)
≡ structural equality (α-equivalent, ignoring sharing)
≣ referential equality (same node in the DAG)
```

```
3 + 4 = 7        # ⊤ : Bool (runtime comparison)

Point ≡ ⟨x: F64, y: F64⟩    # type alias (compile-time)

let a ← [1 2 3] in
let b ← a in
  a ≣ b          # ⊤ (same reference)
  a = [1 2 3]    # ⊤ (same value)
  a ≣ [1 2 3]    # ⊥ (different nodes)
```

Arguments are indicated positionally using De Bruijn indices, `₀ ₁ ₂ ...`.  Let's look at our first function definition thus far:

```
╭─ normalize : [n]F → [n]F
│  ⊨ ‖result‖ = 1
╰─ let μ ← Σ ₀ / n ;
       σ ← √(Σ (₀ - μ)² / n)
   in (₀ - μ) / σ
```

De Bruijn indices only work from 0–9 so for high-arity functions you switch to explicit naming:
```
λ⟨a b c d e f g h i j k⟩→ ...
```

Subscripts nest:
```
λ→ λ→ ₀ + ₁
#      ↑   ↑
#      inner outer
```

`₀` is always the innermost binding.

The language is homoiconic, so `⟨⟩` quotes while `‹›` unquotes (splices).

There is a terse form and an explicit form.  This echoes Hoon's tall form and wide form.  These are examples of equivalent pairs:

```
# Terse: filter primes, square, sum
xs ▸prime ↦(²) Σ

# Explicit: same thing
do xs
  ▸ λ→ prime? ₀
  ↦ λ→ ₀ × ₀
  Σ
end

# Hybrid: terse with one explicit lambda
xs ▸prime ↦(λ→ ₀ × ₀) Σ
```

```
# Nested map-reduce
data ↦(↦f Σ) Σ      # sum of row sums after mapping f

# Clearer with block
do data
  ↦ do ₀
       ↦ f
       Σ
     end
  Σ
end
```

Casting types:

- `⊢as`   compile-time view cast (zero-cost, shape proven)
- `⊢as?`  runtime view cast (returns Option, validates)
- `⊢as!`  runtime view cast (panics on failure)

This function parses a runtime wire:
```
╭─ parse_matrix : [n]Byte → ?⟨[r c]F32, ParseError⟩
│  ◇io
╰─ do ₀
     # Read header (first 8 bytes = two U32 for dimensions)
     ⊢slice[0:8] ⊢as? ⟨U32, U32⟩ → ⟨r, c⟩
     
     # Validate remaining length
     ⊢ n = 8 + 4×r×c
     
     # Cast body
     ⊢slice[8:] ⊢as? [r c]F32
   end
```

Type-level existentials for unknown shapes:
```
∃n. [n]F32     # "some array of floats, length unknown"
```

Once you validate, you can "open" the existential:
```
╭─ process : (∃n. [n]F32) → F64
╰─ unpack [n]xs ← ₀ in    # n now in scope as type variable
     xs Σ / n
```

The unpack binds both the shape witness `n` and the value `xs`.

Foreign buffers with lifetime:
```
[n]Byte⊢foreign⟨'a⟩    # borrowed from FFI, lifetime 'a
```

No copies until you explicitly `⊢clone`.  This lifetime management prevents use-after-free.

## Interval Arithmetic and Error Propagation

This is where it gets powerful. An interval type tracks the _range_ of possible values:

```
# Type says: returns float in [0, 1]
╭─ sigmoid : F64 → F64⊢[0..1]
╰─ 1 / (1 + e^(-₀))

# Type says: array index is in valid range
╭─ safe_index : [n]τ → U64⊢[0..n) → τ
╰─ ₀[₁]   # no bounds check needed, proven safe
```

**Interval arithmetic propagates:**

```
x : F64⊢[0..1]
y : F64⊢[0..1]

x + y : F64⊢[0..2]      # addition
x × y : F64⊢[0..1]      # multiplication
x - y : F64⊢[-1..1]     # subtraction
1/x   : F64⊢[1..∞]      # requires x⊢(0..1], prevents div-by-zero
```

**Error propagation** (this is the formal methods angle):

```
# Measured value with uncertainty
μ ± σ  ≡  F64⊢[μ-3σ..μ+3σ]   # 3-sigma interval

# Operations propagate uncertainty
╭─ add_uncertain : (F64 ± F64) → (F64 ± F64) → (F64 ± F64)
╰─ (₀.μ + ₁.μ) ± √(₀.σ² + ₁.σ²)
```

Or more elegantly, make `±` a type constructor:

```
τ±  ≡  ⟨μ: τ, σ: τ⊢[0..∞)⟩   # value with uncertainty

# Arithmetic on uncertain values is overloaded
(+) : F64± → F64± → F64±
```

**Compile-time bounds checking:**

```
╭─ mat_index : [m n]τ → U64⊢[0..m) → U64⊢[0..n) → τ
╰─ ₀[₁, ₂]

# Usage
A : [100 200]F64
i : U64 ← user_input    # type is U64⊢[0..∞)

A[i, 0]                  # TYPE ERROR: i not proven < 100

# Fix: validate
if i < 100 then
  # here i : U64⊢[0..100)
  A[i, 0]                # OK
else
  error "out of bounds"
```

The conditional _refines_ the type in each branch. This is flow-sensitive typing.

### Constraints (Type Classes)

```
where τ: Ord          # has ordering
where τ: Num          # has +, ×, 0, 1
where τ: Ring         # Num + negation
where τ: Field        # Ring + division
where σ: Broadcastable σ'   # shapes can broadcast
where n > 0           # shape constraint
where n = m × k       # shape equality
```

**Example with multiple constraints:**

```
╭─ normalize : [n]τ → [n]τ
│  where τ: Field
│  where n > 0
│  ⊨ ‖result‖ = 1
╰─ ₀ / ‖₀‖
```

### Refinement Types

The `{x : τ | P(x)}` form lets you embed arbitrary predicates:

```
# Sorted array
Sorted [n]τ  ≡  {xs : [n]τ | ∀i j. i < j → xs[i] ≤ xs[j]}

# Non-empty
NonEmpty [n]τ  ≡  {xs : [n]τ | n > 0}

# Probability distribution (sums to 1)
Prob [n]  ≡  {xs : [n]F64⊢[0..1] | Σxs = 1}
```

**Spec integration:**

```
╭─ qsort : [n]τ → Sorted [n]τ
│  where τ: Ord
│  ⊨ permutation ₀ result
╰─ ...
```

The return type `Sorted [n]τ` is a refinement. The postcondition `⊨ permutation` adds another constraint. The compiler/prover must verify both.

## Effects

### Effect Rows

Effects compose:

```
□              pure, no effects
◇io            I/O (file, network, console)
◇mut           local mutation
◇rand          randomness (requires seed/RNG)
◇div           possible divergence (non-termination)
◇exn⟨E⟩        may throw exception of type E
◇ffi⟨'a⟩       foreign call with lifetime

# Combined
◇io ∪ ◇rand    I/O and randomness
```

**Effect polymorphism:**

```
╭─ map : (τ → σ⊢ε) → [n]τ → [n]σ⊢ε
╰─ ...

# If the mapped function is pure, map is pure
# If the mapped function has ◇io, map has ◇io
```

### Summary: A Typed Expression

```
╭─ train_step : Model → Batch → Model⊢◇rand
│  where Model: Differentiable
│  ⊢ batch.size > 0
│  ⊨ loss(result, batch) ≤ loss(₀, batch)   # loss decreases
╰─ let grads ← ∇(loss, ₀, ₁) ;
       lr ← 0.01 ± 0.001                      # uncertain learning rate
   in ₀ - lr × grads
```

This says:

- Takes Model and Batch, returns Model
- Requires randomness effect
- Model must be differentiable (has gradient)
- Batch must be non-empty
- Postcondition: loss improves
- Implementation uses interval arithmetic on learning rate

## Comments

```
# line comment

#[ 
   block comment
   #[ nestable ]#
]#

#- disabled expression -#   # preserves AST node, marked inactive
```

Example:
```
╭─ f : T → T
╰─ impl_v2 ₀
   #- impl_v1 ₀ -#    # still in AST, not executed
```

## Compiling (& Compression Redux)

The AST is the source of truth.  Text representations can vary:

```
# Dense (for LLM consumption, remove all whitespace)
╭─f:[n]T→[n]T╰─₀↦(λ→₀×2)

# Pretty-printed (for human reading)
╭─ f : [n]T → [n]T
╰─ ₀ ↦ (λ→ ₀ × 2)

# Verbose (for pedagogy)
╭─ f : [n]T → [n]T
╰─ ₀ ↦ (
     λ→ ₀ × 2
   )
```

There are three equivalent representations in the system:

|Format|Extension|Use case|
|---|---|---|
|Text|`.goth`|Human authoring/reading|
|JSON AST|`.gast`|Tooling, diffs, transforms|
|Binary AST|`.gbin`|LLM I/O, compilation, storage|

These are isomorphic and the compiler will accept any of them.  Instead of emitting text and hoping it parses:
```
"╭─ f : [n]T → [n]T\n╰─ ₀ ↦ (λ→ ₀ × 2)"
```

the LLM can emit a structured AST directly (here JSON but can also be binary):
```
{
  "fn": {
    "sig": {"args": [{"tensor": {"shape": ["n"], "elem": "T"}}], 
            "ret": {"tensor": {"shape": ["n"], "elem": "T"}}},
    "body": {"app": [{"op": "map"}, 
                     {"idx": 0}, 
                     {"lam": {"app": [{"op": "mul"}, {"idx": 0}, {"lit": 2}]}}]}
  }
}
```

The parser is really a text→AST deserializer.  The prettyprinter goes from AST→text.  Both are trivial once you have the AST spec.

Here's what the compilation pipeline looks like, in that case:

```
┌──────────────────────────────────────┐
│             GOTH TOOLCHAIN              │
└──────────────────────────────────────┘
                                     
┌─────────┐  ┌─────────┐  ┌─────────┐                      
│  .goth   │  │ .gast   │  │ .gbin   │    ← Source formats  
└────┬────┘  └────┬────┘  └────┬────┘                      
     │             │             │                           
     └────────────┼────────────┘                           
                   ↓                                        
           ┌──────────────┐                                
           │    GOTH AST   │  ← Canonical in-memory repr    
           │     (DAG)     │                                
           └──────┬───────┘                                
                   ↓                                        
           ┌──────────────┐    ┌───────┐                   
           │   Typecheck   │←──→│  Z3   │  ← SMT for        
           │   + Constrain │    └───────┘    intervals,     
           └──────┬───────┘                 shapes,        
                   ↓                         refinements    
           ┌──────────────┐                                
           │   Typed AST   │  ← All types resolved          
           └──────┬───────┘                                
                   ↓                                        
           ┌──────────────┐                                
           │   Monomorph   │  ← Specialize generics         
           │   + Closure   │    Convert closures            
           └──────┬───────┘                                
                   ↓                                        
           ┌──────────────┐                                
           │    GOTH MIR   │  ← Low-level, explicit         
           └──────┬───────┘                                
                   ↓                                        
           ┌──────────────┐                                
           │      MLIR     │  ← tensor, affine, scf         
           └──────┬───────┘                                
                   ↓                                        
           ┌──────────────┐                                
           │    LLVM IR    │                                
           └──────┬───────┘                                
                   ↓                                        
           ┌──────────────┐                                
           │    Machine    │                                
           └──────────────┘
```

Finally, we're going to integrate SMT Z3 into the compiler pipeline so that we have proofs over the AST statements at compile time.

The ultimate goal is for `goth` to not only be bootstrapping but generates its own macros and DSLs seamlessly.

