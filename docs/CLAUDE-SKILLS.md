# Claude Skills: Working with Goth

> **Version:** 2026-02-02T14:47:26Z

This guide describes how Claude (or other LLMs) can effectively generate and modify Goth code using the AST-first workflow.

## Overview

Goth is a functional programming language designed for LLM interaction. The key insight is that **syntax is just serialization** - the canonical representation is the Abstract Syntax Tree (AST), which can be expressed as JSON.

### Why AST-First?

1. **No syntax errors** - JSON structure is validated, not parsed
2. **Unambiguous** - No precedence confusion, no whitespace issues
3. **Bidirectional** - JSON → Goth syntax → JSON roundtrips perfectly
4. **LLM-native** - Structured output is what LLMs do best

## Installation

Download the latest binary for your platform:

```bash
# Linux x86_64
curl -L https://github.com/sigilante/goth/releases/download/latest/goth-linux-x86_64.tar.gz | tar xz
sudo mv goth /usr/local/bin/

# macOS ARM64 (M1/M2/M3)
curl -L https://github.com/sigilante/goth/releases/download/latest/goth-macos-arm64.tar.gz | tar xz
sudo mv goth /usr/local/bin/

# Or run directly without installing
./goth program.goth arg1 arg2
```

For tagged releases (e.g., `v0.1.0`):

```bash
# Replace 'latest' with the version tag
curl -L https://github.com/sigilante/goth/releases/download/v0.1.0/goth-linux-x86_64.tar.gz | tar xz
```

**Available platforms:**
| Platform | Artifact |
|----------|----------|
| Linux x86_64 | `goth-linux-x86_64.tar.gz` |
| macOS ARM64 | `goth-macos-arm64.tar.gz` |

## Workflow

```bash
# 1. Convert Goth source to JSON for LLM editing
goth --to-json program.goth > program.json

# 2. LLM generates/edits the JSON AST

# 3. Validate and render back to Goth syntax
goth --from-json program.json --check --render

# 4. Execute directly from JSON
goth --from-json program.json <args>
```

## AST Structure

### Module

A Goth module contains declarations:

```json
{
  "name": "module_name",
  "decls": [
    { "Fn": { ... } },
    { "Let": { ... } }
  ]
}
```

### Function Declaration

```json
{
  "Fn": {
    "name": "factorial",
    "type_params": [],
    "signature": {
      "Fn": [
        { "Prim": "I64" },
        { "Prim": "I64" }
      ]
    },
    "effects": [],
    "constraints": [],
    "preconditions": [],
    "postconditions": [],
    "body": { ... }
  }
}
```

### Expressions

#### Literals
```json
{ "Lit": { "Int": 42 } }
{ "Lit": { "Float": 3.14 } }
{ "Lit": "True" }
{ "Lit": "False" }
{ "Lit": { "String": "hello" } }
```

#### De Bruijn Indices (Variable References)
```json
{ "Idx": 0 }  // Most recent binding (₀)
{ "Idx": 1 }  // Second most recent (₁)
```

#### Named References (for globals)
```json
{ "Name": "factorial" }
```

#### Lambda
```json
{
  "Lam": { ... body ... }
}
```

#### Application
```json
{
  "App": [
    { "Name": "f" },
    { "Idx": 0 }
  ]
}
```

#### Binary Operations
```json
{
  "BinOp": [
    "Add",
    { "Idx": 1 },
    { "Idx": 0 }
  ]
}
```

Operators: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`, `Eq`, `Neq`, `Lt`, `Gt`, `Leq`, `Geq`, `And`, `Or`, `PlusMinus`, `Write`, `Read`, `Map`, `Filter`, `Compose`, `Bind`, `Concat`, `ZipWith`

#### Unary Operations
```json
{
  "UnaryOp": [
    "Neg",
    { "Idx": 0 }
  ]
}
```

Operators: `Neg`, `Not`, `Sqrt`, `Floor`, `Ceil`, `Sum`, `Prod`

#### Let Binding
```json
{
  "Let": {
    "pattern": { "Var": "x" },
    "type_": null,
    "value": { "Lit": { "Int": 5 } },
    "body": {
      "BinOp": ["Add", { "Idx": 0 }, { "Lit": { "Int": 1 } }]
    }
  }
}
```

#### If-Then-Else
```json
{
  "If": {
    "cond": { "BinOp": ["Lt", { "Idx": 0 }, { "Lit": { "Int": 2 } }] },
    "then_": { "Idx": 0 },
    "else_": { ... }
  }
}
```

#### Array
```json
{
  "Array": [
    { "Lit": { "Int": 1 } },
    { "Lit": { "Int": 2 } },
    { "Lit": { "Int": 3 } }
  ]
}
```

#### Tuple
```json
{
  "Tuple": [
    { "Lit": { "Int": 1 } },
    { "Lit": "True" }
  ]
}
```

## De Bruijn Index Convention

Goth uses De Bruijn indices for variable binding. The key rule:

**Index 0 = most recently bound variable**

For a 2-argument function `f(a, b)`:
- `₀` refers to `b` (second/last argument)
- `₁` refers to `a` (first argument)

For nested let bindings:
```
let x = 1 in
  let y = 2 in
    x + y
```
- `₀` = `y` (most recent)
- `₁` = `x`

After each `let` binding, existing indices shift up by 1.

## Types

### Primitive Types
```json
{ "Prim": "I64" }   // 64-bit integer
{ "Prim": "F64" }   // 64-bit float
{ "Prim": "Bool" }  // Boolean
{ "Prim": "Char" }  // Character
{ "Prim": "String" }
```

### Function Types
```json
{
  "Fn": [
    { "Prim": "I64" },      // argument type
    { "Prim": "I64" }       // return type
  ]
}
```

Multi-argument: `A → B → C` is `Fn[A, Fn[B, C]]`

### Type Variables
```json
{ "Var": "α" }
{ "Var": "T" }
```

## Common Patterns

### Recursive Function
```json
{
  "Fn": {
    "name": "factorial",
    "signature": { "Fn": [{ "Prim": "I64" }, { "Prim": "I64" }] },
    "body": {
      "If": {
        "cond": { "BinOp": ["Leq", { "Idx": 0 }, { "Lit": { "Int": 1 } }] },
        "then_": { "Lit": { "Int": 1 } },
        "else_": {
          "BinOp": ["Mul",
            { "Idx": 0 },
            { "App": [
              { "Name": "factorial" },
              { "BinOp": ["Sub", { "Idx": 0 }, { "Lit": { "Int": 1 } }] }
            ]}
          ]
        }
      }
    }
  }
}
```

### Two-Argument Function
```json
{
  "Fn": {
    "name": "add",
    "signature": {
      "Fn": [
        { "Prim": "I64" },
        { "Fn": [{ "Prim": "I64" }, { "Prim": "I64" }] }
      ]
    },
    "body": {
      "BinOp": ["Add", { "Idx": 1 }, { "Idx": 0 }]
    }
  }
}
```
Note: `₁` = first arg, `₀` = second arg

### Main Function
```json
{
  "Fn": {
    "name": "main",
    "signature": { "Fn": [{ "Prim": "I64" }, { "Prim": "I64" }] },
    "body": {
      "App": [
        { "Name": "myFunction" },
        { "Idx": 0 }
      ]
    }
  }
}
```

## Examples

See `examples/json/` for complete JSON AST examples:
- `ackermann.json` - Ackermann function (deep recursion)
- `fibonacci.json` - Fibonacci sequence
- `sieve.json` - Prime sieve using filter
- `tsp_anneal.json` - Simulated annealing structure

## Contracts (Pre/Postconditions)

Goth supports runtime-checked contracts:

### Preconditions (`⊢`)

Conditions that must be true when a function is called:

```goth
# Square root requires non-negative input
╭─ sqrtSafe : F64 → F64
│  ⊢ ₀ >= 0.0
╰─ √₀
```

### Postconditions (`⊨`)

Conditions that must be true when a function returns. In postconditions, `₀` refers to the result:

```goth
# Absolute value guarantees non-negative result
╭─ absPost : I64 → I64
│  ⊨ ₀ >= 0
╰─ if ₀ < 0 then 0 - ₀ else ₀
```

### Combined Contracts

```goth
# Factorial with both pre and post conditions
╭─ factContract : I64 → I64
│  ⊢ ₀ >= 0
│  ⊨ ₀ >= 1
╰─ if ₀ < 2 then 1 else ₀ × factContract (₀ - 1)
```

ASCII fallbacks: `|-` for `⊢`, `|=` for `⊨`

## Uncertain Values (Error Propagation)

Goth supports values with associated uncertainty:

```goth
# Create an uncertain measurement
╭─ measure : F64 → F64 → (F64 ± F64)
╰─ ₁ ± ₀

# Usage: measure 10.5 0.3 → 10.5±0.3
```

The `±` operator creates uncertain values that track measurement error.

## Generation Tips for Claude

1. **Start with the type signature** - this determines arity and index usage
2. **Count arguments for indices** - 1-arg function uses `₀`, 2-arg uses `₀,₁`, etc.
3. **Track let bindings** - each `let` shifts all indices up by 1
4. **Use `Name` for recursion** - reference the function by name, not index
5. **Validate structure** - use `goth --from-json file.json --check --render`

## FAQ / Clarifications

### 1. List/Array Construction

**Static arrays:**
```goth
[1, 2, 3, 4, 5]           # Literal array
```

**Dynamic range generation:**
```goth
iota 5                    # → [0 1 2 3 4] (0 to n-1)
⍳ 5                       # Same (APL-style)
range 1 5                 # → [1 2 3 4] (start to end-1)
```

**Array type in JSON AST:**
```json
{"Array": [{"Prim": "I64"}]}   // Type: [n]I64
```

Arrays are tensors internally. No cons/nil - use `iota`/`range` with map/filter.

### 2. Higher-Order Functions

Built-in operators (not standalone functions):

| Operation | Unicode | ASCII | Example |
|-----------|---------|-------|---------|
| Map | `↦` | `-:` | `[1,2,3] ↦ (λ→ ₀ × 2)` → `[2 4 6]` |
| Filter | `▸` | `\|>_` | `[1,2,3,4,5] ▸ (λ→ ₀ > 2)` → `[3 4 5]` |
| Fold | `⌿` | `fold` | `⌿ (λ→ λ→ ₁ + ₀) 0 [1,2,3]` → `6` |
| Sum | `Σ` | `+/` | `Σ [1,2,3,4,5]` → `15` |
| Product | `Π` | `*/` | `Π [1,2,3,4,5]` → `120` |
| Compose | `∘` | `.:` | `f ∘ g` (f after g) |
| Bind | `⤇` | `=>>` | Monadic bind |
| Write | `▷` | `\|>` | `"hello" ▷ stdout` or `"data" ▷ "/tmp/out.txt"` |

**Fold** is a 3-argument curried function: `fold f acc arr`. The function `f` takes `(accumulator, element)` and returns the new accumulator. Inside the fold lambda, `₁` = accumulator, `₀` = current element.

**In JSON AST:**
```json
{"BinOp": ["Map", {"Array": [...]}, {"Lam": ...}]}
{"BinOp": ["Filter", {"Array": [...]}, {"Lam": ...}]}
{"BinOp": ["Write", {"Lit": {"String": "hello"}}, {"Name": "stdout"}]}
{"BinOp": ["Write", {"Lit": {"String": "data"}}, {"Lit": {"String": "/tmp/out.txt"}}]}
```

### 3. Input/Output

**`print`** — prints any value followed by a newline, returns `()`:
```goth
print("Hello, world!")
```

**`▷` (write operator)** — writes to a stream or file, returns `()`. No newline is appended:
```goth
"hello" ▷ stdout      # write to stdout (no newline)
"error" ▷ stderr      # write to stderr
"data"  ▷ "/tmp/f.txt" # write to file
```

**`⧏` (readBytes)** — reads n bytes from a file path, returns `[n]I64` array of byte values (0-255):
```goth
⧏ 8 "/dev/urandom"    # read 8 bytes of entropy
readBytes 4 "/tmp/f"   # ASCII fallback
```

**`⧐` (writeBytes)** — writes a byte array to a file path:
```goth
⧐ [72, 101, 108] "/tmp/out"   # write bytes
writeBytes [0, 255] "/tmp/bin" # ASCII fallback
```

`stdout` and `stderr` are built-in stream constants. When the RHS of `▷` is a stream, the content is written to that stream. When it is a string, it is treated as a file path.

**In JSON AST:**
```json
// print (function application)
{"App": [{"Name": "print"}, {"Lit": {"String": "hello"}}]}

// ▷ stdout (BinOp Write with stream)
{"BinOp": ["Write", {"Lit": {"String": "hello"}}, {"Name": "stdout"}]}

// ▷ file (BinOp Write with path)
{"BinOp": ["Write", {"Lit": {"String": "data"}}, {"Lit": {"String": "/tmp/f.txt"}}]}
```

**Array output:** Printed as `[1 2 3]` (space-separated, no commas)

**Tuple output:** Printed as `⟨1, 2, 3⟩`

**Uncertain output:** Printed as `10.5±0.3`

**Main function inputs:** Always via command-line args, converted to the signature type. For array inputs, generate them internally from scalar args:
```goth
# To get [1..n], use:
╭─ main : I64 → [?]I64
╰─ iota ₀ ↦ (λ→ ₀ + 1)    # [1, 2, ..., n]
```

### 4. Benchmark File Format

**Preferred:** Goth syntax (`.goth` files) for readability.

**JSON AST:** Use for LLM reliability testing or programmatic generation.

**Symbol preference:** Unicode preferred (`╭─`, `₀`, `λ`, `→`), but ASCII fallbacks work (`/-`, `_0`, `\`, `->`).

### 5. Type System

**Array types:**
```json
{"Tensor": [{"Prim": "I64"}, [5]]}     // [5]I64 - fixed size
{"Tensor": [{"Prim": "F64"}, ["?"]]}   // [?]F64 - dynamic size
```

**Tuple types:**
```json
{"Tuple": [{"Prim": "I64"}, {"Prim": "Bool"}]}  // ⟨I64, Bool⟩
```

**Uncertain types:**
```json
{"Uncertain": [{"Prim": "F64"}, {"Prim": "F64"}]}  // F64 ± F64
```

**Polymorphic functions:** Use `type_params` and `Var` types:
```json
{
  "type_params": [{"name": "A"}, {"name": "B"}],
  "signature": {"Fn": [{"Var": "A"}, {"Var": "B"}]}
}
```

### 6. Contracts

**Optional** - add when mathematically meaningful.

**Test violations:** Use `"expected_error": "precondition"` in test cases:
```json
{"input": [-1], "expected_error": "precondition"}
```

### 7. Built-in Operators

**Math functions:** `√` (sqrt), `Γ` (gamma), `ln`, `exp`, `sin`, `cos`, `tan`, `abs`, `⌊` (floor), `⌈` (ceil), `log₁₀`, `log₂`

**Array operations:** `len`, `Σ` (sum), `Π` (product), `↦` (map), `▸` (filter), `⌿` (fold), `iota`, `range`, `⊕` (concat)

**Bitwise operations:** `bitand`, `bitor`, `⊻`/`bitxor`, `shl`, `shr` — all curried `I64 → I64 → I64`

**Byte I/O:** `⧏`/`readBytes` (`I64 → String → [n]I64`), `⧐`/`writeBytes` (`[n]I64 → String → ()`)

### 8. Execution Environment

**Float tolerance:** Use `"tolerance": 0.001` or `"reltol": 1e-9` in test cases.

**Recursion depth:** TCO supported for tail-recursive functions. Non-tail recursion limited to ~1000 depth. Use accumulator patterns for deep recursion:
```goth
# Tail-recursive (unlimited depth)
╭─ sumAcc : I64 → I64 → I64
╰─ if ₁ < 1 then ₀ else sumAcc (₁ - 1) (₀ + ₁)
```

### 9. Extending Benchmarks

**Current gaps to consider:**
- Data structures (trees, graphs - would need ADTs)
- More numeric algorithms (differentiation)
- Property-based contract testing

**Stick to:** Pure functional algorithms that fit the current type system.

## Standard Library: Random Numbers

The `stdlib/random.goth` module provides seeded PRNG using xorshift64. Import with `use "stdlib/random.goth"`.

**Key design pattern:** All RNG functions return `⟨value, nextSeed⟩` tuples. Thread the seed through sequential calls:

```goth
use "stdlib/random.goth"

╭─ main : () → ⟨F64, F64, F64⟩
╰─ let seed = entropy ⟨⟩
   in let ⟨v1, s1⟩ = randFloat seed
   in let ⟨v2, s2⟩ = randFloat s1
   in let ⟨v3, s3⟩ = randFloat s2
   in ⟨v1, v2, v3⟩
```

**Available functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `entropy` | `() → I64` | Seed from `/dev/urandom` |
| `bytesToSeed` | `[8]I64 → I64` | Pack 8 bytes into seed |
| `xorshift64` | `I64 → I64` | Raw state transition |
| `randFloat` | `I64 → ⟨F64, I64⟩` | Uniform [0, 1) |
| `randFloatRange` | `F64 → F64 → I64 → ⟨F64, I64⟩` | Uniform [lo, hi) |
| `randInt` | `I64 → I64 → I64 → ⟨I64, I64⟩` | Uniform [lo, hi] |
| `randBool` | `F64 → I64 → ⟨Bool, I64⟩` | Boolean with probability p |
| `randNormal` | `I64 → ⟨F64, I64⟩` | Standard normal (Box-Muller) |
| `randGaussian` | `F64 → F64 → I64 → ⟨F64, I64⟩` | Normal(mean, stddev) |
| `randFloats` | `I64 → I64 → ⟨[n]F64, I64⟩` | Bulk uniform floats |
| `randInts` | `I64 → I64 → I64 → I64 → ⟨[n]I64, I64⟩` | Bulk uniform integers |
| `randNormals` | `I64 → I64 → ⟨[n]F64, I64⟩` | Bulk normal values |

**De Bruijn index note:** When using `let ⟨v, s⟩ = randFloat seed in ...`, the destructuring introduces 2 bindings, so raw De Bruijn indices in the body shift by 2. Named variables (`v`, `s`, `seed`) are unaffected — prefer named references when possible.

## Standard Library: Cryptography

The `stdlib/crypto.goth` module provides pure-Goth implementations of cryptographic hash functions and encoding. All algorithms are implemented entirely in Goth using bitwise primitives — no FFI or Rust crate dependencies. Import with `use "stdlib/crypto.goth"`.

**Key design pattern:** All hash functions take a `String` and return a hex-encoded `String`. Internally, strings are converted to byte arrays via `bytes`, processed through block-oriented compression functions using `fold`, and rendered back to hex strings.

```goth
use "stdlib/crypto.goth"

╭─ main : String → String
╰─ let contents = readFile ₀
   in let _ = print ("SHA-256: " ⧺ sha256 ₀)
   in let _ = print ("MD5:    " ⧺ md5 ₁)
   in "BLAKE3: " ⧺ blake3 ₂
```

**Public functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `sha256` | `String → String` | SHA-256 hash (NIST FIPS 180-4), arbitrary-length input |
| `sha256Bytes` | `[n]I64 → String` | SHA-256 from raw byte array |
| `md5` | `String → String` | MD5 digest (RFC 1321), arbitrary-length input |
| `blake3` | `String → String` | BLAKE3 hash (single chunk, ≤ 64 bytes) |
| `base64EncodeStr` | `String → String` | Base64 encode a string (RFC 4648) |
| `base64Encode` | `[n]I64 → String` | Base64 encode a byte array |
| `base64Decode` | `String → [n]I64` | Base64 decode to byte array |
| `hexEncode` | `[n]I64 → String` | Hex-encode a byte array |

**Internal helpers** (also exported, useful for custom hash constructions):

| Function | Description |
|----------|-------------|
| `add32`, `not32`, `rotr32`, `rotl32` | 32-bit masked arithmetic and rotation |
| `byteToHex`, `wordsToHex`, `wordsToHexLE` | Hex encoding (big-endian and little-endian) |
| `blockToWords`, `blockToWordsLE` | Byte-to-word packing (BE and LE) |
| `sha256Pad`, `md5Pad` | Message padding with length encoding |
| `sha256Compress`, `md5Compress`, `blake3Compress` | Single-block compression functions |
| `aset` | Functional array element update: `aset arr idx val` |

**Implementation notes for LLMs:**
- Goth integers are i128 internally; all 32-bit hash operations must mask with `bitand X 4294967295` after every addition, rotation, or shift
- SHA-256 uses big-endian word packing; MD5 and BLAKE3 use little-endian
- BLAKE3 is limited to single-chunk messages (≤ 64 bytes); SHA-256 and MD5 handle arbitrary length via multi-block processing
- De Bruijn index tracking is critical in hash compression loops — annotate every `let` with a comment showing the current index state (e.g., `# ₀=ki ₁=roundIdx ₂=st ₃=schedule`)
- Use `fold` over `iota N` for round-based processing; inside the fold lambda, `₁` = accumulator, `₀` = round index

## Standard Library: JSON

The `stdlib/json.goth` module provides a pure-Goth JSON parser and serializer. JSON values are represented as tagged 2-tuples `⟨tag, payload⟩` where the integer tag discriminates the type. Import with `use "stdlib/json.goth"`.

**Key design pattern:** JSON values use integer-tagged tuples rather than algebraic data types (Goth's `enum` constructors are not yet wired into the evaluator). Use constructors to build values and predicates/extractors to inspect them. The parser returns a Result triple `⟨Bool, Json, String⟩` following the `result.goth` convention.

**Value representation:**

| Tag | JSON type | Constructor | Payload |
|-----|-----------|-------------|---------|
| 0 | null | `jsonNull ⟨⟩` | `0` (unused) |
| 1 | boolean | `jsonBool ⊤` | `Bool` |
| 2 | number | `jsonNum 3.14` | `F64` |
| 3 | string | `jsonStr "hi"` | `String` |
| 4 | array | `jsonArr [...]` | `[n]Json` |
| 5 | object | `jsonObj [⟨"k", v⟩]` | `[n]⟨String, Json⟩` |

**Parse and extract fields:**

```goth
use "stdlib/json.goth"

╭─ main : () → String
╰─ let r = parseJson "{\"name\":\"Goth\",\"version\":1}"
   in if ¬(r.0) then "error: " ⧺ r.2
   else let json = r.1
   in let name = jsonGet "name" json
   in if name.0 then asStr name.1 else "unknown"
```

**Build and serialize:**

```goth
use "stdlib/json.goth"

╭─ main : () → String
╰─ let obj = jsonObj [
     ⟨"x", jsonNum 1.0⟩,
     ⟨"y", jsonArr [jsonBool ⊤, jsonNull ⟨⟩]⟩
   ]
   in toJson obj
# {"x":1,"y":[true,null]}
```

**Roundtrip (parse → serialize → re-parse → compare):**

```goth
use "stdlib/json.goth"

╭─ main : () → String
╰─ let input = "{\"a\":1,\"b\":[2,3],\"c\":{\"d\":true}}"
   in let r1 = parseJson input
   in if ¬(r1.0) then "parse error"
   else let s1 = toJson r1.1
   in let r2 = parseJson s1
   in if ¬(r2.0) then "re-parse error"
   else if strEq s1 (toJson r2.1) then "PASS" else "FAIL"
```

**Public functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `jsonNull` | `() → Json` | Construct null |
| `jsonBool` | `Bool → Json` | Construct boolean |
| `jsonNum` | `F64 → Json` | Construct number |
| `jsonStr` | `String → Json` | Construct string |
| `jsonArr` | `[n]Json → Json` | Construct array |
| `jsonObj` | `[n]⟨String, Json⟩ → Json` | Construct object |
| `parseJson` | `String → ⟨Bool, Json, String⟩` | Parse JSON; `⟨⊤, val, ""⟩` or `⟨⊥, 0, errMsg⟩` |
| `toJson` | `Json → String` | Serialize to compact JSON |
| `showJson` | `Json → String` | Alias for `toJson` |
| `jsonGet` | `String → Json → ⟨Bool, Json⟩` | Lookup key in object (Option) |
| `jsonIndex` | `ℤ → Json → ⟨Bool, Json⟩` | Index into array (Option) |
| `jsonKeys` | `Json → [n]String` | Object keys |
| `jsonValues` | `Json → [n]Json` | Object values |
| `jsonLen` | `Json → ℤ` | Array/object length |
| `jsonType` | `Json → String` | `"null"`, `"bool"`, `"number"`, `"string"`, `"array"`, `"object"` |
| `isNull` | `Json → Bool` | Tag = 0 |
| `isBool` | `Json → Bool` | Tag = 1 |
| `isNum` | `Json → Bool` | Tag = 2 |
| `isStr` | `Json → Bool` | Tag = 3 |
| `isArr` | `Json → Bool` | Tag = 4 |
| `isObj` | `Json → Bool` | Tag = 5 |
| `asBool` | `Json → Bool` | Extract boolean payload (unsafe) |
| `asNum` | `Json → F64` | Extract number payload (unsafe) |
| `asStr` | `Json → String` | Extract string payload (unsafe) |
| `asArr` | `Json → [n]Json` | Extract array payload (unsafe) |
| `asObj` | `Json → [n]⟨String, Json⟩` | Extract object entries (unsafe) |

**Implementation notes for LLMs:**
- JSON values are just `⟨Int, payload⟩` tuples — check tag with `.0`, extract with `.1`
- `jsonGet` and `jsonIndex` return Option tuples `⟨Bool, Json⟩` — check `.0` before accessing `.1`
- The parser uses recursive descent with position threading: each internal parser function takes `[n]Char → ℤ → ⟨Bool, value, ℤ⟩` (chars, position → ok?, value, new position)
- De Bruijn indices in the parser go deep — the code annotates every `let` with an index comment (e.g., `# ₀=p1 ₁=acc ₂=pos ₃=chars`)
- Chained field access like `x.1.1` does **not** work — the parser reads `.1.1` as the float literal `1.1`. Use a helper function or intermediate `let` to access nested tuple fields
- The `fromChars` primitive is needed for building strings from char arrays (e.g., in the string parser and number parser)
- `⧺` on arrays appends (snoc): `acc ⧺ [elem]` builds arrays element by element during parsing
- `escapeJsonStr` handles `"`, `\`, `\n`, `\t`, `\r`; `\uXXXX` escapes are replaced with `?`
- Number serialization uses Goth's `toString` — formatting may differ from strict JSON (e.g., `1.0` renders as `1`)

## Complex Numbers

Goth has first-class complex number support. Complex values are `Value::Complex(re, im)` internally, with type annotation `ℂ` or `Complex`.

**Literal syntax:**

```goth
3 + 4𝕚              # Complex(3, 4) — Unicode imaginary unit U+1D55A
3 + 4i              # Same — ASCII fallback
𝕚                    # Complex(0, 1) — standalone imaginary unit
5𝕚                   # Complex(0, 5)
```

**Auto-promotion chain:** `ℤ → F → ℂ → ℍ`. When one operand is Complex, the other is promoted automatically.

**Arithmetic:** All standard operators work: `+`, `-`, `×`, `/`, `^`. Multiplication uses the standard formula `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.

**Math functions extended to complex:**

| Function | Complex behavior |
|----------|-----------------|
| `sqrt` | Polar-form square root; `√(0 - 4)` → `2𝕚` |
| `exp` | `e^(a+bi) = e^a(cos b + i sin b)` — Euler's formula |
| `ln` | `ln|z| + i·arg(z)` |
| `sin` | `sin(a)cosh(b) + i·cos(a)sinh(b)` |
| `cos` | `cos(a)cosh(b) - i·sin(a)sinh(b)` |
| `abs` | Returns `F64`: modulus `√(re² + im²)` |

> `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` are **not** extended to complex — they accept real arguments only.

**Decomposition primitives:**

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `re` | `ℂ → F` | Real part (`re(3+4𝕚) = 3`) |
| `im` | `ℂ → F` | Imaginary part (`im(3+4𝕚) = 4`) |
| `conj` | `ℂ → ℂ` | Conjugate (`conj(3+4𝕚) = 3-4𝕚`) |
| `arg` | `ℂ → F` | Argument in radians (`arg(𝕚) = π/2`) |

For plain real numbers, `re` returns the value, `im` returns `0.0`, `conj` returns the value unchanged.

**Examples:**

```goth
exp(π × 𝕚) + 1              # ≈ 0 (Euler's identity)
√(0 - 4)                     # 2𝕚
(3 + 4𝕚) × (1 + 2𝕚)         # -5+10𝕚
abs(3 + 4𝕚)                  # 5.0
```

**Display format:** `3 + 4𝕚`, `3 - 4𝕚`, `5𝕚` (pure imaginary), `3` (pure real). Zero displays as `0`.

## Quaternions

Goth supports quaternion arithmetic with the Hamilton product. Quaternion values are `Value::Quaternion(w, i, j, k)` with type annotation `ℍ` or `Quaternion`.

**Literal syntax:**

```goth
𝕚                    # Quaternion(0,1,0,0) — also Complex if no j/k context
𝕛                    # Quaternion(0,0,1,0) — Unicode U+1D55B
𝕜                    # Quaternion(0,0,0,1) — Unicode U+1D55C
3𝕛                   # Quaternion(0,0,3,0)
1 + 2𝕚 + 3𝕛 + 4𝕜    # Quaternion(1,2,3,4)
```

ASCII fallbacks: `j`, `k` (e.g., `3j`, `4k`).

**Arithmetic:** `+`, `-`, `×`, `/`, negation. Multiplication is the **non-commutative** Hamilton product:

| Rule | Result |
|------|--------|
| `𝕚 × 𝕛` | `𝕜` |
| `𝕛 × 𝕜` | `𝕚` |
| `𝕜 × 𝕚` | `𝕛` |
| `𝕛 × 𝕚` | `-𝕜` |
| `𝕚 × 𝕛 × 𝕜` | `-1` |

Division: `a / b = a × conj(b) / |b|²`.

**Decomposition:**

| Primitive | Quaternion behavior |
|-----------|-------------------|
| `re` | Returns `F64`: the scalar (w) component |
| `im` | Returns `⟨F, F, F⟩`: the (i, j, k) components as a 3-tuple |
| `conj` | Returns `ℍ`: `(w, -i, -j, -k)` |
| `abs` | Returns `F64`: norm `√(w² + i² + j² + k²)` |

> Quaternion `exp`, `ln`, `sqrt`, `pow`, `sin`, `cos` are **not** implemented — only basic arithmetic, `abs`, `conj`, `re`, `im`.

**Example:**

```goth
# Hamilton's identity
╭─ main : I64 → ℍ
│  ⊨ re(₀) = 0.0 - 1.0
╰─ 𝕚 × 𝕛 × 𝕜
```

## Linear Algebra

Goth provides built-in linear algebra primitives operating on rank-2 tensors (matrices) and rank-1 tensors (vectors).

### Core Operations

| Name | Signature | Description |
|------|-----------|-------------|
| `·`, `dot` | `[n]F64 → [n]F64 → F64` | Dot product |
| `norm` | `[n]F64 → F64` | Euclidean (L2) norm |
| `matmul` | `[m n]F64 → [n p]F64 → [m p]F64` | Matrix multiplication (inner dims must match) |
| `⍉`, `transpose` | `[m n]α → [n m]α` | Matrix transpose |

### Matrix Utilities

| Name | Signature | Description |
|------|-----------|-------------|
| `trace`, `tr` | `[n n]F64 → F64` | Sum of diagonal elements |
| `det` | `[n n]F64 → F64` | Determinant (LU decomposition) |
| `inv` | `[n n]F64 → [n n]F64` | Matrix inverse (errors on singular) |
| `diag` | `[n]F64 → [n n]F64` | Vector → diagonal matrix |
| `diag` | `[n n]F64 → [n]F64` | Matrix → diagonal vector |
| `eye` | `ℤ → [n n]F64` | Identity matrix of size n |

`diag` is dual-mode: rank-1 input builds a diagonal matrix, rank-2 input extracts the diagonal.

### Eigenvalue / Eigenvector

| Name | Signature | Description |
|------|-----------|-------------|
| `eig` | `[n n]F64 → [n]F64\|[n]ℂ` | Eigenvalues |
| `eigvecs` | `[n n]F64 → ⟨[n]α, [n n]α⟩` | Eigenvalues + eigenvector matrix |

**Algorithm:** Hessenberg reduction via Householder similarity transforms, then QR iteration with Wilkinson shifts and Givens rotations. Max iterations: `100 × n`.

**Return type:** If all eigenvalues are real (imaginary part < 1e-12), returns Float tensors. If any are complex, returns tensors with Complex values. Eigenvalues sorted by real part descending.

`eigvecs` returns `⟨eigenvalues, eigenvector_matrix⟩` where columns of the matrix are eigenvectors. Eigenvectors computed via inverse iteration (real eigenvalues) or 2n×2n real embedding for complex pairs.

### Linear System Solvers

| Name | Signature | Description |
|------|-----------|-------------|
| `solve` | `[n n]F64 → [n]F64 → [n]F64` | Solve Ax = b (LU, default) |
| `solveWith` | `[n n]F64 → [n]F64 → String → [n]F64` | Solve with method choice |

`solveWith` methods:
- `"lu"` — Doolittle LU with partial pivoting (same as `solve`)
- `"qr"` — Householder QR; handles overdetermined (least-squares) systems where m ≥ n

**Examples:**

```goth
solve [[2,1],[5,3]] [4,7]                          # [5, -6]
solveWith [[1,1],[1,2],[1,3]] [1,2,2] "qr"         # least-squares
det [[6,1,1],[4,0-2,5],[2,8,7]]                    # -306
eig [[2,1],[1,2]]                                   # [3, 1]
eig [[0,0-1],[1,0]]                                 # [0+1𝕚, 0-1𝕚]
eigvecs (diag [5,3])                                # ⟨[5, 3], [[1,0],[0,1]]⟩
inv [[1,2],[3,4]]                                   # [[-2, 1], [1.5, -0.5]]
```

**Implementation notes for LLMs:**
- All matrix operations require rank-2 tensors; pass `[[1,2],[3,4]]` not `[1,2,3,4]`
- `outer` and `inner` are declared in the PrimFn enum but **not implemented** — they produce a runtime "not implemented" error
- Singularity threshold is `1e-12` for inverse, `1e-15` for LU pivot
- Negative literal syntax quirk: Goth parses `0-2` as subtraction; `-2` in a matrix literal may be parsed as unary negation depending on context, so `0-2` is safest in array literals
- `diag` auto-detects rank: `diag [1,2,3]` builds a 3×3 diagonal matrix; `diag [[1,0],[0,2]]` extracts `[1,2]`

## Goth Syntax ↔ JSON

| Goth Syntax | JSON AST |
|-------------|----------|
| `42` | `{"Lit":{"Int":42}}` |
| `₀` | `{"Idx":0}` |
| `λ→ ₀ + 1` | `{"Lam":{"BinOp":["Add",{"Idx":0},{"Lit":{"Int":1}}]}}` |
| `f x` | `{"App":[{"Name":"f"},{"Name":"x"}]}` |
| `if c then a else b` | `{"If":{"cond":...,"then_":...,"else_":...}}` |
| `let x = e in b` | `{"Let":{"pattern":{"Var":"x"},"value":...,"body":...}}` |
| `[1,2,3]` | `{"Array":[...]}` |
| `⟨a,b⟩` | `{"Tuple":[...]}` |

## Error Handling

When JSON is malformed or semantically invalid:

```bash
$ goth --from-json bad.json --check
Error: invalid JSON AST: missing field `body`

$ goth --from-json semantic_error.json --check
OK: parsed 1 declaration(s)
Type error: Cannot unify types: I64 and Bool
```

The `--check` flag validates types before execution (off by default for interpreter use).

## Benchmarking & Testing

### Running Tests

```bash
# Run all benchmark tests
python benchmark/run_tests.py

# Run specific category
python benchmark/run_tests.py --category basic

# Verbose output
python benchmark/run_tests.py --verbose

# JSON output for analysis
python benchmark/run_tests.py --json > results.json
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| basic | 10 | Simple arithmetic, comparisons |
| recursion | 10 | Factorial, fibonacci, GCD, Ackermann |
| algorithms | 6 | Primes, binary search, modpow |
| numeric | 7 | Gamma function, Taylor series, Newton-Raphson |
| higher_order | 6 | Fold, compose, pipeline |

### Generating Prompts

```bash
# Generate prompt for specific test
python benchmark/prompts/prompt_generator.py factorial

# Generate prompt for JSON AST output
python benchmark/prompts/prompt_generator.py --format json gcd

# List all available tests
python benchmark/prompts/prompt_generator.py --list

# Generate all prompts
python benchmark/prompts/prompt_generator.py --all
```

### Prompt Formats

**Syntax prompt** - asks LLM to generate Goth syntax directly:
```
Implement the following function in Goth:
Function: factorial
Signature: I64 → I64
...
```

**JSON AST prompt** - asks LLM to generate structured JSON:
```
Generate a Goth JSON AST for the following function:
Function: factorial
Signature: I64 → I64
...
```

### Python Baselines

Reference implementations in `benchmark/baselines/python/`:
- `basic.py` - identity, abs, max, etc.
- `recursion.py` - factorial, fibonacci, GCD, Ackermann
- `algorithms.py` - isPrime, modpow, binary search
- `numeric.py` - gamma, Taylor series, Newton-Raphson
- `higher_order.py` - fold, compose, pipeline

### Metrics Collected

See `benchmark/METRICS.md` for full details:

1. **First-attempt success rate** - correct on first try
2. **Parse success rate** - syntactically valid
3. **Semantic correctness** - produces right output
4. **Iterations to correct** - attempts needed
5. **Token count** - compared to Python baseline
6. **De Bruijn errors** - index mistakes

## Tail Call Optimization

Goth supports TCO via trampoline for tail-recursive functions:

```goth
# NOT tail recursive (overflows on deep recursion)
╭─ sum : I64 → I64
╰─ if ₀ < 1 then 0 else ₀ + sum (₀ - 1)

# IS tail recursive (constant stack)
╭─ sumAcc : I64 → I64 → I64
╰─ if ₁ < 1 then ₀ else sumAcc (₁ - 1) (₀ + ₁)
```

See `examples/tco/` for naive vs TCO versions of common functions.
