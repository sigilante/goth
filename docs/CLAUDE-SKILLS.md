# Claude Skills: Working with Goth

This guide describes how Claude (or other LLMs) can effectively generate and modify Goth code using the AST-first workflow.

## Overview

Goth is a functional programming language designed for LLM interaction. The key insight is that **syntax is just serialization** - the canonical representation is the Abstract Syntax Tree (AST), which can be expressed as JSON.

### Why AST-First?

1. **No syntax errors** - JSON structure is validated, not parsed
2. **Unambiguous** - No precedence confusion, no whitespace issues
3. **Bidirectional** - JSON → Goth syntax → JSON roundtrips perfectly
4. **LLM-native** - Structured output is what LLMs do best

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

Operators: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`, `Eq`, `Neq`, `Lt`, `Gt`, `Leq`, `Geq`, `And`, `Or`, `PlusMinus`

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

## Generation Tips for Claude

1. **Start with the type signature** - this determines arity and index usage
2. **Count arguments for indices** - 1-arg function uses `₀`, 2-arg uses `₀,₁`, etc.
3. **Track let bindings** - each `let` shifts all indices up by 1
4. **Use `Name` for recursion** - reference the function by name, not index
5. **Validate structure** - use `goth --from-json file.json --check --render`

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

The `--check` flag validates types before execution.
