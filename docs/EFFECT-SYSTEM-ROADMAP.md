# Effect System Roadmap

**Status:** Aspirational — parsed but not enforced (as of v0.1.0)

## Current State

Goth's effect system has complete AST infrastructure but no enforcement. Effect annotations are parsed into the syntax tree and serialized, but ignored by the type checker, evaluator, and codegen backends.

### What Works

- **Lexer/Parser**: The `◇` (Diamond) token and effect names (`io`, `mut`, `rand`, `div`) are recognized in function box middle lines (`│  ◇io`)
- **AST**: `Effect` enum (Pure, Io, Mut, Rand, Div, Exn, Ffi, Custom) with `Effects` set algebra (union, subset, containment) in `goth-ast/src/effect.rs`
- **Type system**: `Type::Effectful(Box<Type>, Effects)` variant exists in `goth-ast/src/types.rs`
- **Function declarations**: `FnDecl.effects: Effects` field stores parsed annotations in `goth-ast/src/decl.rs`
- **Serialization**: Effects serialize correctly via serde

### What Doesn't Work

- **Type checker**: Primitives typed without effects; no effect inference or checking (`goth-check/src/infer.rs`, `builtins.rs`)
- **Evaluator**: I/O primitives dispatch unconditionally; `Closure` doesn't store effects; `EffectNotAllowed` error defined but never raised (`goth-eval/src/prim.rs`, `value.rs`)
- **Parser limitations**: `◇` only works in function box middle lines, not in type signatures
- **Codegen**: Both MLIR and LLVM backends explicitly erase effects during lowering

## Implementation Plan

### Phase 1: Type Checker (core enforcement)

**1.1 Annotate primitive types with effects** (~50 lines)
- File: `goth-check/src/builtins.rs`
- Wrap I/O functions (`print`, `readLine`, `readFile`, `writeFile`, `write`, `flush`, `readKey`, `rawModeEnter`, `rawModeExit`, `sleep`) with `Type::Effectful(..., Effects::single(Effect::Io))`

**1.2 Add effect context to type checker** (~100 lines)
- File: `goth-check/src/context.rs` (or equivalent)
- Add `allowed_effects: Effects` to `Context` struct
- Methods: `with_effect_context()`, `check_effect_allowed()`

**1.3 Effect inference** (~800 lines)
- File: `goth-check/src/infer.rs`
- Modify `infer()` to return `(Type, Effects)` pairs
- Accumulate effects through: function application, let bindings, match arms, if/else branches, binary operators (Write/Read)
- Pure functions: reject effectful sub-computations
- Effectful functions: propagate effects to return type

### Phase 2: Parser Enhancement (~200 lines)

- Support effect annotations in type signatures: `(F64 → F64) ◇io`
- Handle effect composition in type syntax: `String ◇io ◇mut`

### Phase 3: Runtime Support (~130 lines)

**3.1 Store effects on closures** (~30 lines)
- File: `goth-eval/src/value.rs`
- Add `effects: Effects` field to `Closure`

**3.2 Runtime effect checking** (~100 lines)
- File: `goth-eval/src/eval.rs`
- Check allowed effects before dispatching I/O primitives
- Raise `EffectNotAllowed` when violated

### Phase 4: Codegen (optional)

Effects are a static property and can remain erased at codegen. Optionally:
- Emit MLIR function attributes for effects
- Preserve effects through MIR lowering for tooling/analysis

### Phase 5: Testing (~300 lines)

- Pure function calling `print` → type error
- Effectful function calling `print` → allowed
- Effect inference through let/match/if chains
- Effect union across branches
- Polymorphic effects (advanced): `∀ε. (α →ε β) → α →ε β`

## Key Files

| File | Role | Work Needed |
|------|------|-------------|
| `goth-ast/src/effect.rs` | Effect data model | Complete — no changes needed |
| `goth-ast/src/types.rs` | `Type::Effectful` | Complete — no changes needed |
| `goth-ast/src/decl.rs` | `FnDecl.effects` | Complete — no changes needed |
| `goth-check/src/builtins.rs` | Primitive types | Add `Type::Effectful` wrappers |
| `goth-check/src/infer.rs` | Type inference | Add effect inference throughout |
| `goth-check/src/check.rs` | Type checking | Add effect validation |
| `goth-eval/src/value.rs` | Closure definition | Add `effects` field |
| `goth-eval/src/eval.rs` | Evaluation | Add runtime effect checks |
| `goth-eval/src/prim.rs` | Primitive dispatch | Guard I/O behind effect checks |
| `goth-parse/src/parser.rs` | Parsing | Extend effect syntax to types |

## Design Decisions (pending)

1. **Strictness**: Should missing `◇io` on a function that calls `print` be a hard error or a warning?
2. **Inference vs annotation**: Should effects be inferred automatically, or must they be declared?
3. **Polymorphic effects**: Support `∀ε` effect variables? (Significant complexity increase)
4. **Effect handlers**: Algebraic effect handlers (like Koka/Eff) or just tracking/checking?
5. **Backwards compatibility**: How to handle existing `.goth` files that use `print` without `◇io`?

## Estimated Scope

~1,200 lines of new/modified code for core enforcement (Phases 1 + 3 + 5). Parser enhancement (Phase 2) adds ~200 lines. Codegen (Phase 4) is optional.
