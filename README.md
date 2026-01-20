# 🌸☠︎ 𝔊𝔬𝔱𝔥 𓂀🖤
### the `goth` language for machine spirits

`goth` is an LLM-native programming language designed for efficient code generation, editing, and comprehension by large language models.

```sh
$ cargo build
$ cargo test
```

© 2026 Sigilante.  Goth is made available under the MIT License.

## Usage

REPL shell:

```sh
$ goth
```

Examples:

```sh
cd goth/crates
cargo run --package goth-ast --example json_ast_demo
cargo run --package goth-ast --example roundtrip_validation
cargo run --package goth-mlir --example end-to-end-example
```

## Status

`goth` is an experiment in language design and exotic syntax.  It targets certain desiderata for a language that is human readable but optimized for LLM characteristics.

`goth` was born on 2026-01-16.  The alpha version of the language and interpreter were completed on 2026-01-17.  Type checking was implemented from 2026-01-17 to 2026-01-18.  The compiler was implemented starting on 2026-01-18.

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

Here's the target compilation pipeline:

```
┌──────────────────────────────────────┐
│            GOTH TOOLCHAIN            │
└──────────────────────────────────────┘
                                     
┌─────────┐  ┌─────────┐  ┌─────────┐                      
│ .goth   │  │ .gast   │  │ .gbin   │    ← Source formats  
└────┬────┘  └────┬────┘  └────┬────┘                      
     │            │            │                           
     └────────────┼────────────┘                           
                  ↓                                        
           ┌──────────────┐                                
           │   GOTH AST   │  ← Canonical in-memory repr    
           │    (DAG)     │                                
           └──────┬───────┘                                
                  ↓                                        
           ┌──────────────┐    ┌───────┐                   
           │  Typecheck   │←──→│  Z3   │  ← SMT for        
           │  + Constrain │    └───────┘    intervals,     
           └──────┬───────┘                 shapes,        
                  ↓                         refinements    
           ┌──────────────┐                                
           │  Typed AST   │  ← All types resolved          
           └──────┬───────┘                                
                  ↓                                        
           ┌──────────────┐                                
           │  Monomorph   │  ← Specialize generics         
           │  + Closure   │    Convert closures            
           └──────┬───────┘                                
                  ↓                                        
           ┌──────────────┐                                
           │   GOTH MIR   │  ← Low-level, explicit         
           └──────┬───────┘                                
                  ↓                                        
           ┌──────────────┐                                
           │     MLIR     │  ← tensor, affine, scf         
           └──────┬───────┘                                
                  ↓                                        
           ┌──────────────┐                                
           │   LLVM IR    │                                
           └──────┬───────┘                                
                  ↓                                        
           ┌──────────────┐                                
           │   Machine    │                                
           └──────────────┘
```
