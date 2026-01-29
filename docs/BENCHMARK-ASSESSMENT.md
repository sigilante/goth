# The Experiment                                                                                               
                                                                                                             
Goth asks a genuinely novel question: what if the programmer is a statistical model rather than a human? The 
answer it proposes — De Bruijn indices, Unicode density, tensor-shaped types, contracts-as-spec,             
AST-as-source-of-truth — is coherent and draws intelligently from lambda calculus, APL, and dependent type   
theory.

## What the Benchmarks Actually Show

```
┌─────────────┬──────────────┬───────────┐                                                                   
│    Model    │    Format    │ Pass Rate │                                                                   
├─────────────┼──────────────┼───────────┤                                                                   
│ Grok Expert │ .goth syntax │ 64%       │                                                                   
├─────────────┼──────────────┼───────────┤                                                                   
│ Grok Expert │ JSON AST     │ 50%       │                                                                   
├─────────────┼──────────────┼───────────┤                                                                   
│ Devstral 1  │ .goth syntax │ 0%        │                                                                   
├─────────────┼──────────────┼───────────┤                                                                   
│ Devstral 2  │ .goth syntax │ 47%       │                                                                   
└─────────────┴──────────────┴───────────┘                                                                   
```

Three findings stand out:                                                                                    
                                                                                                             
1. De Bruijn indices work. Index errors were <10% of failures. The central hypothesis — that positional      
binding reduces scoping bugs — is supported. LLMs handle ₀, ₁, ₂ without much confusion, even in nested      
contexts.                                                                                                    
2. JSON AST performed worse than text syntax. This contradicts the "structured output is easier" thesis. The 
currying format for App nodes was a systematic trap (31% of JSON failures). LLMs are better at               
pattern-completing text than constructing well-formed trees.                                                 
3. Most errors are API surface knowledge, not structural confusion. Using mod instead of %, == instead of =, 
hallucinating nonexistent builtins. These are documentation gaps, not design failures. Devstral2 would jump  
from 47% to ~80% with a single == → = fix.                                                                   
                                                                                                             
## Tensions Worth Noting                                                                                        
                                                                                                             
- Token density vs. learnability. The Unicode operators are elegant and genuinely compress programs, but LLMs
  trained on billions of lines of Python/JS/Rust have strong priors toward map, filter, sum. The ASCII
  fallbacks are wisely provided but somewhat undercut the density argument.
- Spec-implementation gap. Many of the most ambitious features — refinement types, effect checking, shape
  unification, Z3 integration — are parsed but not enforced. The type checker exists but isn't wired into the
  CLI pipeline. This means the contracts and shape annotations are currently decoration rather than guarantees.
- The 26-variant AST is moderate complexity — comparable to a Haskell subset. Manageable for an LLM that has
  seen the reference doc, but the interaction of De Bruijn indices with let bindings (which shift all outer
  indices) is where humans and LLMs alike stumble.
                                                                                                             
## The Verdict                                                                                                  
                                                                                                             
Goth succeeds as a falsifiable design experiment. The benchmark infrastructure is well-built for testing     
hypotheses rigorously. The 64% first-attempt success rate on a language that didn't exist two weeks prior,   
with only documentation as reference, is a meaningful signal — most new human-targeted languages would fare  
worse under similar conditions.                                                                              
                                                                                                             
The strongest contribution isn't the language itself but the methodology: build a language around specific   
hypotheses about LLM cognition, then measure. The data suggests that what LLMs need most isn't exotic syntax 
but a small, consistent API surface with unambiguous operator names — a finding that applies far beyond Goth.
                                                                                                             
Whether Goth evolves into a practical tool depends on closing the spec-implementation gap: wiring up the type
checker, enforcing contracts statically, and making shape inference real. The foundation is sound. The      
question is whether the alchemist's gold is in the language design or in the experimental method — and I     
suspect it is the latter.    
