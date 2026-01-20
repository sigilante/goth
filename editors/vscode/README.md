# Goth Language Support for VSCode

Syntax highlighting for the Goth programming language.

## Features

- Syntax highlighting for `.goth` files
- Support for Unicode operators (λ, Σ, Π, →, etc.)
- De Bruijn index highlighting (₀, ₁, ₂, etc.)
- Function box notation (╭─ ╰─)
- Auto-closing pairs for brackets and special delimiters

## Installation

### From VSIX (recommended for development)

1. Package the extension:
   ```bash
   cd editors/vscode
   npx vsce package
   ```

2. Install the generated `.vsix` file:
   - Open VSCode
   - Go to Extensions (Ctrl+Shift+X)
   - Click "..." menu → "Install from VSIX..."
   - Select the `.vsix` file

### Manual Installation

Copy the extension folder to your VSCode extensions directory:

```bash
# Linux/macOS
cp -r editors/vscode ~/.vscode/extensions/goth-lang

# Windows
xcopy /E editors\vscode %USERPROFILE%\.vscode\extensions\goth-lang
```

Then restart VSCode.

## Syntax Examples

```goth
-- Function definition with type signature
╭─ factorial : I → I
╰─ if ₀ < 2 then 1 else Π (range 1 (₀ + 1))

-- Lambda expressions
λ x → x × x

-- Tensor operations
Σ (⍳ n)      -- sum of 0..n-1
Π [1,2,3,4]  -- product = 24

-- Math functions
√ 16.0       -- square root
Γ 5.0        -- gamma function (= 24)
⌊ 3.7 ⌋      -- floor
⌈ 3.2 ⌉      -- ceiling
```

## Supported Tokens

### Keywords
`if`, `then`, `else`, `let`, `in`, `match`, `with`, `where`, `do`, `end`, `rec`, `fn`, `type`, `class`, `impl`

### Types
`I64`, `F64`, `Bool`, `Char`, `I`, `F`, `B`, `N`, `ℤ`, `ℕ`, `ℝ`, `𝔹`

### Operators
- Lambda: `λ`, `Λ`, `\`
- Arrows: `→`, `←`, `⇒`, `->`, `<-`, `=>`
- Reduction: `Σ`, `Π`, `⍀`, `+/`, `*/`
- Math: `Γ`, `√`, `gamma`, `sqrt`, `ln`, `exp`, `sin`, `cos`, `abs`
- Arithmetic: `+`, `-`, `×`, `÷`, `^`, `%`, `±`
- Comparison: `<`, `>`, `≤`, `≥`, `=`, `≠`
- Logical: `∧`, `∨`, `¬`, `&&`, `||`, `!`
- Composition: `∘`, `↦`, `▸`, `⤇`
- Tensor: `⍳`, `⊗`, `⊕`
- Spec: `⊢`, `⊨`

### De Bruijn Indices
`₀`, `₁`, `₂`, `₃`, `₄`, `₅`, `₆`, `₇`, `₈`, `₉`

## License

MIT
