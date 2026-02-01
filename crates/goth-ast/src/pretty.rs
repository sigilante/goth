//! Pretty printing for Goth AST
//!
//! Renders AST to Unicode text format (.goth)

use crate::decl::{FnDecl, Module, Decl, TypeDecl, ClassDecl, ImplDecl, LetDecl};
use crate::expr::Expr;
use crate::types::Type;

/// Pretty print configuration
#[derive(Debug, Clone)]
pub struct PrettyConfig {
    /// Use Unicode operators (true) or ASCII fallbacks (false)
    pub unicode: bool,
    /// Indentation string
    pub indent: String,
    /// Maximum line width before wrapping
    pub max_width: usize,
}

impl Default for PrettyConfig {
    fn default() -> Self {
        PrettyConfig {
            unicode: true,
            indent: "  ".to_string(),
            max_width: 100,
        }
    }
}

impl PrettyConfig {
    pub fn ascii() -> Self {
        PrettyConfig {
            unicode: false,
            ..Default::default()
        }
    }

    pub fn compact() -> Self {
        PrettyConfig {
            max_width: usize::MAX,
            ..Default::default()
        }
    }
}

/// Pretty printer
pub struct Pretty {
    config: PrettyConfig,
    output: String,
    current_indent: usize,
}

impl Pretty {
    pub fn new(config: PrettyConfig) -> Self {
        Pretty {
            config,
            output: String::new(),
            current_indent: 0,
        }
    }

    pub fn default() -> Self {
        Pretty::new(PrettyConfig::default())
    }

    /// Pretty print a module
    pub fn print_module(&mut self, module: &Module) -> &str {
        if let Some(name) = &module.name {
            self.write("module ");
            self.write(name);
            self.newline();
            self.newline();
        }

        for (i, decl) in module.decls.iter().enumerate() {
            if i > 0 { self.newline(); }
            self.print_decl(decl);
        }

        &self.output
    }

    /// Pretty print a declaration
    pub fn print_decl(&mut self, decl: &Decl) {
        match decl {
            Decl::Fn(f) => self.print_fn(f),
            Decl::Type(t) => self.print_type_decl(t),
            Decl::Enum(e) => self.print_enum_decl(e),
            Decl::Class(c) => self.print_class(c),
            Decl::Impl(i) => self.print_impl(i),
            Decl::Let(l) => self.print_let_decl(l),
            Decl::Op(o) => self.print_op_decl(o),
            Decl::Use(u) => {
                self.write("use \"");
                self.write(&u.path);
                self.write("\"");
                self.newline();
            }
        }
    }

    /// Pretty print an enum declaration
    pub fn print_enum_decl(&mut self, e: &crate::decl::EnumDecl) {
        self.write("enum ");
        self.write(&e.name);
        for param in &e.params {
            self.write(" ");
            self.write(&param.name);
        }
        self.write(" where ");
        for (i, variant) in e.variants.iter().enumerate() {
            if i > 0 {
                self.write(" | ");
            }
            self.write(&variant.name);
            if let Some(payload) = &variant.payload {
                self.write(" ");
                self.print_type(payload);
            }
        }
        self.newline();
    }

    /// Pretty print a function declaration
    pub fn print_fn(&mut self, f: &FnDecl) {
        // Header line: ╭─ name : sig
        self.write(if self.config.unicode { "╭─ " } else { "/- " });
        self.write(&f.name);
        self.write(" : ");
        self.print_type(&f.signature);
        self.newline();

        // Constraints
        for c in &f.constraints {
            self.write(if self.config.unicode { "│  where " } else { "|  where " });
            self.write(&format!("{:?}", c)); // TODO: proper constraint printing
            self.newline();
        }

        // Preconditions
        for pre in &f.preconditions {
            self.write(if self.config.unicode { "│  ⊢ " } else { "|  |- " });
            self.print_expr(pre);
            self.newline();
        }

        // Postconditions
        for post in &f.postconditions {
            self.write(if self.config.unicode { "│  ⊨ " } else { "|  |= " });
            self.print_expr(post);
            self.newline();
        }

        // Body line: ╰─ expr
        self.write(if self.config.unicode { "╰─ " } else { "\\- " });
        self.print_expr(&f.body);
        self.newline();
    }

    fn print_type_decl(&mut self, t: &TypeDecl) {
        self.write(&t.name);
        self.write(if self.config.unicode { " ≡ " } else { " == " });
        self.print_type(&t.definition);
        self.newline();
    }

    fn print_class(&mut self, c: &ClassDecl) {
        self.write("class ");
        self.write(&c.name);
        self.write(" ");
        self.write(&c.param.name);
        if !c.superclasses.is_empty() {
            self.write(" extends ");
            for (i, sc) in c.superclasses.iter().enumerate() {
                if i > 0 { self.write(", "); }
                self.write(sc);
            }
        }
        self.write(" where");
        self.newline();
        self.indent();
        for m in &c.methods {
            self.write_indent();
            self.write(&m.name);
            self.write(" : ");
            self.print_type(&m.signature);
            self.newline();
        }
        self.dedent();
    }

    fn print_impl(&mut self, i: &ImplDecl) {
        self.write("impl ");
        self.write(&i.class_name);
        self.write(" ");
        self.print_type(&i.target);
        self.write(" where");
        self.newline();
        self.indent();
        for m in &i.methods {
            self.write_indent();
            self.write(&m.name);
            self.write(if self.config.unicode { " ← " } else { " <- " });
            self.print_expr(&m.body);
            self.newline();
        }
        self.dedent();
    }

    fn print_let_decl(&mut self, l: &LetDecl) {
        self.write("let ");
        self.write(&l.name);
        if let Some(ty) = &l.type_ {
            self.write(" : ");
            self.print_type(ty);
        }
        self.write(if self.config.unicode { " ← " } else { " <- " });
        self.print_expr(&l.value);
        self.newline();
    }

    fn print_op_decl(&mut self, o: &crate::decl::OpDecl) {
        self.write("op ");
        self.write(&o.name);
        self.write(" ");
        self.write(&o.glyph);
        self.write(" (");
        self.write(&o.ascii);
        self.write(") : ");
        self.print_type(&o.signature);
        self.write(&format!(" [prec={}, assoc={:?}]", o.precedence, o.assoc));
        self.write(if self.config.unicode { " ← " } else { " <- " });
        self.print_expr(&o.body);
        self.newline();
    }

    /// Pretty print a type
    pub fn print_type(&mut self, ty: &Type) {
        match ty {
            Type::Prim(p) => self.write(&format!("{:?}", p)),
            
            Type::Tensor(shape, elem) => {
                self.write("[");
                for (i, dim) in shape.0.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    match dim {
                        crate::shape::Dim::Const(n) => self.write(&n.to_string()),
                        crate::shape::Dim::Var(v) => self.write(v),
                        _ => self.write("?"),
                    }
                }
                self.write("]");
                self.print_type(elem);
            }
            
            Type::Tuple(fields) if fields.is_empty() => {
                self.write("()");
            }
            
            Type::Tuple(fields) => {
                self.write(if self.config.unicode { "⟨" } else { "(" });
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    if let Some(label) = &field.label {
                        self.write(label);
                        self.write(": ");
                    }
                    self.print_type(&field.ty);
                }
                self.write(if self.config.unicode { "⟩" } else { ")" });
            }
            
            Type::Fn(arg, ret) => {
                // Check if we need parens around arg
                let needs_parens = matches!(**arg, Type::Fn(..));
                if needs_parens { self.write("("); }
                self.print_type(arg);
                if needs_parens { self.write(")"); }
                
                self.write(if self.config.unicode { " → " } else { " -> " });
                self.print_type(ret);
            }
            
            Type::Var(v) => self.write(v),
            
            Type::Effectful(ty, _effects) => {
                self.print_type(ty);
                self.write(if self.config.unicode { "⊢ε" } else { " !! " });
            }
            
            Type::Interval(ty, _interval) => {
                self.print_type(ty);
                self.write(if self.config.unicode { "⊢[..]" } else { " @[..] " });
            }
            
            Type::Refinement { name, base, predicate: _ } => {
                self.write("{");
                self.write(name);
                self.write(" : ");
                self.print_type(base);
                self.write(" | ... }");
            }
            
            Type::Forall(params, body) => {
                self.write(if self.config.unicode { "∀" } else { "forall " });
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { self.write(" "); }
                    self.write(&param.name);
                }
                self.write(". ");
                self.print_type(body);
            }
            
            Type::Exists(params, body) => {
                self.write(if self.config.unicode { "∃" } else { "exists " });
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { self.write(" "); }
                    self.write(&param.name);
                }
                self.write(". ");
                self.print_type(body);
            }
            
            Type::Variant(arms) => {
                self.write(if self.config.unicode { "⟨" } else { "<" });
                for (i, arm) in arms.iter().enumerate() {
                    if i > 0 { self.write(" | "); }
                    self.write(&arm.name);
                    if let Some(payload) = &arm.payload {
                        self.write(" ");
                        self.print_type(payload);
                    }
                }
                self.write(if self.config.unicode { "⟩" } else { ">" });
            }
            
            _ => self.write(&format!("{:?}", ty)),
        }
    }

    /// Pretty print an expression
    pub fn print_expr(&mut self, expr: &Expr) {
        self.print_expr_prec(expr, 0);
    }
    
    /// Print expression with precedence for parenthesization
    fn print_expr_prec(&mut self, expr: &Expr, prec: u8) {
        use crate::literal::Literal;
        
        match expr {
            Expr::Lit(lit) => match lit {
                Literal::Int(n) => self.write(&n.to_string()),
                Literal::Float(f) => self.write(&f.to_string()),
                Literal::True => self.write("true"),
                Literal::False => self.write("false"),
                Literal::String(s) => {
                    self.write("\"");
                    self.write(s);
                    self.write("\"");
                }
                Literal::Char(c) => {
                    self.write("'");
                    self.write(&c.to_string());
                    self.write("'");
                }
                Literal::Unit => self.write("()"),
                Literal::ImagI(x) => { self.write(&x.to_string()); self.write("𝕚"); }
                Literal::ImagJ(x) => { self.write(&x.to_string()); self.write("𝕛"); }
                Literal::ImagK(x) => { self.write(&x.to_string()); self.write("𝕜"); }
            }
            
            Expr::Name(name) => self.write(name),
            
            Expr::Idx(n) => {
                if self.config.unicode {
                    // Convert index to subscript digits (e.g., 12 -> ₁₂)
                    for digit in n.to_string().chars() {
                        let subscript = match digit {
                            '0' => '₀', '1' => '₁', '2' => '₂', '3' => '₃', '4' => '₄',
                            '5' => '₅', '6' => '₆', '7' => '₇', '8' => '₈', '9' => '₉',
                            _ => digit,
                        };
                        self.output.push(subscript);
                    }
                } else {
                    // ASCII: _0, _1, _12, etc.
                    self.write("_");
                    self.write(&n.to_string());
                }
            }
            
            Expr::Lam(body) => {
                let needs_parens = prec > 0;
                if needs_parens { self.write("("); }
                
                self.write(if self.config.unicode { "λ→ " } else { "\\-> " });
                self.print_expr_prec(body, 0);
                
                if needs_parens { self.write(")"); }
            }
            
            Expr::App(f, x) => {
                let app_prec = 10;
                let needs_parens = prec > app_prec;
                if needs_parens { self.write("("); }
                
                self.print_expr_prec(f, app_prec);
                self.write(" ");
                self.print_expr_prec(x, app_prec + 1);
                
                if needs_parens { self.write(")"); }
            }
            
            Expr::Let { pattern, type_, value, body } => {
                let needs_parens = prec > 0;
                if needs_parens { self.write("("); }

                self.write("let ");
                self.print_pattern(pattern);
                if let Some(ty) = type_ {
                    self.write(" : ");
                    self.print_type(ty);
                }
                self.write(if self.config.unicode { " ← " } else { " = " });
                self.print_expr_prec(value, 0);
                self.write(" in ");
                self.print_expr_prec(body, 0);

                if needs_parens { self.write(")"); }
            }
            
            Expr::If { cond, then_, else_ } => {
                let needs_parens = prec > 0;
                if needs_parens { self.write("("); }
                
                self.write("if ");
                self.print_expr_prec(cond, 0);
                self.write(" then ");
                self.print_expr_prec(then_, 0);
                self.write(" else ");
                self.print_expr_prec(else_, 0);
                
                if needs_parens { self.write(")"); }
            }
            
            Expr::BinOp(op, left, right) => {
                let op_prec = binop_prec(op);
                let needs_parens = prec > op_prec;
                if needs_parens { self.write("("); }
                
                self.print_expr_prec(left, op_prec);
                self.write(" ");
                let op_str = binop_str(op, self.config.unicode);
                self.write(&op_str);
                self.write(" ");
                self.print_expr_prec(right, op_prec + 1);
                
                if needs_parens { self.write(")"); }
            }
            
            Expr::UnaryOp(op, operand) => {
                let un_prec = 11;
                let needs_parens = prec > un_prec;
                if needs_parens { self.write("("); }
                
                let op_str = unop_str(op, self.config.unicode);
                self.write(&op_str);
                self.print_expr_prec(operand, un_prec);
                
                if needs_parens { self.write(")"); }
            }
            
            Expr::Tuple(elems) => {
                self.write(if self.config.unicode { "⟨" } else { "(" });
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    self.print_expr_prec(elem, 0);
                }
                self.write(if self.config.unicode { "⟩" } else { ")" });
            }
            
            Expr::Array(elems) => {
                self.write("[");
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    self.print_expr_prec(elem, 0);
                }
                self.write("]");
            }
            
            Expr::Match { scrutinee, arms } => {
                self.write("match ");
                self.print_expr_prec(scrutinee, 0);
                self.write(" of");
                self.newline();
                self.indent();
                for arm in arms {
                    self.write_indent();
                    self.print_pattern(&arm.pattern);
                    self.write(if self.config.unicode { " → " } else { " -> " });
                    self.print_expr_prec(&arm.body, 0);
                    self.newline();
                }
                self.dedent();
            }
            
            Expr::Annot(e, ty) => {
                let needs_parens = prec > 1;
                if needs_parens { self.write("("); }
                
                self.print_expr_prec(e, 1);
                self.write(" : ");
                self.print_type(ty);
                
                if needs_parens { self.write(")"); }
            }
            
            _ => self.write(&format!("{:?}", expr)),
        }
    }
    
    fn print_pattern(&mut self, pat: &crate::pattern::Pattern) {
        use crate::pattern::Pattern;
        
        match pat {
            Pattern::Wildcard => self.write("_"),
            Pattern::Var(None) => self.write("_"),
            Pattern::Var(Some(name)) => self.write(name),
            Pattern::Lit(lit) => {
                use crate::literal::Literal;
                match lit {
                    Literal::Int(n) => self.write(&n.to_string()),
                    Literal::Float(f) => self.write(&f.to_string()),
                    Literal::True => self.write("true"),
                    Literal::False => self.write("false"),
                    Literal::String(s) => {
                        self.write("\"");
                        self.write(s);
                        self.write("\"");
                    }
                    Literal::Char(c) => {
                        self.write("'");
                        self.write(&c.to_string());
                        self.write("'");
                    }
                    Literal::Unit => self.write("()"),
                    Literal::ImagI(x) => { self.write(&x.to_string()); self.write("𝕚"); }
                    Literal::ImagJ(x) => { self.write(&x.to_string()); self.write("𝕛"); }
                    Literal::ImagK(x) => { self.write(&x.to_string()); self.write("𝕜"); }
                }
            }
            Pattern::Tuple(pats) => {
                self.write(if self.config.unicode { "⟨" } else { "(" });
                for (i, pat) in pats.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    self.print_pattern(pat);
                }
                self.write(if self.config.unicode { "⟩" } else { ")" });
            }
            Pattern::Variant { constructor, payload } => {
                self.write(constructor);
                if let Some(pat) = payload {
                    self.write(" ");
                    self.print_pattern(pat);
                }
            }
            Pattern::Or(left, right) => {
                self.print_pattern(left);
                self.write(" | ");
                self.print_pattern(right);
            }
            Pattern::Typed(pat, ty) => {
                self.print_pattern(pat);
                self.write(" : ");
                self.print_type(ty);
            }
            Pattern::Array(pats) => {
                self.write("[");
                for (i, pat) in pats.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    self.print_pattern(pat);
                }
                self.write("]");
            }
            Pattern::ArraySplit { head, tail } => {
                self.write("[");
                for (i, pat) in head.iter().enumerate() {
                    if i > 0 { self.write(", "); }
                    self.print_pattern(pat);
                }
                self.write(" | ");
                self.print_pattern(tail);
                self.write("]");
            }
            Pattern::Guard(pat, _) => {
                self.print_pattern(pat);
                self.write(" if ...");
            }
        }
    }

    // ============ Helpers ============

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn newline(&mut self) {
        self.output.push('\n');
    }

    fn indent(&mut self) {
        self.current_indent += 1;
    }

    fn dedent(&mut self) {
        self.current_indent = self.current_indent.saturating_sub(1);
    }

    fn write_indent(&mut self) {
        for _ in 0..self.current_indent {
            self.output.push_str(&self.config.indent);
        }
    }

    /// Get the output string
    pub fn finish(self) -> String {
        self.output
    }
}

// ============ Convenience Functions ============

/// Pretty print a module with default config
pub fn print_module(module: &Module) -> String {
    let mut p = Pretty::default();
    p.print_module(module);
    p.finish()
}

/// Pretty print a declaration with default config
pub fn print_decl(decl: &Decl) -> String {
    let mut p = Pretty::default();
    p.print_decl(decl);
    p.finish()
}

/// Pretty print a function with default config
pub fn print_fn(f: &FnDecl) -> String {
    let mut p = Pretty::default();
    p.print_fn(f);
    p.finish()
}

/// Pretty print an expression with default config
pub fn print_expr(expr: &Expr) -> String {
    let mut p = Pretty::default();
    p.print_expr(expr);
    p.finish()
}

/// Pretty print a type with default config
pub fn print_type(ty: &Type) -> String {
    let mut p = Pretty::default();
    p.print_type(ty);
    p.finish()
}

// ============ Operator Helpers ============

/// Get operator precedence (higher = tighter binding)
/// Must match BinOp::precedence() in op.rs
fn binop_prec(op: &crate::op::BinOp) -> u8 {
    op.precedence()
}

/// Get operator string representation
fn binop_str(op: &crate::op::BinOp, unicode: bool) -> String {
    use crate::op::BinOp;
    match op {
        BinOp::Add => "+".to_string(),
        BinOp::Sub => "-".to_string(),
        BinOp::Mul => if unicode { "×".to_string() } else { "*".to_string() },
        BinOp::Div => if unicode { "÷".to_string() } else { "/".to_string() },
        BinOp::Mod => "%".to_string(),
        BinOp::Pow => if unicode { "^".to_string() } else { "**".to_string() },
        BinOp::Eq => "=".to_string(),
        BinOp::StructEq => if unicode { "≡".to_string() } else { "==".to_string() },
        BinOp::Neq => if unicode { "≠".to_string() } else { "!=".to_string() },
        BinOp::Lt => "<".to_string(),
        BinOp::Gt => ">".to_string(),
        BinOp::Leq => if unicode { "≤".to_string() } else { "<=".to_string() },
        BinOp::Geq => if unicode { "≥".to_string() } else { ">=".to_string() },
        BinOp::And => if unicode { "∧".to_string() } else { "&&".to_string() },
        BinOp::Or => if unicode { "∨".to_string() } else { "||".to_string() },
        BinOp::PlusMinus => if unicode { "±".to_string() } else { "+/-".to_string() },
        BinOp::Concat => if unicode { "⊕".to_string() } else { "++".to_string() },
        BinOp::Compose => if unicode { "∘".to_string() } else { ".".to_string() },
        BinOp::Map => if unicode { "↦".to_string() } else { "->".to_string() },
        BinOp::Filter => if unicode { "▸".to_string() } else { "|>".to_string() },
        BinOp::Bind => if unicode { "⤇".to_string() } else { ">>=".to_string() },
        BinOp::ZipWith => if unicode { "⊗".to_string() } else { "<*>".to_string() },
        BinOp::Write => if unicode { "▷".to_string() } else { ">!".to_string() },
        BinOp::Read => if unicode { "◁".to_string() } else { "<!".to_string() },
        BinOp::Custom(name) => name.to_string(),
    }
}

/// Get unary operator string
fn unop_str(op: &crate::op::UnaryOp, unicode: bool) -> &'static str {
    use crate::op::UnaryOp;
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Not => if unicode { "¬" } else { "!" },
        UnaryOp::Sqrt => if unicode { "√" } else { "sqrt " },
        UnaryOp::Floor => if unicode { "⌊" } else { "floor " },
        UnaryOp::Ceil => if unicode { "⌈" } else { "ceil " },
        UnaryOp::Sum => if unicode { "Σ" } else { "sum " },
        UnaryOp::Prod => if unicode { "Π" } else { "prod " },
        UnaryOp::Scan => if unicode { "⍀" } else { "scan " },
        UnaryOp::Gamma => if unicode { "Γ" } else { "gamma " },
        UnaryOp::Ln => "ln ",
        UnaryOp::Log10 => if unicode { "log₁₀ " } else { "log10 " },
        UnaryOp::Log2 => if unicode { "log₂ " } else { "log2 " },
        UnaryOp::Exp => "exp ",
        UnaryOp::Sin => "sin ",
        UnaryOp::Cos => "cos ",
        UnaryOp::Tan => "tan ",
        UnaryOp::Asin => "asin ",
        UnaryOp::Acos => "acos ",
        UnaryOp::Atan => "atan ",
        UnaryOp::Sinh => "sinh ",
        UnaryOp::Cosh => "cosh ",
        UnaryOp::Tanh => "tanh ",
        UnaryOp::Abs => "abs ",
        UnaryOp::Sign => "sign ",
        UnaryOp::Round => "round ",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    use crate::literal::Literal;
    use crate::op::BinOp;
    use crate::types::{Type, PrimType};
    use crate::pattern::Pattern;
    
    #[test]
    fn test_print_literal() {
        let expr = Expr::Lit(Literal::Int(42));
        assert_eq!(print_expr(&expr), "42");
        
        let expr = Expr::Lit(Literal::Float(3.14));
        assert_eq!(print_expr(&expr), "3.14");
        
        let expr = Expr::Lit(Literal::True);
        assert_eq!(print_expr(&expr), "true");
    }
    
    #[test]
    fn test_print_binop() {
        // 1 + 2
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        assert_eq!(print_expr(&expr), "1 + 2");
    }
    
    #[test]
    fn test_print_binop_precedence() {
        // 1 + 2 * 3 should be 1 + (2 * 3) in precedence, but print as 1 + 2 × 3
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Lit(Literal::Int(2))),
                Box::new(Expr::Lit(Literal::Int(3))),
            )),
        );
        let output = print_expr(&expr);
        assert!(output.contains("1 + 2"));
        assert!(output.contains("3"));
    }
    
    #[test]
    fn test_print_lambda() {
        // λ→ ₀ + 1
        let expr = Expr::Lam(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),
            Box::new(Expr::Lit(Literal::Int(1))),
        )));
        let output = print_expr(&expr);
        assert!(output.contains("λ→"));
        assert!(output.contains("+"));
        assert!(output.contains("1"));
    }
    
    #[test]
    fn test_print_lambda_ascii() {
        let expr = Expr::Lam(Box::new(Expr::Idx(0)));
        let mut p = Pretty::new(PrettyConfig::ascii());
        p.print_expr(&expr);
        let output = p.finish();
        assert!(output.contains("\\->"));
    }
    
    #[test]
    fn test_print_let() {
        // let x = 5 in x + 1
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            type_: None,
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(1))),
            )),
        };
        let output = print_expr(&expr);
        assert!(output.contains("let"));
        assert!(output.contains("x"));
        assert!(output.contains("5"));
        assert!(output.contains("in"));
    }
    
    #[test]
    fn test_print_if() {
        // if true then 1 else 2
        let expr = Expr::If {
            cond: Box::new(Expr::Lit(Literal::True)),
            then_: Box::new(Expr::Lit(Literal::Int(1))),
            else_: Box::new(Expr::Lit(Literal::Int(2))),
        };
        let output = print_expr(&expr);
        assert!(output.contains("if"));
        assert!(output.contains("then"));
        assert!(output.contains("else"));
    }
    
    #[test]
    fn test_print_tuple() {
        // ⟨1, 2, 3⟩
        let expr = Expr::Tuple(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
        ]);
        let output = print_expr(&expr);
        assert!(output.contains("1"));
        assert!(output.contains("2"));
        assert!(output.contains("3"));
        assert!(output.contains(","));
    }
    
    #[test]
    fn test_print_array() {
        // [1, 2, 3]
        let expr = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
        ]);
        assert_eq!(print_expr(&expr), "[1, 2, 3]");
    }
    
    #[test]
    fn test_print_application() {
        // f x
        let expr = Expr::App(
            Box::new(Expr::Name("f".into())),
            Box::new(Expr::Name("x".into())),
        );
        assert_eq!(print_expr(&expr), "f x");
    }
    
    #[test]
    fn test_print_type_prim() {
        assert_eq!(print_type(&Type::Prim(PrimType::I64)), "I64");
        assert_eq!(print_type(&Type::Prim(PrimType::F64)), "F64");
        assert_eq!(print_type(&Type::Prim(PrimType::Bool)), "Bool");
    }
    
    #[test]
    fn test_print_type_function() {
        // I64 → I64
        let ty = Type::func(
            Type::Prim(PrimType::I64),
            Type::Prim(PrimType::I64),
        );
        let output = print_type(&ty);
        assert!(output.contains("I64"));
        assert!(output.contains("→") || output.contains("->"));
    }
    
    #[test]
    fn test_print_type_tuple() {
        use crate::types::TupleField;
        
        let ty = Type::Tuple(vec![
            TupleField { label: None, ty: Type::Prim(PrimType::I64) },
            TupleField { label: None, ty: Type::Prim(PrimType::F64) },
        ]);
        let output = print_type(&ty);
        assert!(output.contains("I64"));
        assert!(output.contains("F64"));
    }
    
    #[test]
    fn test_print_module() {
        use crate::decl::{Module, Decl, LetDecl};
        
        let module = Module {
            name: Some("test".into()),
            decls: vec![
                Decl::Let(LetDecl {
                    name: "x".into(),
                    type_: Some(Type::Prim(PrimType::I64)),
                    value: Expr::Lit(Literal::Int(42)),
                }),
            ],
        };
        
        let output = print_module(&module);
        assert!(output.contains("module test"));
        assert!(output.contains("let x"));
        assert!(output.contains("42"));
    }
    
    #[test]
    fn test_roundtrip_simple() {
        // Pretty print should produce readable output
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        
        let output = print_expr(&expr);
        
        // Should be readable
        assert!(!output.is_empty());
        assert!(output.len() < 20); // Should be compact
    }
    
    #[test]
    fn test_complex_expression() {
        // let f = λ→ ₀ + 1 in f 5
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("f".into())),
            type_: None,
            value: Box::new(Expr::Lam(Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(1))),
            )))),
            body: Box::new(Expr::App(
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
        };
        
        let output = print_expr(&expr);
        println!("Complex: {}", output);
        
        assert!(output.contains("let"));
        assert!(output.contains("f"));
        assert!(output.contains("λ→") || output.contains("\\->"));
        assert!(output.contains("in"));
    }
    
    #[test]
    fn test_unicode_vs_ascii() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Lit(Literal::Int(2))),
            Box::new(Expr::Lit(Literal::Int(3))),
        );
        
        // Unicode
        let mut p = Pretty::default();
        p.print_expr(&expr);
        let unicode_output = p.finish();
        assert!(unicode_output.contains("×"));
        
        // ASCII
        let mut p = Pretty::new(PrettyConfig::ascii());
        p.print_expr(&expr);
        let ascii_output = p.finish();
        assert!(ascii_output.contains("*"));
    }
}
