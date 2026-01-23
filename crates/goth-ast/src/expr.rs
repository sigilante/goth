//! Expressions in Goth
//!
//! The core of the AST. Uses de Bruijn indices for variable binding.

use serde::{Deserialize, Serialize};
use crate::literal::Literal;
use crate::op::{BinOp, UnaryOp};
use crate::types::Type;
use crate::pattern::Pattern;

/// Expression (the core AST node)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    // ============ Atoms ============
    
    /// De Bruijn index (reference to lambda-bound variable)
    /// ₀ is innermost, ₁ is next out, etc.
    Idx(u32),

    /// Named reference (for top-level definitions, not lambda-bound)
    Name(Box<str>),

    /// Literal value
    Lit(Literal),

    /// Primitive operation marker (implementation provided by runtime)
    Prim(Box<str>),

    // ============ Compound ============
    
    /// Function application: f x
    App(Box<Expr>, Box<Expr>),

    /// Lambda abstraction: λ→ body
    /// The parameter is implicit (accessed as ₀ in body)
    Lam(Box<Expr>),

    /// Lambda with explicit parameter count (for multi-arg)
    /// λ⟨a b c⟩→ body introduces 3 bindings
    LamN(u32, Box<Expr>),

    /// Let binding: let pattern [: type] ← value in body
    Let {
        pattern: Pattern,
        /// Optional type annotation
        type_: Option<Type>,
        value: Box<Expr>,
        body: Box<Expr>,
    },

    /// Recursive let: let rec bindings in body
    LetRec {
        bindings: Vec<(Pattern, Expr)>,
        body: Box<Expr>,
    },

    /// Match expression
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// If-then-else
    If {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
    },

    // ============ Operators ============
    
    /// Binary operation
    BinOp(BinOp, Box<Expr>, Box<Expr>),

    /// Unary operation
    UnaryOp(UnaryOp, Box<Expr>),

    /// Norm: ‖x‖
    Norm(Box<Expr>),

    // ============ Data Construction ============
    
    /// Tuple: ⟨e₀, e₁, ...⟩
    Tuple(Vec<Expr>),

    /// Labeled tuple (record): ⟨x: e₀, y: e₁, ...⟩
    Record(Vec<(Box<str>, Expr)>),

    /// Array: [e₀, e₁, ...]
    Array(Vec<Expr>),

    /// Array fill: [shape]⊢value
    ArrayFill {
        shape: Vec<Expr>,  // Each dimension
        value: Box<Expr>,  // Fill value
    },

    /// Variant constructor: Constructor or Constructor value
    Variant {
        constructor: Box<str>,
        payload: Option<Box<Expr>>,
    },

    // ============ Access ============
    
    /// Field access: e.field or e.0
    Field(Box<Expr>, FieldAccess),

    /// Array indexing: e[i, j, ...]
    Index(Box<Expr>, Vec<Expr>),

    /// Array slice: e[start:end]
    Slice {
        array: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },

    // ============ Type Annotations ============
    
    /// Type annotation: e ⊢ T
    Annot(Box<Expr>, Type),

    /// Cast: e ⊢as T
    Cast {
        expr: Box<Expr>,
        target: Type,
        kind: CastKind,
    },

    /// Record update: e⊢{field: value, ...}
    Update {
        base: Box<Expr>,
        fields: Vec<(Box<str>, Expr)>,
    },

    // ============ Do Notation ============
    
    /// Do block: do e ops end
    Do {
        init: Box<Expr>,
        ops: Vec<DoOp>,
    },

    // ============ Special ============
    
    /// Disabled expression (preserved in AST but not executed)
    Disabled(Box<Expr>),

    /// Hole (for synthesis/inference): _
    Hole,

    /// Quote (for macros): ⟨expr⟩
    Quote(Box<Expr>),

    /// Unquote (splice in macro): ‹expr›
    Unquote(Box<Expr>),
}

/// Field access (by name or index)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldAccess {
    Named(Box<str>),
    Index(u32),
}

/// Match arm: pattern → body
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

/// Cast kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CastKind {
    /// Compile-time cast (zero-cost view)
    Static,
    /// Runtime cast (returns Option)
    Try,
    /// Runtime cast (panics on failure)
    Force,
}

/// Operation in do-notation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DoOp {
    /// Map: ↦ f
    Map(Expr),
    /// Filter: ▸ pred
    Filter(Expr),
    /// Bind: ⤇ f
    Bind(Expr),
    /// Generic binary op
    Op(BinOp, Expr),
    /// Let binding within do
    Let(Pattern, Expr),
}

// ============ Constructors ============

impl Expr {
    /// De Bruijn index
    pub fn idx(i: u32) -> Self {
        Expr::Idx(i)
    }

    /// Named reference
    pub fn name(n: impl Into<Box<str>>) -> Self {
        Expr::Name(n.into())
    }

    /// Literal
    pub fn lit(l: impl Into<Literal>) -> Self {
        Expr::Lit(l.into())
    }

    /// Integer literal
    pub fn int(n: i128) -> Self {
        Expr::Lit(Literal::Int(n))
    }

    /// Float literal
    pub fn float(f: f64) -> Self {
        Expr::Lit(Literal::Float(f))
    }

    /// Boolean literal
    pub fn bool(b: bool) -> Self {
        Expr::Lit(Literal::bool(b))
    }

    /// Lambda
    pub fn lam(body: Expr) -> Self {
        Expr::Lam(Box::new(body))
    }

    /// Multi-argument lambda
    pub fn lam_n(n: u32, body: Expr) -> Self {
        if n == 1 {
            Expr::lam(body)
        } else {
            Expr::LamN(n, Box::new(body))
        }
    }

    /// Application
    pub fn app(func: Expr, arg: Expr) -> Self {
        Expr::App(Box::new(func), Box::new(arg))
    }

    /// Multi-argument application
    pub fn app_n(func: Expr, args: impl IntoIterator<Item = Expr>) -> Self {
        args.into_iter().fold(func, |f, a| Expr::app(f, a))
    }

    /// Binary operation
    pub fn binop(op: BinOp, l: Expr, r: Expr) -> Self {
        Expr::BinOp(op, Box::new(l), Box::new(r))
    }

    /// Addition
    pub fn add(l: Expr, r: Expr) -> Self {
        Expr::binop(BinOp::Add, l, r)
    }

    /// Subtraction
    pub fn sub(l: Expr, r: Expr) -> Self {
        Expr::binop(BinOp::Sub, l, r)
    }

    /// Multiplication
    pub fn mul(l: Expr, r: Expr) -> Self {
        Expr::binop(BinOp::Mul, l, r)
    }

    /// Division
    pub fn div(l: Expr, r: Expr) -> Self {
        Expr::binop(BinOp::Div, l, r)
    }

    /// Map
    pub fn map(arr: Expr, f: Expr) -> Self {
        Expr::binop(BinOp::Map, arr, f)
    }

    /// Filter
    pub fn filter(arr: Expr, pred: Expr) -> Self {
        Expr::binop(BinOp::Filter, arr, pred)
    }

    /// Zip-with (tensor product)
    pub fn zip_with(a: Expr, b: Expr) -> Self {
        Expr::binop(BinOp::ZipWith, a, b)
    }

    /// Sum
    pub fn sum(e: Expr) -> Self {
        Expr::UnaryOp(UnaryOp::Sum, Box::new(e))
    }

    /// Let binding
    pub fn let_(pat: Pattern, value: Expr, body: Expr) -> Self {
        Expr::Let {
            pattern: pat,
            type_: None,
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    /// Let binding with type annotation
    pub fn let_typed(pat: Pattern, ty: Type, value: Expr, body: Expr) -> Self {
        Expr::Let {
            pattern: pat,
            type_: Some(ty),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    /// Simple let with variable pattern
    pub fn let_var(name: impl Into<Box<str>>, value: Expr, body: Expr) -> Self {
        Expr::let_(Pattern::var(name), value, body)
    }

    /// Match
    pub fn match_(scrutinee: Expr, arms: Vec<MatchArm>) -> Self {
        Expr::Match {
            scrutinee: Box::new(scrutinee),
            arms,
        }
    }

    /// If-then-else
    pub fn if_(cond: Expr, then_: Expr, else_: Expr) -> Self {
        Expr::If {
            cond: Box::new(cond),
            then_: Box::new(then_),
            else_: Box::new(else_),
        }
    }

    /// Tuple
    pub fn tuple(exprs: Vec<Expr>) -> Self {
        Expr::Tuple(exprs)
    }

    /// Array
    pub fn array(exprs: Vec<Expr>) -> Self {
        Expr::Array(exprs)
    }

    /// Field access by name
    pub fn field(e: Expr, name: impl Into<Box<str>>) -> Self {
        Expr::Field(Box::new(e), FieldAccess::Named(name.into()))
    }

    /// Field access by index
    pub fn field_idx(e: Expr, idx: u32) -> Self {
        Expr::Field(Box::new(e), FieldAccess::Index(idx))
    }

    /// Array indexing
    pub fn index(arr: Expr, indices: Vec<Expr>) -> Self {
        Expr::Index(Box::new(arr), indices)
    }

    /// Type annotation
    pub fn annot(e: Expr, ty: Type) -> Self {
        Expr::Annot(Box::new(e), ty)
    }

    /// Disabled expression
    pub fn disabled(e: Expr) -> Self {
        Expr::Disabled(Box::new(e))
    }

    /// Primitive
    pub fn prim(name: impl Into<Box<str>>) -> Self {
        Expr::Prim(name.into())
    }

    /// Variant constructor
    pub fn variant(name: impl Into<Box<str>>, payload: Option<Expr>) -> Self {
        Expr::Variant {
            constructor: name.into(),
            payload: payload.map(Box::new),
        }
    }
}

impl MatchArm {
    pub fn new(pattern: Pattern, body: Expr) -> Self {
        MatchArm { pattern, guard: None, body }
    }

    pub fn with_guard(pattern: Pattern, guard: Expr, body: Expr) -> Self {
        MatchArm { pattern, guard: Some(guard), body }
    }
}

// ============ De Bruijn Utilities ============

impl Expr {
    /// Shift all de Bruijn indices >= cutoff by delta
    pub fn shift(&self, cutoff: u32, delta: i32) -> Self {
        match self {
            Expr::Idx(i) => {
                if *i >= cutoff {
                    Expr::Idx((*i as i32 + delta) as u32)
                } else {
                    Expr::Idx(*i)
                }
            }
            Expr::Lam(body) => {
                Expr::Lam(Box::new(body.shift(cutoff + 1, delta)))
            }
            Expr::LamN(n, body) => {
                Expr::LamN(*n, Box::new(body.shift(cutoff + n, delta)))
            }
            Expr::App(f, a) => {
                Expr::App(
                    Box::new(f.shift(cutoff, delta)),
                    Box::new(a.shift(cutoff, delta)),
                )
            }
            Expr::Let { pattern, type_, value, body } => {
                let bindings = pattern.binding_count() as u32;
                Expr::Let {
                    pattern: pattern.clone(),
                    type_: type_.clone(),
                    value: Box::new(value.shift(cutoff, delta)),
                    body: Box::new(body.shift(cutoff + bindings, delta)),
                }
            }
            Expr::BinOp(op, l, r) => {
                Expr::BinOp(
                    op.clone(),
                    Box::new(l.shift(cutoff, delta)),
                    Box::new(r.shift(cutoff, delta)),
                )
            }
            Expr::UnaryOp(op, e) => {
                Expr::UnaryOp(*op, Box::new(e.shift(cutoff, delta)))
            }
            Expr::If { cond, then_, else_ } => {
                Expr::If {
                    cond: Box::new(cond.shift(cutoff, delta)),
                    then_: Box::new(then_.shift(cutoff, delta)),
                    else_: Box::new(else_.shift(cutoff, delta)),
                }
            }
            Expr::Match { scrutinee, arms } => {
                Expr::Match {
                    scrutinee: Box::new(scrutinee.shift(cutoff, delta)),
                    arms: arms.iter().map(|arm| {
                        let bindings = arm.pattern.binding_count() as u32;
                        MatchArm {
                            pattern: arm.pattern.clone(),
                            guard: arm.guard.as_ref().map(|g| g.shift(cutoff + bindings, delta)),
                            body: arm.body.shift(cutoff + bindings, delta),
                        }
                    }).collect(),
                }
            }
            Expr::Tuple(es) => {
                Expr::Tuple(es.iter().map(|e| e.shift(cutoff, delta)).collect())
            }
            Expr::Array(es) => {
                Expr::Array(es.iter().map(|e| e.shift(cutoff, delta)).collect())
            }
            // Atoms that don't contain indices
            Expr::Name(_) | Expr::Lit(_) | Expr::Prim(_) | Expr::Hole => self.clone(),
            // TODO: Handle remaining cases
            _ => self.clone(),
        }
    }

    /// Substitute expression s for index i
    pub fn subst(&self, i: u32, s: &Expr) -> Self {
        match self {
            Expr::Idx(j) => {
                if *j == i {
                    s.clone()
                } else if *j > i {
                    Expr::Idx(j - 1)
                } else {
                    Expr::Idx(*j)
                }
            }
            Expr::Lam(body) => {
                Expr::Lam(Box::new(body.subst(i + 1, &s.shift(0, 1))))
            }
            Expr::App(f, a) => {
                Expr::App(
                    Box::new(f.subst(i, s)),
                    Box::new(a.subst(i, s)),
                )
            }
            Expr::BinOp(op, l, r) => {
                Expr::BinOp(op.clone(), Box::new(l.subst(i, s)), Box::new(r.subst(i, s)))
            }
            _ => self.clone(), // TODO: complete
        }
    }
}

// ============ Display ============

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Idx(i) => {
                let subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
                if *i < 10 {
                    write!(f, "{}", subscripts[*i as usize])
                } else {
                    write!(f, "_{}", i)
                }
            }
            Expr::Name(n) => write!(f, "{}", n),
            Expr::Lit(l) => match l {
                Literal::Int(n) => write!(f, "{}", n),
                Literal::Float(x) => write!(f, "{}", x),
                Literal::Char(c) => write!(f, "'{}'", c),
                Literal::String(s) => write!(f, "\"{}\"", s),
                Literal::True => write!(f, "⊤"),
                Literal::False => write!(f, "⊥"),
                Literal::Unit => write!(f, "⟨⟩"),
            },
            Expr::Prim(name) => write!(f, "⊥{}", name),
            Expr::App(func, arg) => write!(f, "({} {})", func, arg),
            Expr::Lam(body) => write!(f, "λ→ {}", body),
            Expr::LamN(n, body) => write!(f, "λ{}→ {}", n, body),
            Expr::Let { pattern, type_, value, body } => {
                if let Some(ty) = type_ {
                    write!(f, "let {} : {} ← {} in {}", pattern, ty, value, body)
                } else {
                    write!(f, "let {} ← {} in {}", pattern, value, body)
                }
            }
            Expr::BinOp(op, l, r) => write!(f, "({} {} {})", l, op.glyph(), r),
            Expr::UnaryOp(op, e) => write!(f, "{}{}", op.glyph(), e),
            Expr::If { cond, then_, else_ } => {
                write!(f, "if {} then {} else {}", cond, then_, else_)
            }
            Expr::Match { scrutinee, arms } => {
                write!(f, "match {} {{ ", scrutinee)?;
                for arm in arms {
                    write!(f, "{} → {}; ", arm.pattern, arm.body)?;
                }
                write!(f, "}}")
            }
            Expr::Tuple(es) => {
                write!(f, "⟨")?;
                for (i, e) in es.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "⟩")
            }
            Expr::Array(es) => {
                write!(f, "[")?;
                for (i, e) in es.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Expr::Hole => write!(f, "_"),
            Expr::Disabled(e) => write!(f, "#- {} -#", e),
            _ => write!(f, "..."), // TODO: complete
        }
    }
}
