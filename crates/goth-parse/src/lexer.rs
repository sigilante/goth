//! Lexer for Goth
//!
//! Tokenizes Goth source code using logos for efficient lexing.
//! Supports both Unicode glyphs and ASCII fallbacks.

use logos::{Logos, Span};
use std::fmt;

/// Source location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Loc {
    pub start: usize,
    pub end: usize,
}

impl Loc {
    pub fn new(start: usize, end: usize) -> Self {
        Loc { start, end }
    }

    pub fn from_span(span: Span) -> Self {
        Loc { start: span.start, end: span.end }
    }

    pub fn merge(self, other: Loc) -> Loc {
        Loc {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// Token with location
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub value: T,
    pub loc: Loc,
}

impl<T> Spanned<T> {
    pub fn new(value: T, loc: Loc) -> Self {
        Spanned { value, loc }
    }
}

/// Token type
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"#[^\n]*")]
pub enum Token {
    // ============ Keywords ============
    #[token("let")]
    Let,
    #[token("in")]
    In,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("with")]
    With,
    #[token("where")]
    Where,
    #[token("class")]
    Class,
    #[token("impl")]
    Impl,
    #[token("do")]
    Do,
    #[token("end")]
    End,
    #[token("rec")]
    Rec,
    #[token("type")]
    Type,
    #[token("fn")]
    Fn,

    // ============ Function Box ============
    #[token("╭─")]
    #[token("/-")]
    FnStart,
    #[token("│")]
    FnMid,
    #[token("╰─")]
    FnEnd,

    // ============ Delimiters ============
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("⟨")]
    #[token("<|")]
    LAngle,
    #[token("⟩")]
    #[token("|>")]
    RAngle,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(",")]
    Comma,
    #[token(";")]
    Semi,
    #[token(":")]
    Colon,
    #[token("::")]
    DoubleColon,
    #[token(".")]
    Dot,
    #[token("@")]
    At,

    // ============ Arithmetic ============
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("×")]
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("^")]
    Caret,
    #[token("%")]
    Percent,
    #[token("±")]
    PlusMinus,
    #[token("√")]
    Sqrt,
    #[token("⌊")]
    Floor,
    #[token("⌋")]
    FloorClose,
    #[token("⌈")]
    Ceil,
    #[token("⌉")]
    CeilClose,

    // ============ Comparison ============
    #[token("=")]
    Eq,
    #[token("≠")]
    #[token("/=")]
    Neq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("≤")]
    #[token("<=")]
    Leq,
    #[token("≥")]
    #[token(">=")]
    Geq,

    // ============ Logical ============
    #[token("∧")]
    #[token("&&")]
    And,
    #[token("∨")]
    #[token("||")]
    Or,
    #[token("¬")]
    #[token("!")]
    Not,

    // ============ Functional ============
    #[token("↦")]
    #[token("-:")]
    Map,
    #[token("▸")]
    #[token("|>_")]
    Filter,
    #[token("⤇")]
    #[token("=>>")]
    Bind,
    #[token("∘")]
    #[token(".:")]
    Compose,
    #[token("⊗")]
    #[token("*:")]
    ZipWith,
    #[token("⊕")]
    #[token("+:")]
    Concat,

    // ============ Reduction ============
    #[token("Σ")]
    #[token("+/")]
    Sum,
    #[token("Π")]
    #[token("*/")]
    Prod,
    #[token("⍀")]
    #[token("\\/")]
    Scan,

    // ============ Math Functions ============
    #[token("Γ")]
    #[token("gamma")]
    Gamma,
    #[token("ln")]
    Ln,
    #[token("exp")]
    Exp,
    #[token("sin")]
    Sin,
    #[token("cos")]
    Cos,
    #[token("abs")]
    Abs,

    // ============ Arrows ============
    #[token("→")]
    #[token("->")]
    Arrow,
    #[token("←")]
    #[token("<-")]
    BackArrow,
    #[token("⇒")]
    #[token("=>")]
    FatArrow,

    // ============ Lambda ============
    #[token("λ")]
    #[token("\\")]
    Lambda,

    // ============ Spec ============
    #[token("⊢")]
    #[token("|-")]
    Turnstile,
    #[token("⊨")]
    #[token("|=")]
    Models,

    // ============ Quantifiers ============
    #[token("∀")]
    #[token("forall")]
    Forall,
    #[token("∃")]
    #[token("exists")]
    Exists,

    // ============ Equivalence ============
    #[token("≡")]
    #[token("===")]
    Equiv,

    // ============ Effects ============
    #[token("□")]
    Pure,
    #[token("◇")]
    Diamond,

    // ============ Booleans ============
    #[token("⊤")]
    #[token("true")]
    True,
    #[token("⊥")]
    #[token("false")]
    False,

    // ============ Special ============
    #[token("∞")]
    #[token("inf")]
    Infinity,
    #[token("..")]
    DotDot,
    #[token("?")]
    Question,
    #[token("_", priority = 3)]
    Underscore,
    #[token("|")]
    Pipe,

    // ============ Norm ============
    #[token("‖")]
    #[token("||_")]
    Norm,

    // ============ De Bruijn Indices ============
    #[regex(r"₀|₁|₂|₃|₄|₅|₆|₇|₈|₉|_[0-9]+", priority = 3, callback = |lex| parse_index(lex.slice()))]
    Index(u32),

    // ============ Literals ============
    #[regex(r"[0-9]+", priority = 2, callback = |lex| lex.slice().parse::<i128>().ok())]
    Int(i128),

    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(unescape_string(&s[1..s.len()-1]))
    })]
    String(String),

    #[regex(r"'([^'\\]|\\.)'", |lex| {
        let s = lex.slice();
        unescape_char(&s[1..s.len()-1])
    })]
    Char(char),

    // ============ Identifiers ============
    // Identifiers include Greek letters and some APL symbols
    #[regex(r"[a-zA-Z_αβγδεζηθικμνξοπρστυφχψωρι][a-zA-Z0-9_'αβγδεζηθικμνξοπρστυφχψω²³]*", |lex| lex.slice().to_string())]
    Ident(String),

    // ============ Type Variables (Greek) - excludes λ (lambda keyword) ============
    #[regex(r"[αβγδεζηθικμνξοπρστυφχψω]", priority = 4, callback = |lex| lex.slice().to_string())]
    TyVar(String),

    // ============ APL-style single character identifiers ============
    #[regex(r"[⍳⍴⌽⍉⧺·…↑↓]", priority = 5, callback = |lex| lex.slice().to_string())]
    AplIdent(String),

    // ============ Primitive Types ============
    #[token("F64")]
    TyF64,
    #[token("F32")]
    TyF32,
    #[token("I64")]
    TyI64,
    #[token("I32")]
    TyI32,
    #[token("U64")]
    TyU64,
    #[token("U8")]
    TyU8,
    #[token("Bool")]
    TyBool,
    #[token("Char")]
    TyChar,
    #[token("Byte")]
    TyByte,
    #[token("ℕ")]
    #[token("Nat")]
    TyNat,
    #[token("ℤ")]
    #[token("Int")]
    TyInt,
    #[token("Unit")]
    TyUnit,
}

fn parse_index(s: &str) -> Option<u32> {
    if s.starts_with('_') {
        s[1..].parse().ok()
    } else {
        // Unicode subscript
        let c = s.chars().next()?;
        let digit = match c {
            '₀' => 0, '₁' => 1, '₂' => 2, '₃' => 3, '₄' => 4,
            '₅' => 5, '₆' => 6, '₇' => 7, '₈' => 8, '₉' => 9,
            _ => return None,
        };
        Some(digit)
    }
}

fn unescape_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('0') => result.push('\0'),
                Some(c) => { result.push('\\'); result.push(c); }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn unescape_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    match chars.next()? {
        '\\' => match chars.next()? {
            'n' => Some('\n'),
            't' => Some('\t'),
            'r' => Some('\r'),
            '\\' => Some('\\'),
            '\'' => Some('\''),
            '0' => Some('\0'),
            c => Some(c),
        },
        c => Some(c),
    }
}

impl Token {
    /// Check if token can start an expression
    pub fn can_start_expr(&self) -> bool {
        matches!(self,
            Token::Int(_) | Token::Float(_) | Token::String(_) | Token::Char(_) |
            Token::True | Token::False |
            Token::Ident(_) | Token::TyVar(_) | Token::AplIdent(_) |
            Token::Lambda | Token::LParen | Token::LBracket | Token::LAngle |
            Token::If | Token::Match | Token::Let | Token::Do |
            Token::Index(_) |
            Token::Minus | Token::Not | Token::Sum | Token::Prod | Token::Scan |
            Token::Norm | Token::Underscore
        )
    }

    /// Check if token is a binary operator
    pub fn is_binop(&self) -> bool {
        self.precedence().is_some()
    }

    /// Get operator precedence (higher = tighter binding)
    pub fn precedence(&self) -> Option<u8> {
        match self {
            Token::Or => Some(3),
            Token::And => Some(4),
            Token::Eq | Token::Neq | Token::Lt | Token::Gt | Token::Leq | Token::Geq => Some(5),
            Token::Concat => Some(6),
            Token::Plus | Token::Minus | Token::PlusMinus => Some(7),
            Token::Star | Token::Slash | Token::Percent | Token::ZipWith => Some(8),
            Token::Caret => Some(9),
            Token::Compose => Some(10),
            Token::Map | Token::Filter | Token::Bind => Some(2),
            _ => None,
        }
    }

    /// Check if operator is right-associative
    pub fn is_right_assoc(&self) -> bool {
        matches!(self, Token::Caret | Token::Arrow | Token::Map | Token::Filter | Token::Bind | Token::Compose)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Int(n) => write!(f, "{}", n),
            Token::Float(x) => write!(f, "{}", x),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Char(c) => write!(f, "'{}'", c),
            Token::Ident(s) | Token::TyVar(s) => write!(f, "{}", s),
            Token::Index(i) => write!(f, "_{}", i),
            Token::True => write!(f, "⊤"),
            Token::False => write!(f, "⊥"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "×"),
            Token::Slash => write!(f, "/"),
            Token::Arrow => write!(f, "→"),
            Token::Lambda => write!(f, "λ"),
            Token::Map => write!(f, "↦"),
            Token::Filter => write!(f, "▸"),
            Token::Bind => write!(f, "⤇"),
            Token::Sum => write!(f, "Σ"),
            Token::Prod => write!(f, "Π"),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Lexer with peek support
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, Token>,
    peeked: Option<Option<Spanned<Token>>>,
    source: &'a str,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            inner: Token::lexer(source),
            peeked: None,
            source,
        }
    }

    /// Peek at the next token
    pub fn peek(&mut self) -> Option<&Token> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_inner());
        }
        self.peeked.as_ref().unwrap().as_ref().map(|s| &s.value)
    }

    /// Peek at next token with location
    pub fn peek_spanned(&mut self) -> Option<&Spanned<Token>> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_inner());
        }
        self.peeked.as_ref().unwrap().as_ref()
    }

    /// Get the next token
    pub fn next(&mut self) -> Option<Token> {
        self.next_spanned().map(|s| s.value)
    }

    /// Get next token with location
    pub fn next_spanned(&mut self) -> Option<Spanned<Token>> {
        if let Some(peeked) = self.peeked.take() {
            peeked
        } else {
            self.next_inner()
        }
    }

    fn next_inner(&mut self) -> Option<Spanned<Token>> {
        loop {
            match self.inner.next() {
                Some(Ok(token)) => {
                    return Some(Spanned::new(token, Loc::from_span(self.inner.span())));
                }
                Some(Err(_)) => continue, // Skip invalid tokens
                None => return None,
            }
        }
    }

    /// Check if there are more tokens
    pub fn has_more(&mut self) -> bool {
        self.peek().is_some()
    }

    /// Expect a specific token
    pub fn expect(&mut self, expected: &Token) -> Result<Spanned<Token>, String> {
        match self.next_spanned() {
            Some(tok) if &tok.value == expected => Ok(tok),
            Some(tok) => Err(format!("Expected {:?}, got {:?}", expected, tok.value)),
            None => Err(format!("Expected {:?}, got EOF", expected)),
        }
    }

    /// Get source slice for a location
    pub fn slice(&self, loc: Loc) -> &'a str {
        &self.source[loc.start..loc.end]
    }

    /// Current position
    pub fn pos(&self) -> usize {
        self.inner.span().start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lex = Lexer::new("let x = 5 in x + 3");
        assert_eq!(lex.next(), Some(Token::Let));
        assert_eq!(lex.next(), Some(Token::Ident("x".into())));
        assert_eq!(lex.next(), Some(Token::Eq));
        assert_eq!(lex.next(), Some(Token::Int(5)));
        assert_eq!(lex.next(), Some(Token::In));
        assert_eq!(lex.next(), Some(Token::Ident("x".into())));
        assert_eq!(lex.next(), Some(Token::Plus));
        assert_eq!(lex.next(), Some(Token::Int(3)));
        assert_eq!(lex.next(), None);
    }

    #[test]
    fn test_unicode_operators() {
        let mut lex = Lexer::new("λ→ ₀ × ₁");
        assert_eq!(lex.next(), Some(Token::Lambda));
        assert_eq!(lex.next(), Some(Token::Arrow));
        assert_eq!(lex.next(), Some(Token::Index(0)));
        assert_eq!(lex.next(), Some(Token::Star));
        assert_eq!(lex.next(), Some(Token::Index(1)));
    }

    #[test]
    fn test_ascii_fallbacks() {
        let mut lex = Lexer::new("\\-> _0 * _1 -: +/");
        assert_eq!(lex.next(), Some(Token::Lambda));
        assert_eq!(lex.next(), Some(Token::Arrow));
        assert_eq!(lex.next(), Some(Token::Index(0)));
        assert_eq!(lex.next(), Some(Token::Star));
        assert_eq!(lex.next(), Some(Token::Index(1)));
        assert_eq!(lex.next(), Some(Token::Map));
        assert_eq!(lex.next(), Some(Token::Sum));
    }

    #[test]
    fn test_literals() {
        let mut lex = Lexer::new("42 3.14 \"hello\" 'c' ⊤ ⊥");
        assert_eq!(lex.next(), Some(Token::Int(42)));
        assert_eq!(lex.next(), Some(Token::Float(3.14)));
        assert_eq!(lex.next(), Some(Token::String("hello".into())));
        assert_eq!(lex.next(), Some(Token::Char('c')));
        assert_eq!(lex.next(), Some(Token::True));
        assert_eq!(lex.next(), Some(Token::False));
    }

    #[test]
    fn test_string_escapes() {
        let mut lex = Lexer::new(r#""hello\nworld""#);
        assert_eq!(lex.next(), Some(Token::String("hello\nworld".into())));
    }

    #[test]
    fn test_comments() {
        let mut lex = Lexer::new("x # comment\n+ y");
        assert_eq!(lex.next(), Some(Token::Ident("x".into())));
        assert_eq!(lex.next(), Some(Token::Plus));
        assert_eq!(lex.next(), Some(Token::Ident("y".into())));
    }

    #[test]
    fn test_function_box() {
        let mut lex = Lexer::new("╭─ f : T\n╰─ ₀");
        assert_eq!(lex.next(), Some(Token::FnStart));
        assert_eq!(lex.next(), Some(Token::Ident("f".into())));
        assert_eq!(lex.next(), Some(Token::Colon));
        assert_eq!(lex.next(), Some(Token::Ident("T".into())));
        assert_eq!(lex.next(), Some(Token::FnEnd));
        assert_eq!(lex.next(), Some(Token::Index(0)));
    }

    #[test]
    fn test_de_bruijn_indices() {
        let mut lex = Lexer::new("_0 _1 _10 _99");
        assert_eq!(lex.next(), Some(Token::Index(0)));
        assert_eq!(lex.next(), Some(Token::Index(1)));
        assert_eq!(lex.next(), Some(Token::Index(10)));
        assert_eq!(lex.next(), Some(Token::Index(99)));
    }

    #[test]
    fn test_prim_types() {
        let mut lex = Lexer::new("F64 I32 Bool ℕ");
        assert_eq!(lex.next(), Some(Token::TyF64));
        assert_eq!(lex.next(), Some(Token::TyI32));
        assert_eq!(lex.next(), Some(Token::TyBool));
        assert_eq!(lex.next(), Some(Token::TyNat));
    }

    #[test]
    fn test_greek_type_vars() {
        let mut lex = Lexer::new("α β γ");
        assert_eq!(lex.next(), Some(Token::TyVar("α".into())));
        assert_eq!(lex.next(), Some(Token::TyVar("β".into())));
        assert_eq!(lex.next(), Some(Token::TyVar("γ".into())));
    }

    #[test]
    fn test_greek_in_value_context() {
        // Greek letters are lexed as TyVar but parser accepts them as value identifiers
        let mut lex = Lexer::new("let μ = 5");
        assert_eq!(lex.next(), Some(Token::Let));
        assert_eq!(lex.next(), Some(Token::TyVar("μ".into())));
        assert_eq!(lex.next(), Some(Token::Eq));
        assert_eq!(lex.next(), Some(Token::Int(5)));
    }

    #[test]
    fn test_greek_with_suffix() {
        // Greek + suffix is Ident, not TyVar
        let mut lex = Lexer::new("μ_value σ²");
        assert_eq!(lex.next(), Some(Token::Ident("μ_value".into())));
        assert_eq!(lex.next(), Some(Token::Ident("σ²".into())));
    }

    #[test]
    fn test_back_arrow() {
        let mut lex = Lexer::new("let x ← 5");
        assert_eq!(lex.next(), Some(Token::Let));
        assert_eq!(lex.next(), Some(Token::Ident("x".into())));
        assert_eq!(lex.next(), Some(Token::BackArrow));
        assert_eq!(lex.next(), Some(Token::Int(5)));
    }

    #[test]
    fn test_back_arrow_ascii() {
        let mut lex = Lexer::new("<-");
        assert_eq!(lex.next(), Some(Token::BackArrow));
    }
}
