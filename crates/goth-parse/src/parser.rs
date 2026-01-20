//! Parser for Goth
//!
//! Pratt parser for expressions, recursive descent for types and declarations.

use crate::lexer::{Token, Lexer};
use goth_ast::expr::{Expr, MatchArm, FieldAccess, DoOp};
use goth_ast::types::{Type, PrimType, TupleField, TypeParam, TypeParamKind};
use goth_ast::pattern::Pattern;
use goth_ast::literal::Literal;
use goth_ast::op::{BinOp, UnaryOp};
use goth_ast::shape::{Shape, Dim};
use goth_ast::decl::{Module, Decl, FnDecl, TypeDecl};
use goth_ast::effect::{Effect, Effects};
use thiserror::Error;

/// Parse error
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("Unexpected token: {found:?}, expected {expected}")]
    Unexpected { found: Option<Token>, expected: String },

    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Invalid syntax: {0}")]
    InvalidSyntax(String),
}

pub type ParseResult<T> = Result<T, ParseError>;

/// Parser
pub struct Parser<'a> {
    lexer: Lexer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        Parser { lexer: Lexer::new(source) }
    }

    // ============ Utilities ============

    fn peek(&mut self) -> Option<&Token> {
        self.lexer.peek()
    }

    fn next(&mut self) -> Option<Token> {
        self.lexer.next()
    }

    fn expect(&mut self, expected: Token) -> ParseResult<()> {
        match self.next() {
            Some(ref t) if t == &expected => Ok(()),
            other => Err(ParseError::Unexpected {
                found: other,
                expected: format!("{:?}", expected),
            }),
        }
    }

    fn expect_ident(&mut self) -> ParseResult<String> {
        match self.next() {
            Some(Token::Ident(s)) | Some(Token::TyVar(s)) | Some(Token::AplIdent(s)) => Ok(s),
            other => Err(ParseError::Unexpected {
                found: other,
                expected: "identifier".into(),
            }),
        }
    }

    fn at(&mut self, token: &Token) -> bool {
        self.peek() == Some(token)
    }

    fn eat(&mut self, token: &Token) -> bool {
        if self.at(token) {
            self.next();
            true
        } else {
            false
        }
    }

    // ============ Expression Parsing (Pratt) ============

    /// Parse an expression
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_expr_bp(0)
    }

    /// Parse expression with minimum binding power
    fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Try infix operators first
            if let Some(t) = self.peek() {
                if t.is_binop() {
                    let op = t.clone();
                    let (l_bp, r_bp) = self.infix_binding_power(&op);
                    if l_bp >= min_bp {
                        self.next(); // consume operator
                        let rhs = self.parse_expr_bp(r_bp)?;
                        lhs = self.make_binop(op, lhs, rhs);
                        continue; // Check for more operators/applications
                    }
                }
            }

            // Try function application (juxtaposition)
            if self.peek().map(|t| t.can_start_expr()).unwrap_or(false) {
                if let Some(t) = self.peek() {
                    if t.is_binop() {
                        break; // Let infix handle it
                    }
                    // Don't treat postfix operators as application arguments
                    // Let parse_postfix handle them
                    if matches!(t, Token::Sum | Token::Prod | Token::Scan) {
                        break;
                    }
                }
                let arg = self.parse_atom()?;
                lhs = Expr::App(Box::new(lhs), Box::new(arg));
                continue; // Check for more operators/applications
            }

            // No more operators or applications
            break;
        }

        // Handle postfix operators AFTER infix and application
        // Only at top level (min_bp == 0) to ensure they bind loosest
        // This ensures [1,2,3] × [4,5,6] Σ parses as ([1,2,3] × [4,5,6]) Σ
        if min_bp == 0 {
            lhs = self.parse_postfix(lhs)?;
        }

        Ok(lhs)
    }

    /// Get binding power for infix operators
    fn infix_binding_power(&self, op: &Token) -> (u8, u8) {
        let prec = op.precedence().unwrap_or(0);
        if op.is_right_assoc() {
            (prec, prec)
        } else {
            (prec, prec + 1)
        }
    }

    /// Parse prefix expression (atoms and unary operators)
    fn parse_prefix(&mut self) -> ParseResult<Expr> {
        match self.peek().cloned() {
            // Unary operators
            Some(Token::Minus) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Neg, Box::new(operand)))
            }
            Some(Token::Not) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Not, Box::new(operand)))
            }
            Some(Token::Sum) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Sum, Box::new(operand)))
            }
            Some(Token::Prod) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Prod, Box::new(operand)))
            }
            Some(Token::Scan) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Scan, Box::new(operand)))
            }
            Some(Token::Norm) => {
                self.next();
                let operand = self.parse_atom()?;
                self.expect(Token::Norm)?;
                Ok(Expr::Norm(Box::new(operand)))
            }
            Some(Token::Sqrt) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Sqrt, Box::new(operand)))
            }
            Some(Token::Floor) => {
                self.next();
                let operand = self.parse_expr()?;
                self.expect(Token::FloorClose)?;
                Ok(Expr::UnaryOp(UnaryOp::Floor, Box::new(operand)))
            }
            Some(Token::Ceil) => {
                self.next();
                let operand = self.parse_expr()?;
                self.expect(Token::CeilClose)?;
                Ok(Expr::UnaryOp(UnaryOp::Ceil, Box::new(operand)))
            }
            Some(Token::Gamma) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Gamma, Box::new(operand)))
            }
            Some(Token::Ln) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Ln, Box::new(operand)))
            }
            Some(Token::Exp) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Exp, Box::new(operand)))
            }
            Some(Token::Sin) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Sin, Box::new(operand)))
            }
            Some(Token::Cos) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Cos, Box::new(operand)))
            }
            Some(Token::Abs) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Abs, Box::new(operand)))
            }
            Some(Token::Tan) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Tan, Box::new(operand)))
            }
            Some(Token::Asin) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Asin, Box::new(operand)))
            }
            Some(Token::Acos) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Acos, Box::new(operand)))
            }
            Some(Token::Atan) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Atan, Box::new(operand)))
            }
            Some(Token::Sinh) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Sinh, Box::new(operand)))
            }
            Some(Token::Cosh) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Cosh, Box::new(operand)))
            }
            Some(Token::Tanh) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Tanh, Box::new(operand)))
            }
            Some(Token::Log10) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Log10, Box::new(operand)))
            }
            Some(Token::Log2) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Log2, Box::new(operand)))
            }
            Some(Token::Round) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Round, Box::new(operand)))
            }
            Some(Token::Sign) => {
                self.next();
                let operand = self.parse_prefix()?;
                Ok(Expr::UnaryOp(UnaryOp::Sign, Box::new(operand)))
            }
            _ => self.parse_atom(),
        }
    }

    /// Parse atomic expression
    fn parse_atom(&mut self) -> ParseResult<Expr> {
        let expr = match self.peek().cloned() {
            // Literals
            Some(Token::Int(n)) => { self.next(); Expr::Lit(Literal::Int(n)) }
            Some(Token::Float(f)) => { self.next(); Expr::Lit(Literal::Float(f)) }
            Some(Token::String(s)) => { self.next(); Expr::Lit(Literal::String(s.into())) }
            Some(Token::Char(c)) => { self.next(); Expr::Lit(Literal::Char(c)) }
            Some(Token::True) => { self.next(); Expr::Lit(Literal::True) }
            Some(Token::False) => { self.next(); Expr::Lit(Literal::False) }
            Some(Token::Pi) => { self.next(); Expr::Lit(Literal::Float(std::f64::consts::PI)) }
            Some(Token::Euler) => { self.next(); Expr::Lit(Literal::Float(std::f64::consts::E)) }

            // De Bruijn index
            Some(Token::Index(i)) => { self.next(); Expr::Idx(i) }

            // Identifier (including Greek letters and APL symbols used as variable names)
            Some(Token::Ident(name)) | Some(Token::TyVar(name)) | Some(Token::AplIdent(name)) => { self.next(); Expr::Name(name.into()) }

            // Lambda
            Some(Token::Lambda) => self.parse_lambda()?,

            // Parenthesized or tuple
            Some(Token::LParen) => self.parse_paren_or_tuple()?,

            // Array
            Some(Token::LBracket) => self.parse_array()?,

            // Angle brackets (tuple literal)
            Some(Token::LAngle) => self.parse_angle_tuple()?,

            // Let
            Some(Token::Let) => self.parse_let()?,

            // If
            Some(Token::If) => self.parse_if()?,

            // Match
            Some(Token::Match) => self.parse_match()?,

            // Do
            Some(Token::Do) => self.parse_do()?,

            // Wildcard (in expression position, becomes a hole)
            Some(Token::Underscore) => { self.next(); Expr::Hole }

            other => return Err(ParseError::Unexpected {
                found: other,
                expected: "expression".into(),
            }),
        };

        // Handle postfix access (field, indexing) - binds tightest
        self.parse_postfix_access(expr)
    }

    /// Parse postfix access operations (field access, indexing)
    /// These bind tighter than reduction operators
    fn parse_postfix_access(&mut self, mut expr: Expr) -> ParseResult<Expr> {
        loop {
            match self.peek() {
                Some(Token::Dot) => {
                    self.next();
                    match self.next() {
                        Some(Token::Ident(name)) | Some(Token::TyVar(name)) | Some(Token::AplIdent(name)) => {
                            expr = Expr::Field(Box::new(expr), FieldAccess::Named(name.into()));
                        }
                        Some(Token::Int(i)) => {
                            expr = Expr::Field(Box::new(expr), FieldAccess::Index(i as u32));
                        }
                        other => return Err(ParseError::Unexpected {
                            found: other,
                            expected: "field name or index".into(),
                        }),
                    }
                }
                Some(Token::LBracket) => {
                    // Only allow indexing on expressions that could be arrays:
                    // names, field access, other indices, parenthesized, arrays themselves
                    // Don't allow indexing on literals (e.g., 2[1,2] makes no sense)
                    let can_index = matches!(
                        &expr,
                        Expr::Name(_) | Expr::Field(_, _) | Expr::Index(_, _) |
                        Expr::App(_, _) | Expr::Array(_) | Expr::Tuple(_) |
                        Expr::If { .. } | Expr::Let { .. } | Expr::LetRec { .. } |
                        Expr::Match { .. } | Expr::Do { .. } | Expr::BinOp(_, _, _)
                    );
                    if !can_index {
                        break; // Treat [..] as a separate array, not indexing
                    }
                    self.next();
                    let mut indices = vec![self.parse_expr()?];
                    while self.eat(&Token::Comma) {
                        indices.push(self.parse_expr()?);
                    }
                    self.expect(Token::RBracket)?;
                    expr = Expr::Index(Box::new(expr), indices);
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Parse postfix operations (field access, indexing, reduction operators)
    fn parse_postfix(&mut self, mut expr: Expr) -> ParseResult<Expr> {
        // Postfix reduction operators - bind loosest
        loop {
            match self.peek() {
                Some(Token::Sum) => {
                    self.next();
                    expr = Expr::UnaryOp(UnaryOp::Sum, Box::new(expr));
                }
                Some(Token::Prod) => {
                    self.next();
                    expr = Expr::UnaryOp(UnaryOp::Prod, Box::new(expr));
                }
                Some(Token::Scan) => {
                    self.next();
                    expr = Expr::UnaryOp(UnaryOp::Scan, Box::new(expr));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Parse lambda expression
    fn parse_lambda(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Lambda)?;
        
        // Count arrows to determine arity
        let mut arity = 0;
        while self.eat(&Token::Arrow) {
            arity += 1;
        }

        // If no arrows, it's λ→ body (single argument)
        if arity == 0 {
            self.expect(Token::Arrow)?;
            arity = 1;
        }

        let body = self.parse_expr()?;

        if arity == 1 {
            Ok(Expr::Lam(Box::new(body)))
        } else {
            Ok(Expr::LamN(arity, Box::new(body)))
        }
    }

    /// Parse parenthesized expression or tuple
    fn parse_paren_or_tuple(&mut self) -> ParseResult<Expr> {
        self.expect(Token::LParen)?;

        if self.eat(&Token::RParen) {
            return Ok(Expr::Tuple(vec![])); // Unit
        }

        let first = self.parse_expr()?;

        if self.eat(&Token::Comma) {
            // Tuple
            let mut exprs = vec![first];
            if !self.at(&Token::RParen) {
                exprs.push(self.parse_expr()?);
                while self.eat(&Token::Comma) {
                    exprs.push(self.parse_expr()?);
                }
            }
            self.expect(Token::RParen)?;
            Ok(Expr::Tuple(exprs))
        } else {
            // Parenthesized expression
            self.expect(Token::RParen)?;
            Ok(first)
        }
    }

    /// Parse array literal
    fn parse_array(&mut self) -> ParseResult<Expr> {
        self.expect(Token::LBracket)?;

        if self.eat(&Token::RBracket) {
            return Ok(Expr::Array(vec![]));
        }

        let mut exprs = vec![self.parse_expr()?];
        while self.eat(&Token::Comma) {
            exprs.push(self.parse_expr()?);
        }

        // Check for array fill syntax: [shape ; value]
        if self.eat(&Token::Semi) {
            let value = self.parse_expr()?;
            self.expect(Token::RBracket)?;
            return Ok(Expr::ArrayFill {
                shape: exprs,
                value: Box::new(value),
            });
        }

        self.expect(Token::RBracket)?;
        Ok(Expr::Array(exprs))
    }

    /// Parse angle-bracket tuple ⟨x, y, z⟩
    fn parse_angle_tuple(&mut self) -> ParseResult<Expr> {
        self.expect(Token::LAngle)?;

        if self.eat(&Token::RAngle) {
            return Ok(Expr::Tuple(vec![]));
        }

        let mut exprs = vec![self.parse_expr()?];
        while self.eat(&Token::Comma) {
            exprs.push(self.parse_expr()?);
        }
        self.expect(Token::RAngle)?;
        Ok(Expr::Tuple(exprs))
    }

    /// Parse let expression
    fn parse_let(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Let)?;

        let is_rec = self.eat(&Token::Rec);

        if is_rec {
            // let rec { ... } in body
            let mut bindings = Vec::new();
            
            // Optional braces
            let has_braces = self.eat(&Token::LBrace);
            
            loop {
                let name = self.expect_ident()?;
                // Accept either = or ← for bindings
                if !self.eat(&Token::Eq) && !self.eat(&Token::BackArrow) {
                    return Err(ParseError::Unexpected {
                        found: self.peek().cloned(),
                        expected: "'=' or '←'".into(),
                    });
                }
                let value = self.parse_expr()?;
                let name_box: Box<str> = name.into();
                bindings.push((Pattern::Var(Some(name_box)), value));

                if has_braces {
                    if !self.eat(&Token::Semi) && !self.at(&Token::RBrace) {
                        break;
                    }
                    if self.at(&Token::RBrace) {
                        break;
                    }
                } else {
                    if !self.eat(&Token::Comma) {
                        break;
                    }
                }
            }

            if has_braces {
                self.expect(Token::RBrace)?;
            }

            self.expect(Token::In)?;
            let body = self.parse_expr()?;
            Ok(Expr::LetRec { bindings, body: Box::new(body) })
        } else {
            // Parse first binding
            let pattern = self.parse_pattern()?;
            // Accept either = or ← for let bindings
            if !self.eat(&Token::Eq) && !self.eat(&Token::BackArrow) {
                return Err(ParseError::Unexpected {
                    found: self.peek().cloned(),
                    expected: "'=' or '←'".into(),
                });
            }
            let value = self.parse_expr()?;
            
            // Check for semicolon - indicates sequential bindings
            if self.eat(&Token::Semi) {
                // Build nested let expressions from sequential bindings
                let mut bindings = vec![(pattern, value)];
                
                // Parse additional bindings until we see 'in'
                while !self.at(&Token::In) && self.peek().is_some() {
                    let pat = self.parse_pattern()?;
                    if !self.eat(&Token::Eq) && !self.eat(&Token::BackArrow) {
                        return Err(ParseError::Unexpected {
                            found: self.peek().cloned(),
                            expected: "'=' or '←'".into(),
                        });
                    }
                    let val = self.parse_expr()?;
                    bindings.push((pat, val));
                    
                    // Only continue if there's a semicolon
                    if !self.eat(&Token::Semi) {
                        break;
                    }
                }
                
                self.expect(Token::In)?;
                let mut body = self.parse_expr()?;
                
                // Build nested lets from right to left
                for (pat, val) in bindings.into_iter().rev() {
                    body = Expr::Let {
                        pattern: pat,
                        value: Box::new(val),
                        body: Box::new(body),
                    };
                }
                
                Ok(body)
            } else {
                // Single binding - original behavior
                self.expect(Token::In)?;
                let body = self.parse_expr()?;
                Ok(Expr::Let {
                    pattern,
                    value: Box::new(value),
                    body: Box::new(body),
                })
            }
        }
    }

    /// Parse if expression
    fn parse_if(&mut self) -> ParseResult<Expr> {
        self.expect(Token::If)?;
        let cond = self.parse_expr()?;
        self.expect(Token::Then)?;
        let then_ = self.parse_expr()?;
        self.expect(Token::Else)?;
        let else_ = self.parse_expr()?;
        Ok(Expr::If {
            cond: Box::new(cond),
            then_: Box::new(then_),
            else_: Box::new(else_),
        })
    }

    /// Parse match expression
    fn parse_match(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Match)?;
        let scrutinee = self.parse_expr()?;
        
        // Optional 'with'
        self.eat(&Token::With);
        self.eat(&Token::LBrace);

        let mut arms = Vec::new();
        while !self.at(&Token::RBrace) && !self.at(&Token::End) && self.peek().is_some() {
            // Optional leading |
            self.eat(&Token::FnMid);
            self.eat(&Token::Pipe);
            
            let pattern = self.parse_pattern()?;
            
            // Optional guard
            let guard = if self.eat(&Token::If) {
                Some(self.parse_expr()?)
            } else {
                None
            };

            self.expect(Token::Arrow)?;
            let body = self.parse_expr()?;
            
            arms.push(MatchArm { pattern, guard, body });

            // Optional semicolon or newline between arms
            self.eat(&Token::Semi);
        }

        self.eat(&Token::RBrace);
        self.eat(&Token::End);

        Ok(Expr::Match {
            scrutinee: Box::new(scrutinee),
            arms,
        })
    }

    /// Parse do-notation
    fn parse_do(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Do)?;
        
        let init = self.parse_expr()?;
        let mut ops = Vec::new();

        while !self.at(&Token::End) && self.peek().is_some() {
            let op = match self.peek().cloned() {
                Some(Token::Map) => { self.next(); DoOp::Map(self.parse_expr()?) }
                Some(Token::Filter) => { self.next(); DoOp::Filter(self.parse_expr()?) }
                Some(Token::Bind) => { self.next(); DoOp::Bind(self.parse_expr()?) }
                Some(Token::Let) => {
                    self.next();
                    let pat = self.parse_pattern()?;
                    // Accept either = or ← for bindings
                    if !self.eat(&Token::Eq) && !self.eat(&Token::BackArrow) {
                        return Err(ParseError::Unexpected {
                            found: self.peek().cloned(),
                            expected: "'=' or '←'".into(),
                        });
                    }
                    let e = self.parse_expr()?;
                    DoOp::Let(pat, e)
                }
                Some(t) if t.is_binop() => {
                    let op = self.token_to_binop(&t);
                    self.next();
                    DoOp::Op(op, self.parse_expr()?)
                }
                _ => break,
            };
            ops.push(op);
        }

        self.eat(&Token::End);
        Ok(Expr::Do { init: Box::new(init), ops })
    }

    /// Convert token to BinOp
    fn token_to_binop(&self, token: &Token) -> BinOp {
        match token {
            Token::Plus => BinOp::Add,
            Token::Minus => BinOp::Sub,
            Token::Star => BinOp::Mul,
            Token::Slash => BinOp::Div,
            Token::Caret => BinOp::Pow,
            Token::Percent => BinOp::Mod,
            Token::PlusMinus => BinOp::PlusMinus,
            Token::Eq => BinOp::Eq,
            Token::Neq => BinOp::Neq,
            Token::Lt => BinOp::Lt,
            Token::Gt => BinOp::Gt,
            Token::Leq => BinOp::Leq,
            Token::Geq => BinOp::Geq,
            Token::And => BinOp::And,
            Token::Or => BinOp::Or,
            Token::Map => BinOp::Map,
            Token::Filter => BinOp::Filter,
            Token::Bind => BinOp::Bind,
            Token::Compose => BinOp::Compose,
            Token::ZipWith => BinOp::ZipWith,
            Token::Concat => BinOp::Concat,
            _ => BinOp::Add, // fallback
        }
    }

    /// Create binary operation expression
    fn make_binop(&self, token: Token, lhs: Expr, rhs: Expr) -> Expr {
        let op = self.token_to_binop(&token);
        Expr::BinOp(op, Box::new(lhs), Box::new(rhs))
    }

    // ============ Pattern Parsing ============

    /// Parse a pattern
    pub fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        let pat = self.parse_pattern_atom()?;
        
        // Or pattern (use Pipe token, not Or which is ||)
        if self.eat(&Token::Pipe) {
            let right = self.parse_pattern()?;
            return Ok(Pattern::Or(Box::new(pat), Box::new(right)));
        }

        Ok(pat)
    }

    fn parse_pattern_atom(&mut self) -> ParseResult<Pattern> {
        match self.peek().cloned() {
            Some(Token::Underscore) => { self.next(); Ok(Pattern::Wildcard) }
            Some(Token::Int(n)) => { self.next(); Ok(Pattern::Lit(Literal::Int(n))) }
            Some(Token::Float(f)) => { self.next(); Ok(Pattern::Lit(Literal::Float(f))) }
            Some(Token::Char(c)) => { self.next(); Ok(Pattern::Lit(Literal::Char(c))) }
            Some(Token::String(s)) => { self.next(); Ok(Pattern::Lit(Literal::String(s.into()))) }
            Some(Token::True) => { self.next(); Ok(Pattern::Lit(Literal::True)) }
            Some(Token::False) => { self.next(); Ok(Pattern::Lit(Literal::False)) }
            Some(Token::Pi) => { self.next(); Ok(Pattern::Lit(Literal::Float(std::f64::consts::PI))) }
            Some(Token::Euler) => { self.next(); Ok(Pattern::Lit(Literal::Float(std::f64::consts::E))) }

            Some(Token::Ident(name)) | Some(Token::TyVar(name)) | Some(Token::AplIdent(name)) => {
                self.next();
                // Check if it's a variant constructor (starts with uppercase)
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    let payload = if self.peek().map(|t| t.can_start_expr()).unwrap_or(false)
                        && !self.peek().map(|t| t.is_binop()).unwrap_or(false)
                        && !matches!(self.peek(), Some(Token::Arrow) | Some(Token::If)) {
                        Some(Box::new(self.parse_pattern_atom()?))
                    } else {
                        None
                    };
                    Ok(Pattern::Variant { constructor: name.into(), payload })
                } else {
                    Ok(Pattern::Var(Some(name.into())))
                }
            }

            Some(Token::LParen) => {
                self.next();
                if self.eat(&Token::RParen) {
                    return Ok(Pattern::Tuple(vec![]));
                }
                let first = self.parse_pattern()?;
                if self.eat(&Token::Comma) {
                    let mut pats = vec![first];
                    if !self.at(&Token::RParen) {
                        pats.push(self.parse_pattern()?);
                        while self.eat(&Token::Comma) {
                            pats.push(self.parse_pattern()?);
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Pattern::Tuple(pats))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(first)
                }
            }

            Some(Token::LBracket) => {
                self.next();
                if self.eat(&Token::RBracket) {
                    return Ok(Pattern::Array(vec![]));
                }
                let mut pats = vec![self.parse_pattern()?];
                
                // Check for split pattern [head | tail]
                if self.eat(&Token::FnMid) || self.eat(&Token::Pipe) {
                    let tail = self.parse_pattern()?;
                    self.expect(Token::RBracket)?;
                    return Ok(Pattern::ArraySplit { head: pats, tail: Box::new(tail) });
                }

                while self.eat(&Token::Comma) {
                    pats.push(self.parse_pattern()?);
                    
                    // Check for split after comma
                    if self.eat(&Token::FnMid) || self.eat(&Token::Pipe) {
                        let tail = self.parse_pattern()?;
                        self.expect(Token::RBracket)?;
                        return Ok(Pattern::ArraySplit { head: pats, tail: Box::new(tail) });
                    }
                }
                self.expect(Token::RBracket)?;
                Ok(Pattern::Array(pats))
            }

            Some(Token::LAngle) => {
                self.next();
                if self.eat(&Token::RAngle) {
                    return Ok(Pattern::Tuple(vec![]));
                }
                let mut pats = vec![self.parse_pattern()?];
                while self.eat(&Token::Comma) {
                    pats.push(self.parse_pattern()?);
                }
                self.expect(Token::RAngle)?;
                Ok(Pattern::Tuple(pats))
            }

            other => Err(ParseError::Unexpected {
                found: other,
                expected: "pattern".into(),
            }),
        }
    }

    // ============ Type Parsing ============

    /// Parse a type
    pub fn parse_type(&mut self) -> ParseResult<Type> {
        let ty = self.parse_type_atom()?;

        // Uncertain type: T ± U
        if self.eat(&Token::PlusMinus) {
            let uncertainty = self.parse_type_atom()?;
            return Ok(Type::Uncertain(Box::new(ty), Box::new(uncertainty)));
        }

        // Function type
        if self.eat(&Token::Arrow) {
            let ret = self.parse_type()?;
            return Ok(Type::Fn(Box::new(ty), Box::new(ret)));
        }

        Ok(ty)
    }

    fn parse_type_atom(&mut self) -> ParseResult<Type> {
        match self.peek().cloned() {
            // Primitive types
            Some(Token::TyF64) => { self.next(); Ok(Type::Prim(PrimType::F64)) }
            Some(Token::TyF32) => { self.next(); Ok(Type::Prim(PrimType::F32)) }
            Some(Token::TyI64) => { self.next(); Ok(Type::Prim(PrimType::I64)) }
            Some(Token::TyI32) => { self.next(); Ok(Type::Prim(PrimType::I32)) }
            Some(Token::TyU64) => { self.next(); Ok(Type::Prim(PrimType::U64)) }
            Some(Token::TyU8) => { self.next(); Ok(Type::Prim(PrimType::U8)) }
            Some(Token::TyBool) => { self.next(); Ok(Type::Prim(PrimType::Bool)) }
            Some(Token::TyChar) => { self.next(); Ok(Type::Prim(PrimType::Char)) }
            Some(Token::TyByte) => { self.next(); Ok(Type::Prim(PrimType::Byte)) }
            Some(Token::TyString) => { self.next(); Ok(Type::Prim(PrimType::String)) }
            Some(Token::TyNat) => { self.next(); Ok(Type::Prim(PrimType::Nat)) }
            Some(Token::TyInt) => { self.next(); Ok(Type::Prim(PrimType::Int)) }
            Some(Token::TyUnit) => { self.next(); Ok(Type::Tuple(vec![])) }

            // Type variable
            Some(Token::TyVar(v)) => { self.next(); Ok(Type::Var(v.into())) }
            Some(Token::Ident(name)) | Some(Token::AplIdent(name)) => { self.next(); Ok(Type::Var(name.into())) }

            // Tensor type [shape]T
            Some(Token::LBracket) => {
                self.next();
                let shape = self.parse_shape()?;
                self.expect(Token::RBracket)?;
                let elem = self.parse_type_atom()?;
                Ok(Type::Tensor(shape, Box::new(elem)))
            }

            // Tuple type (T, U, V) or parenthesized
            Some(Token::LParen) => {
                self.next();
                if self.eat(&Token::RParen) {
                    return Ok(Type::Tuple(vec![]));
                }
                let first = self.parse_type()?;
                if self.eat(&Token::Comma) {
                    let mut fields = vec![TupleField { label: None, ty: first }];
                    fields.push(TupleField { label: None, ty: self.parse_type()? });
                    while self.eat(&Token::Comma) {
                        fields.push(TupleField { label: None, ty: self.parse_type()? });
                    }
                    self.expect(Token::RParen)?;
                    Ok(Type::Tuple(fields))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(first)
                }
            }

            // Angle bracket tuple type ⟨T, U, V⟩ or record type ⟨x: T, y: U⟩
            Some(Token::LAngle) => {
                self.next();
                if self.eat(&Token::RAngle) {
                    return Ok(Type::Tuple(vec![]));  // Empty tuple ⟨⟩
                }
                
                // Simple approach: try to parse as unnamed tuple first
                // If we see Ident/TyVar, check next token
                // But we can't easily backtrack, so let's be smarter:
                // Parse first element, check if followed by colon (record) or comma/rangle (tuple)
                
                // Try parsing as record if first thing looks like "name:"
                if let Some(Token::Ident(name)) | Some(Token::TyVar(name)) | Some(Token::AplIdent(name)) = self.peek().cloned() {
                    self.next();
                    if self.at(&Token::Colon) {
                        // It's a record! name: Type
                        self.expect(Token::Colon)?;
                        let ty = self.parse_type()?;
                        let mut fields = vec![TupleField { label: Some(name.into()), ty }];
                        
                        while self.eat(&Token::Comma) {
                            let label = self.expect_ident()?;
                            self.expect(Token::Colon)?;
                            let ty = self.parse_type()?;
                            fields.push(TupleField { label: Some(label.into()), ty });
                        }
                        
                        self.expect(Token::RAngle)?;
                        return Ok(Type::Tuple(fields));
                    }
                    // Not a colon, so this identifier is actually a type variable
                    // Continue parsing it as an unnamed tuple with this type var as first element
                    let first_ty = Type::Var(name.into());
                    if self.eat(&Token::Comma) {
                        let mut fields = vec![TupleField { label: None, ty: first_ty }];
                        loop {
                            fields.push(TupleField { label: None, ty: self.parse_type()? });
                            if !self.eat(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RAngle)?;
                        return Ok(Type::Tuple(fields));
                    } else {
                        self.expect(Token::RAngle)?;
                        return Ok(first_ty);
                    }
                }
                
                // Not an identifier/tyvar, so parse as regular tuple
                let first = self.parse_type()?;
                if self.eat(&Token::Comma) {
                    let mut fields = vec![TupleField { label: None, ty: first }];
                    loop {
                        fields.push(TupleField { label: None, ty: self.parse_type()? });
                        if !self.eat(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RAngle)?;
                    Ok(Type::Tuple(fields))
                } else {
                    self.expect(Token::RAngle)?;
                    Ok(first)
                }
            }

            // Refinement type {x : T | P}
            Some(Token::LBrace) => {
                self.next();
                let var = self.expect_ident()?;
                self.expect(Token::Colon)?;
                let base = self.parse_type()?;
                self.expect(Token::Pipe)?;
                let pred = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(Type::Refinement {
                    name: var.into(),
                    base: Box::new(base),
                    predicate: Box::new(pred),
                })
            }

            // Forall
            Some(Token::Forall) => {
                self.next();
                let mut params = Vec::new();
                while let Some(Token::Ident(name)) | Some(Token::TyVar(name)) | Some(Token::AplIdent(name)) = self.peek().cloned() {
                    self.next();
                    params.push(TypeParam { name: name.into(), kind: TypeParamKind::Type });
                }
                self.expect(Token::Dot)?;
                let body = self.parse_type()?;
                Ok(Type::Forall(params, Box::new(body)))
            }

            // Option type T?
            Some(Token::Question) => {
                self.next();
                let inner = self.parse_type_atom()?;
                Ok(Type::Option(Box::new(inner)))
            }

            other => Err(ParseError::Unexpected {
                found: other,
                expected: "type".into(),
            }),
        }
    }

    /// Parse tensor shape
    fn parse_shape(&mut self) -> ParseResult<Shape> {
        let mut dims = Vec::new();

        while !self.at(&Token::RBracket) {
            let dim = match self.peek().cloned() {
                Some(Token::Int(n)) => { self.next(); Dim::Const(n as u64) }
                Some(Token::Ident(name)) | Some(Token::TyVar(name)) | Some(Token::AplIdent(name)) => { self.next(); Dim::Var(name.into()) }
                _ => break,
            };
            dims.push(dim);
        }

        Ok(Shape(dims))
    }

    // ============ Declaration Parsing ============

    /// Parse a module (sequence of declarations)
    pub fn parse_module(&mut self, name: &str) -> ParseResult<Module> {
        let mut decls = Vec::new();
        
        while self.peek().is_some() {
            if let Some(decl) = self.try_parse_decl()? {
                decls.push(decl);
            } else {
                break;
            }
        }

        Ok(Module {
            name: Some(name.into()),
            decls,
        })
    }

    /// Try to parse a declaration
    fn try_parse_decl(&mut self) -> ParseResult<Option<Decl>> {
        match self.peek() {
            Some(Token::FnStart) => Ok(Some(self.parse_fn_decl()?)),
            Some(Token::Let) => Ok(Some(self.parse_let_decl()?)),
            Some(Token::Type) => Ok(Some(self.parse_type_decl()?)),
            Some(Token::Enum) => Ok(Some(self.parse_enum_decl()?)),
            Some(Token::Use) => Ok(Some(self.parse_use_decl()?)),
            _ => Ok(None),
        }
    }

    /// Parse use declaration: use "path/to/file.goth"
    fn parse_use_decl(&mut self) -> ParseResult<Decl> {
        self.expect(Token::Use)?;
        match self.next() {
            Some(Token::String(path)) => {
                Ok(Decl::Use(goth_ast::decl::UseDecl::new(path)))
            }
            other => Err(ParseError::Unexpected {
                found: other,
                expected: "string path".to_string()
            }),
        }
    }

    /// Parse function declaration
    fn parse_fn_decl(&mut self) -> ParseResult<Decl> {
        self.expect(Token::FnStart)?;

        let name = self.expect_ident()?;
        self.expect(Token::Colon)?;
        let sig = self.parse_type()?;

        // Optional where clause, preconditions, postconditions, effects
        let constraints = Vec::new();
        let mut preconditions = Vec::new();
        let mut postconditions = Vec::new();
        let mut effects = Effects::pure();

        while self.eat(&Token::FnMid) {
            match self.peek() {
                Some(Token::Diamond) => {
                    self.next();  // consume ◇
                    // Parse effect name: io, mut, rand, div, etc.
                    let effect_name = self.expect_ident()?;
                    let effect = match effect_name.as_str() {
                        "io" | "IO" => Effect::Io,
                        "mut" | "Mut" => Effect::Mut,
                        "rand" | "Rand" => Effect::Rand,
                        "div" | "Div" => Effect::Div,
                        other => Effect::Custom(other.into()),
                    };
                    effects = effects.with(effect);
                }
                Some(Token::Where) => {
                    self.next();
                    // Parse constraints (simplified)
                }
                Some(Token::Turnstile) => {
                    self.next();
                    preconditions.push(self.parse_expr()?);
                }
                Some(Token::Models) => {
                    self.next();
                    postconditions.push(self.parse_expr()?);
                }
                _ => break,
            }
        }

        self.expect(Token::FnEnd)?;
        let body = self.parse_expr()?;

        Ok(Decl::Fn(FnDecl {
            name: name.into(),
            type_params: vec![],
            signature: sig,
            effects,
            constraints,
            preconditions,
            postconditions,
            body,
        }))
    }

    /// Parse let declaration
    fn parse_let_decl(&mut self) -> ParseResult<Decl> {
        self.expect(Token::Let)?;
        let name = self.expect_ident()?;
        
        let type_ = if self.eat(&Token::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        // Accept either = or ← for let declarations
        if !self.eat(&Token::Eq) && !self.eat(&Token::BackArrow) {
            return Err(ParseError::Unexpected {
                found: self.peek().cloned(),
                expected: "'=' or '←'".into(),
            });
        }
        let value = self.parse_expr()?;

        Ok(Decl::Let(goth_ast::decl::LetDecl {
            name: name.into(),
            type_,
            value,
        }))
    }

    /// Parse type declaration
    fn parse_type_decl(&mut self) -> ParseResult<Decl> {
        self.expect(Token::Type)?;
        let name = self.expect_ident()?;

        let mut params = Vec::new();
        while let Some(Token::Ident(p)) | Some(Token::TyVar(p)) | Some(Token::AplIdent(p)) = self.peek().cloned() {
            self.next();
            params.push(TypeParam { name: p.into(), kind: TypeParamKind::Type });
        }

        self.expect(Token::Eq)?;
        let def = self.parse_type()?;

        Ok(Decl::Type(TypeDecl {
            name: name.into(),
            params,
            definition: def,
        }))
    }

    /// Parse enum declaration: enum Name τ where Variant₁ T₁ | Variant₂ T₂ | ...
    /// Examples:
    ///   enum Bool where True | False
    ///   enum Option τ where Some τ | None
    ///   enum Either α β where Left α | Right β
    fn parse_enum_decl(&mut self) -> ParseResult<Decl> {
        use goth_ast::decl::{EnumDecl, EnumVariant};

        self.expect(Token::Enum)?;
        let name = self.expect_ident()?;

        // Parse optional type parameters (lowercase identifiers or Greek letters)
        let mut params = Vec::new();
        while let Some(Token::Ident(p)) | Some(Token::TyVar(p)) | Some(Token::AplIdent(p)) = self.peek().cloned() {
            // Only accept lowercase type variables
            if p.chars().next().map(|c| c.is_lowercase() || c == 'τ' || c == 'α' || c == 'β' || c == 'σ').unwrap_or(false) {
                self.next();
                params.push(TypeParam { name: p.into(), kind: TypeParamKind::Type });
            } else {
                break;
            }
        }

        self.expect(Token::Where)?;

        // Parse variants separated by |
        let mut variants = Vec::new();
        loop {
            // Variant name must start with uppercase
            let variant_name = self.expect_ident()?;

            // Optional payload type (check if next token could start a type)
            let payload = if self.peek_is_type_start() {
                Some(self.parse_type()?)
            } else {
                None
            };

            variants.push(EnumVariant {
                name: variant_name.into(),
                payload,
            });

            // Check for more variants
            if !self.eat(&Token::Pipe) {
                break;
            }
        }

        Ok(Decl::Enum(EnumDecl {
            name: name.into(),
            params,
            variants,
        }))
    }

    /// Check if the next token could start a type (for optional payload parsing)
    fn peek_is_type_start(&mut self) -> bool {
        match self.peek() {
            Some(Token::Ident(s)) => {
                // Type names start with uppercase, type variables with lowercase
                s.chars().next().map(|c| c.is_uppercase() || c.is_lowercase()).unwrap_or(false)
            }
            Some(Token::TyVar(_)) | Some(Token::AplIdent(_)) => true,
            Some(Token::LParen) | Some(Token::LBracket) | Some(Token::LAngle) => true,
            _ => false,
        }
    }
}

// ============ Convenience Functions ============

/// Parse an expression from a string
pub fn parse_expr(source: &str) -> ParseResult<Expr> {
    Parser::new(source).parse_expr()
}

/// Parse a type from a string
pub fn parse_type(source: &str) -> ParseResult<Type> {
    Parser::new(source).parse_type()
}

/// Parse a pattern from a string
pub fn parse_pattern(source: &str) -> ParseResult<Pattern> {
    Parser::new(source).parse_pattern()
}

/// Parse a module from a string
pub fn parse_module(source: &str, name: &str) -> ParseResult<Module> {
    Parser::new(source).parse_module(name)
}