//! Top-level declarations in Goth

use serde::{Deserialize, Serialize};
use crate::expr::Expr;
use crate::types::{Type, Constraint, TypeParam};
use crate::effect::Effects;

/// A complete Goth program/module
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub name: Option<Box<str>>,
    pub decls: Vec<Decl>,
}

/// Top-level declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Decl {
    /// Function definition
    Fn(FnDecl),

    /// Type alias
    Type(TypeDecl),

    /// Typeclass definition
    Class(ClassDecl),

    /// Typeclass implementation
    Impl(ImplDecl),

    /// Top-level let binding
    Let(LetDecl),

    /// Operator definition
    Op(OpDecl),
}

/// Function declaration
/// ╭─ name : TypeSig
/// │  ◇io           (effect annotation)
/// │  where Constraints
/// │  ⊢ Preconditions
/// │  ⊨ Postconditions
/// ╰─ Implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnDecl {
    pub name: Box<str>,
    pub type_params: Vec<TypeParam>,
    pub signature: Type,
    pub effects: Effects,
    pub constraints: Vec<Constraint>,
    pub preconditions: Vec<Expr>,
    pub postconditions: Vec<Expr>,
    pub body: Expr,
}

/// Type alias declaration
/// Name ≡ Type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeDecl {
    pub name: Box<str>,
    pub params: Vec<TypeParam>,
    pub definition: Type,
}

/// Typeclass declaration
/// class ClassName τ where ...
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassDecl {
    pub name: Box<str>,
    pub param: TypeParam,
    pub superclasses: Vec<Box<str>>,
    pub methods: Vec<MethodSig>,
}

/// Method signature in a class
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodSig {
    pub name: Box<str>,
    pub signature: Type,
    pub default: Option<Expr>,
}

/// Typeclass implementation
/// impl ClassName Type where ...
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImplDecl {
    pub class_name: Box<str>,
    pub target: Type,
    pub constraints: Vec<Constraint>,
    pub methods: Vec<MethodImpl>,
}

/// Method implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodImpl {
    pub name: Box<str>,
    pub body: Expr,
}

/// Top-level let binding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetDecl {
    pub name: Box<str>,
    pub type_: Option<Type>,
    pub value: Expr,
}

/// Operator declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpDecl {
    pub name: Box<str>,
    pub glyph: Box<str>,
    pub ascii: Box<str>,
    pub signature: Type,
    pub assoc: crate::op::Assoc,
    pub precedence: u8,
    pub body: Expr,
}

// ============ Constructors ============

impl Module {
    pub fn new(decls: Vec<Decl>) -> Self {
        Module { name: None, decls }
    }

    pub fn named(name: impl Into<Box<str>>, decls: Vec<Decl>) -> Self {
        Module { name: Some(name.into()), decls }
    }
}

impl FnDecl {
    /// Simple function with just name, signature, and body
    pub fn simple(name: impl Into<Box<str>>, sig: Type, body: Expr) -> Self {
        FnDecl {
            name: name.into(),
            type_params: vec![],
            signature: sig,
            effects: Effects::pure(),
            constraints: vec![],
            preconditions: vec![],
            postconditions: vec![],
            body,
        }
    }

    /// Add effects annotation
    pub fn with_effects(mut self, effects: Effects) -> Self {
        self.effects = effects;
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Add a precondition
    pub fn with_pre(mut self, pre: Expr) -> Self {
        self.preconditions.push(pre);
        self
    }

    /// Add a postcondition
    pub fn with_post(mut self, post: Expr) -> Self {
        self.postconditions.push(post);
        self
    }

    /// Add type parameters
    pub fn with_type_params(mut self, params: Vec<TypeParam>) -> Self {
        self.type_params = params;
        self
    }
}

impl TypeDecl {
    pub fn alias(name: impl Into<Box<str>>, def: Type) -> Self {
        TypeDecl {
            name: name.into(),
            params: vec![],
            definition: def,
        }
    }

    pub fn generic(name: impl Into<Box<str>>, params: Vec<TypeParam>, def: Type) -> Self {
        TypeDecl {
            name: name.into(),
            params,
            definition: def,
        }
    }
}

impl ClassDecl {
    pub fn new(name: impl Into<Box<str>>, param: TypeParam) -> Self {
        ClassDecl {
            name: name.into(),
            param,
            superclasses: vec![],
            methods: vec![],
        }
    }

    pub fn with_method(mut self, name: impl Into<Box<str>>, sig: Type) -> Self {
        self.methods.push(MethodSig {
            name: name.into(),
            signature: sig,
            default: None,
        });
        self
    }

    pub fn with_default_method(mut self, name: impl Into<Box<str>>, sig: Type, default: Expr) -> Self {
        self.methods.push(MethodSig {
            name: name.into(),
            signature: sig,
            default: Some(default),
        });
        self
    }

    pub fn extends(mut self, superclass: impl Into<Box<str>>) -> Self {
        self.superclasses.push(superclass.into());
        self
    }
}

impl ImplDecl {
    pub fn new(class: impl Into<Box<str>>, target: Type) -> Self {
        ImplDecl {
            class_name: class.into(),
            target,
            constraints: vec![],
            methods: vec![],
        }
    }

    pub fn with_method(mut self, name: impl Into<Box<str>>, body: Expr) -> Self {
        self.methods.push(MethodImpl {
            name: name.into(),
            body,
        });
        self
    }
}

impl LetDecl {
    pub fn new(name: impl Into<Box<str>>, value: Expr) -> Self {
        LetDecl {
            name: name.into(),
            type_: None,
            value,
        }
    }

    pub fn typed(name: impl Into<Box<str>>, ty: Type, value: Expr) -> Self {
        LetDecl {
            name: name.into(),
            type_: Some(ty),
            value,
        }
    }
}

// ============ Into Decl ============

impl From<FnDecl> for Decl {
    fn from(f: FnDecl) -> Self { Decl::Fn(f) }
}

impl From<TypeDecl> for Decl {
    fn from(t: TypeDecl) -> Self { Decl::Type(t) }
}

impl From<ClassDecl> for Decl {
    fn from(c: ClassDecl) -> Self { Decl::Class(c) }
}

impl From<ImplDecl> for Decl {
    fn from(i: ImplDecl) -> Self { Decl::Impl(i) }
}

impl From<LetDecl> for Decl {
    fn from(l: LetDecl) -> Self { Decl::Let(l) }
}
