//! Type system for Goth
//!
//! Features:
//! - Primitive types (F64, I64, Bool, etc.)
//! - Tensor types with shapes
//! - Tuples (products) and Variants (sums)
//! - Function types with effects
//! - Interval types (value ranges)
//! - Refinement types (predicates)
//! - Universal and existential quantification

use serde::{Deserialize, Serialize};
use crate::shape::Shape;
use crate::effect::Effects;
use crate::interval::IntervalSet;
use crate::expr::Expr;

/// Primitive types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimType {
    // Floating point
    F64,
    F32,

    // Signed integers
    I64,
    I32,
    I16,
    I8,

    // Unsigned integers
    U64,
    U32,
    U16,
    U8,

    // Other primitives
    Bool,
    Char,   // Unicode scalar (32-bit)
    Byte,   // Alias for U8
    String, // UTF-8 string (heap-allocated)

    // Arbitrary precision (for compile-time computation)
    Nat,    // ℕ - natural numbers
    Int,    // ℤ - integers

    // Complex number types
    Complex,    // ℂ - complex (f64, f64)
    Quaternion, // ℍ - quaternion (f64, f64, f64, f64)
}

/// Type representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    /// Primitive type
    Prim(PrimType),

    /// Tensor type: [shape]elem
    Tensor(Shape, Box<Type>),

    /// Tuple (product): ⟨T₀, T₁, ...⟩
    Tuple(Vec<TupleField>),

    /// Variant (sum): ⟨L T | R U | ...⟩
    Variant(Vec<VariantArm>),

    /// Function type: A → B
    Fn(Box<Type>, Box<Type>),

    /// Type with effect annotation: T⊢ε
    Effectful(Box<Type>, Effects),

    /// Type with interval constraint: T⊢[a..b]
    Interval(Box<Type>, IntervalSet),

    /// Refinement type: {x : T | P(x)}
    Refinement {
        name: Box<str>,
        base: Box<Type>,
        predicate: Box<Expr>,
    },

    /// Universal quantification: ∀α. T
    Forall(Vec<TypeParam>, Box<Type>),

    /// Existential quantification: ∃α. T
    Exists(Vec<TypeParam>, Box<Type>),

    /// Type variable
    Var(Box<str>),

    /// Type application: F⟨T⟩
    App(Box<Type>, Vec<Type>),

    /// Optional/nullable: T?
    Option(Box<Type>),

    /// Uncertain type (with error): T± or T ± U (value ± uncertainty)
    Uncertain(Box<Type>, Box<Type>),

    /// Hole (for inference): _
    Hole,
}

/// A field in a tuple type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TupleField {
    /// Optional label
    pub label: Option<Box<str>>,
    /// Field type
    pub ty: Type,
}

/// An arm in a variant type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantArm {
    /// Constructor name
    pub name: Box<str>,
    /// Optional payload type
    pub payload: Option<Type>,
}

/// Type parameter (for quantification)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeParam {
    pub name: Box<str>,
    pub kind: TypeParamKind,
}

/// Kind of type parameter
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeParamKind {
    /// Type variable
    Type,
    /// Shape/dimension variable
    Shape,
    /// Effect variable
    Effect,
}

/// Type constraint (for where clauses)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Type implements typeclass: T: Class
    HasClass(Type, Box<str>),

    /// Shape comparison: n > 0, m = n × k
    ShapeCmp(Shape, ShapeCmpOp, Shape),

    /// Types are equal
    TypeEq(Type, Type),

    /// Custom constraint (for refinements)
    Predicate(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeCmpOp {
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
}

// ============ Constructors ============

impl Type {
    // Primitives
    pub fn f64() -> Self { Type::Prim(PrimType::F64) }
    pub fn f32() -> Self { Type::Prim(PrimType::F32) }
    pub fn i64() -> Self { Type::Prim(PrimType::I64) }
    pub fn i32() -> Self { Type::Prim(PrimType::I32) }
    pub fn u64() -> Self { Type::Prim(PrimType::U64) }
    pub fn u8() -> Self { Type::Prim(PrimType::U8) }
    pub fn bool() -> Self { Type::Prim(PrimType::Bool) }
    pub fn char() -> Self { Type::Prim(PrimType::Char) }
    pub fn nat() -> Self { Type::Prim(PrimType::Nat) }
    pub fn complex() -> Self { Type::Prim(PrimType::Complex) }
    pub fn quaternion() -> Self { Type::Prim(PrimType::Quaternion) }

    // Tensor
    pub fn tensor(shape: Shape, elem: Type) -> Self {
        Type::Tensor(shape, Box::new(elem))
    }

    // Vector shorthand
    pub fn vector(n: impl Into<crate::shape::Dim>, elem: Type) -> Self {
        Type::Tensor(Shape::vector(n.into()), Box::new(elem))
    }

    // Function
    pub fn func(arg: Type, ret: Type) -> Self {
        Type::Fn(Box::new(arg), Box::new(ret))
    }

    // Multi-arg function (curried)
    pub fn func_n(args: impl IntoIterator<Item = Type>, ret: Type) -> Self {
        let args: Vec<_> = args.into_iter().collect();
        args.into_iter().rev().fold(ret, |acc, arg| Type::func(arg, acc))
    }

    // Tuple
    pub fn tuple(fields: Vec<Type>) -> Self {
        Type::Tuple(fields.into_iter().map(|ty| TupleField { label: None, ty }).collect())
    }

    // Labeled tuple (record)
    pub fn record(fields: Vec<(&str, Type)>) -> Self {
        Type::Tuple(fields.into_iter().map(|(name, ty)| TupleField {
            label: Some(name.into()),
            ty,
        }).collect())
    }

    // Unit type (empty tuple)
    pub fn unit() -> Self {
        Type::Tuple(vec![])
    }

    // Type variable
    pub fn var(name: impl Into<Box<str>>) -> Self {
        Type::Var(name.into())
    }

    // Option
    pub fn option(inner: Type) -> Self {
        Type::Option(Box::new(inner))
    }

    // With effect
    pub fn with_effect(self, effects: Effects) -> Self {
        if effects.is_pure() {
            self
        } else {
            Type::Effectful(Box::new(self), effects)
        }
    }

    // With interval
    pub fn with_interval(self, interval: IntervalSet) -> Self {
        Type::Interval(Box::new(self), interval)
    }

    // Check if this is a function type
    pub fn is_fn(&self) -> bool {
        matches!(self, Type::Fn(_, _))
    }

    // Check if this is a tensor type
    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor(_, _))
    }

    // Get the return type if this is a function
    pub fn return_type(&self) -> Option<&Type> {
        match self {
            Type::Fn(_, ret) => Some(ret),
            _ => None,
        }
    }
}

impl TupleField {
    pub fn new(ty: Type) -> Self {
        TupleField { label: None, ty }
    }

    pub fn labeled(name: impl Into<Box<str>>, ty: Type) -> Self {
        TupleField { label: Some(name.into()), ty }
    }
}

impl VariantArm {
    pub fn new(name: impl Into<Box<str>>) -> Self {
        VariantArm { name: name.into(), payload: None }
    }

    pub fn with_payload(name: impl Into<Box<str>>, payload: Type) -> Self {
        VariantArm { name: name.into(), payload: Some(payload) }
    }
}

impl PrimType {
    pub fn is_numeric(&self) -> bool {
        !matches!(self, PrimType::Bool | PrimType::Char)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, PrimType::F64 | PrimType::F32 | PrimType::Complex | PrimType::Quaternion)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, 
            PrimType::I64 | PrimType::I32 | PrimType::I16 | PrimType::I8 |
            PrimType::U64 | PrimType::U32 | PrimType::U16 | PrimType::U8 |
            PrimType::Byte | PrimType::Nat | PrimType::Int
        )
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, 
            PrimType::I64 | PrimType::I32 | PrimType::I16 | PrimType::I8 |
            PrimType::Int | PrimType::F64 | PrimType::F32
        )
    }

    pub fn bit_width(&self) -> Option<u8> {
        match self {
            PrimType::F64 | PrimType::I64 | PrimType::U64 => Some(64),
            PrimType::F32 | PrimType::I32 | PrimType::U32 | PrimType::Char => Some(32),
            PrimType::I16 | PrimType::U16 => Some(16),
            PrimType::I8 | PrimType::U8 | PrimType::Byte | PrimType::Bool => Some(8),
            PrimType::Complex => Some(128),    // 2 × f64
            PrimType::Quaternion => None,       // 4 × f64 = 256, doesn't fit u8
            PrimType::Nat | PrimType::Int | PrimType::String => None,
        }
    }
}

// ============ Display ============

impl std::fmt::Display for PrimType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimType::F64 => write!(f, "F64"),
            PrimType::F32 => write!(f, "F32"),
            PrimType::I64 => write!(f, "I64"),
            PrimType::I32 => write!(f, "I32"),
            PrimType::I16 => write!(f, "I16"),
            PrimType::I8 => write!(f, "I8"),
            PrimType::U64 => write!(f, "U64"),
            PrimType::U32 => write!(f, "U32"),
            PrimType::U16 => write!(f, "U16"),
            PrimType::U8 => write!(f, "U8"),
            PrimType::Bool => write!(f, "Bool"),
            PrimType::Char => write!(f, "Char"),
            PrimType::Byte => write!(f, "Byte"),
            PrimType::String => write!(f, "String"),
            PrimType::Nat => write!(f, "ℕ"),
            PrimType::Int => write!(f, "ℤ"),
            PrimType::Complex => write!(f, "ℂ"),
            PrimType::Quaternion => write!(f, "ℍ"),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Prim(p) => write!(f, "{}", p),
            Type::Tensor(shape, elem) => write!(f, "{}{}",  shape, elem),
            Type::Tuple(fields) if fields.is_empty() => write!(f, "⟨⟩"),
            Type::Tuple(fields) => {
                write!(f, "⟨")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    if let Some(label) = &field.label {
                        write!(f, "{}: ", label)?;
                    }
                    write!(f, "{}", field.ty)?;
                }
                write!(f, "⟩")
            }
            Type::Variant(arms) => {
                write!(f, "⟨")?;
                for (i, arm) in arms.iter().enumerate() {
                    if i > 0 { write!(f, " | ")?; }
                    write!(f, "{}", arm.name)?;
                    if let Some(payload) = &arm.payload {
                        write!(f, " {}", payload)?;
                    }
                }
                write!(f, "⟩")
            }
            Type::Fn(arg, ret) => write!(f, "{} → {}", arg, ret),
            Type::Effectful(ty, eff) => write!(f, "{}⊢{}", ty, eff),
            Type::Interval(ty, interval) => write!(f, "{}⊢{}", ty, interval),
            Type::Refinement { name, base, predicate: _ } => {
                write!(f, "{{{} : {} | ...}}", name, base)
            }
            Type::Forall(params, ty) => {
                write!(f, "∀")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "{}", p.name)?;
                }
                write!(f, ". {}", ty)
            }
            Type::Exists(params, ty) => {
                write!(f, "∃")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "{}", p.name)?;
                }
                write!(f, ". {}", ty)
            }
            Type::Var(name) => write!(f, "{}", name),
            Type::App(func, args) => {
                write!(f, "{}⟨", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, "⟩")
            }
            Type::Option(inner) => write!(f, "{}?", inner),
            Type::Uncertain(val, unc) => write!(f, "{} ± {}", val, unc),
            Type::Hole => write!(f, "_"),
        }
    }
}
