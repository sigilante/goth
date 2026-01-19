//! Middle IR for Goth
//!
//! MIR is a simplified, explicit form where:
//! - De Bruijn indices are resolved to explicit locals
//! - Closures are converted to explicit environment passing
//! - Polymorphism is instantiated (monomorphic)
//! - Pattern matching is compiled to decision trees

use goth_ast::types::Type;
use goth_ast::op::{BinOp, UnaryOp};

/// A MIR program is a collection of functions
#[derive(Debug, Clone)]
pub struct Program {
    /// All functions (including lifted lambdas)
    pub functions: Vec<Function>,
    /// Entry point function name
    pub entry: String,
}

/// A function in MIR
#[derive(Debug, Clone)]
pub struct Function {
    /// Unique name (may be generated for lambdas)
    pub name: String,
    /// Parameter types
    pub params: Vec<Type>,
    /// Return type
    pub ret_ty: Type,
    /// Function body (single block for simple functions)
    pub body: Block,
    /// Additional blocks for control flow (if/match)
    pub blocks: Vec<(BlockId, Block)>,
    /// Is this a closure? If so, first param is environment
    pub is_closure: bool,
}

/// A basic block (sequence of statements + terminator)
#[derive(Debug, Clone)]
pub struct Block {
    /// Statements in order
    pub stmts: Vec<Stmt>,
    /// Block terminator
    pub term: Terminator,
}

/// Statement (defines a local variable)
#[derive(Debug, Clone)]
pub struct Stmt {
    /// Local variable being defined
    pub dest: LocalId,
    /// Type of the variable
    pub ty: Type,
    /// Right-hand side
    pub rhs: Rhs,
}

/// Right-hand side of a statement
#[derive(Debug, Clone)]
pub enum Rhs {
    /// Use another local
    Use(Operand),
    
    /// Literal constant
    Const(Constant),
    
    /// Binary operation
    BinOp(BinOp, Operand, Operand),
    
    /// Unary operation
    UnaryOp(UnaryOp, Operand),
    
    /// Function call
    Call {
        func: String,
        args: Vec<Operand>,
    },
    
    /// Closure call (first arg is closure value)
    ClosureCall {
        closure: Operand,
        args: Vec<Operand>,
    },
    
    /// Create tuple
    Tuple(Vec<Operand>),
    
    /// Access tuple field
    TupleField(Operand, usize),
    
    /// Create array from elements
    Array(Vec<Operand>),
    
    /// Array indexing
    Index(Operand, Operand),
    
    /// Array slice
    Slice {
        array: Operand,
        start: Option<Operand>,
        end: Option<Operand>,
    },
    
    /// Create closure (captures environment)
    MakeClosure {
        func: String,
        captures: Vec<Operand>,
    },
    
    /// Tensor map: tensor ↦ func
    TensorMap {
        tensor: Operand,
        func: Operand,  // Closure
    },
    
    /// Tensor reduce: tensor Σ
    TensorReduce {
        tensor: Operand,
        op: ReduceOp,
    },
    
    /// Tensor filter: tensor ▸ pred
    TensorFilter {
        tensor: Operand,
        pred: Operand,  // Closure
    },
    
    /// Tensor zip: a ⊗ b
    TensorZip {
        left: Operand,
        right: Operand,
    },
    
    /// Contract check (for debugging, can be stripped)
    ContractCheck {
        predicate: Operand,
        message: String,
        is_precondition: bool,
    },
    
    /// Uncertain value: value ± uncertainty
    Uncertain {
        value: Operand,
        uncertainty: Operand,
    },

    /// Primitive operation (built-in function)
    Prim {
        name: String,
        args: Vec<Operand>,
    },

    /// Iota: generate sequence [0, 1, ..., n-1]
    Iota(Operand),

    /// Range: generate sequence [start, start+1, ..., end-1]
    Range(Operand, Operand),
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
}

/// Block terminator
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return value
    Return(Operand),
    
    /// Unconditional branch
    Goto(BlockId),
    
    /// Conditional branch
    If {
        cond: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    
    /// Multi-way branch (for pattern matching)
    Switch {
        scrutinee: Operand,
        cases: Vec<(Constant, BlockId)>,
        default: BlockId,
    },
    
    /// Unreachable (after exhaustive match)
    Unreachable,
}

/// An operand (value reference)
#[derive(Debug, Clone)]
pub enum Operand {
    /// Local variable
    Local(LocalId),
    /// Constant
    Const(Constant),
}

/// Constant values
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    Unit,
}

/// Local variable ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

/// Block ID  
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl Block {
    pub fn new() -> Self {
        Block {
            stmts: Vec::new(),
            term: Terminator::Unreachable,
        }
    }
    
    pub fn with_return(value: Operand) -> Self {
        Block {
            stmts: Vec::new(),
            term: Terminator::Return(value),
        }
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalId {
    pub fn new(id: u32) -> Self {
        LocalId(id)
    }
}

impl BlockId {
    pub fn new(id: u32) -> Self {
        BlockId(id)
    }
}

impl std::fmt::Display for LocalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.0)
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}
