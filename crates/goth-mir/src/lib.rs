//! Goth Middle IR (MIR)
//!
//! MIR is an intermediate representation that sits between the typed AST and MLIR.
//! It provides a simplified, explicit form where:
//!
//! - **De Bruijn indices** are resolved to explicit local variables
//! - **Closures** are converted to explicit environment passing
//! - **Polymorphism** is instantiated (all types are concrete)
//! - **Pattern matching** is compiled to decision trees
//!
//! ## Example
//!
//! **Goth source:**
//! ```goth
//! let x = 5 in
//! let y = x + 3 in
//! y * 2
//! ```
//!
//! **MIR:**
//! ```text
//! fn main() -> I64 {
//!   _0: I64 = Const(5)
//!   _1: I64 = BinOp(Add, _0, Const(3))
//!   _2: I64 = BinOp(Mul, _1, Const(2))
//!   Return(_2)
//! }
//! ```

pub mod mir;
pub mod lower;
pub mod closure;
pub mod print;
pub mod error;

pub use mir::*;
pub use error::{MirError, MirResult};

/// Pretty-print a MIR program
pub fn print_program(program: &Program) -> String {
    format!("{}", program)
}

use goth_ast::expr::Expr;
use goth_ast::decl::Module;

/// Lower a typed AST expression to MIR
pub fn lower_expr(expr: &Expr) -> MirResult<Program> {
    lower::lower_expr(expr)
}

/// Lower a typed AST module to MIR
pub fn lower_module(module: &Module) -> MirResult<Program> {
    lower::lower_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::literal::Literal;
    
    #[test]
    fn test_mir_types_compile() {
        // Just verify the types compile and can be constructed
        let _block = Block::new();
        let _local = LocalId::new(0);
        let _block_id = BlockId::new(0);
    }
}
