//! Pretty printing for MIR

use std::fmt;
use crate::mir::*;

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for func in &self.functions {
            writeln!(f, "{}", func)?;
            writeln!(f)?;
        }
        writeln!(f, "// Entry point: {}", self.entry)?;
        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Function signature
        write!(f, "fn {}(", self.name)?;

        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if i == 0 && self.is_closure {
                write!(f, "env: {}", param)?;
            } else {
                write!(f, "arg{}: {}", i, param)?;
            }
        }

        writeln!(f, ") -> {} {{", self.ret_ty)?;

        // Entry block
        writeln!(f, "  entry:")?;
        write!(f, "{}", self.body)?;

        // Additional blocks
        for (block_id, block) in &self.blocks {
            writeln!(f, "  {}:", block_id)?;
            write!(f, "{}", block)?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for stmt in &self.stmts {
            writeln!(f, "  {}", stmt)?;
        }
        writeln!(f, "  {}", self.term)?;
        Ok(())
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} = {}", self.dest, self.ty, self.rhs)
    }
}

impl fmt::Display for Rhs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rhs::Use(op) => write!(f, "{}", op),
            Rhs::Const(c) => write!(f, "Const({})", c),
            Rhs::BinOp(op, left, right) => {
                write!(f, "BinOp({:?}, {}, {})", op, left, right)
            }
            Rhs::UnaryOp(op, operand) => {
                write!(f, "UnaryOp({:?}, {})", op, operand)
            }
            Rhs::Call { func, args } => {
                write!(f, "Call({}",  func)?;
                for arg in args {
                    write!(f, ", {}", arg)?;
                }
                write!(f, ")")
            }
            Rhs::ClosureCall { closure, args } => {
                write!(f, "ClosureCall({}", closure)?;
                for arg in args {
                    write!(f, ", {}", arg)?;
                }
                write!(f, ")")
            }
            Rhs::Tuple(ops) => {
                write!(f, "Tuple(")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op)?;
                }
                write!(f, ")")
            }
            Rhs::TupleField(op, idx) => {
                write!(f, "TupleField({}, {})", op, idx)
            }
            Rhs::Array(ops) => {
                write!(f, "Array(")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op)?;
                }
                write!(f, ")")
            }
            Rhs::Index(arr, idx) => {
                write!(f, "Index({}, {})", arr, idx)
            }
            Rhs::Slice { array, start, end } => {
                write!(f, "Slice({}, {:?}, {:?})", array, start, end)
            }
            Rhs::MakeClosure { func, captures } => {
                write!(f, "MakeClosure({}, [", func)?;
                for (i, cap) in captures.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", cap)?;
                }
                write!(f, "])")
            }
            Rhs::TensorMap { tensor, func } => {
                write!(f, "TensorMap({}, {})", tensor, func)
            }
            Rhs::TensorReduce { tensor, op } => {
                write!(f, "TensorReduce({}, {:?})", tensor, op)
            }
            Rhs::TensorFilter { tensor, pred } => {
                write!(f, "TensorFilter({}, {})", tensor, pred)
            }
            Rhs::TensorZip { left, right } => {
                write!(f, "TensorZip({}, {})", left, right)
            }
            Rhs::ContractCheck { predicate, message, is_precondition } => {
                let kind = if *is_precondition { "pre" } else { "post" };
                write!(f, "ContractCheck({}, {}, \"{}\")", kind, predicate, message)
            }
            Rhs::Uncertain { value, uncertainty } => {
                write!(f, "Uncertain({}, {})", value, uncertainty)
            }
            Rhs::Prim { name, args } => {
                write!(f, "Prim({}",  name)?;
                for arg in args {
                    write!(f, ", {}", arg)?;
                }
                write!(f, ")")
            }
            Rhs::Iota(n) => write!(f, "Iota({})", n),
            Rhs::Range(start, end) => write!(f, "Range({}, {})", start, end),
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Return(op) => write!(f, "Return({})", op),
            Terminator::Goto(block) => write!(f, "Goto({})", block),
            Terminator::If { cond, then_block, else_block } => {
                write!(f, "If({}, {}, {})", cond, then_block, else_block)
            }
            Terminator::Switch { scrutinee, cases, default } => {
                write!(f, "Switch({}, [", scrutinee)?;
                for (i, (c, b)) in cases.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} => {}", c, b)?;
                }
                write!(f, "], {})", default)
            }
            Terminator::Unreachable => write!(f, "Unreachable"),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Local(id) => write!(f, "{}", id),
            Operand::Const(c) => write!(f, "Const({})", c),
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Int(n) => write!(f, "{}", n),
            Constant::Float(x) => write!(f, "{}", x),
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::Unit => write!(f, "()"),
        }
    }
}
