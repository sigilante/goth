//! CF (Control Flow) dialect operations for Goth MLIR emission
//!
//! The cf dialect provides unstructured control flow:
//! - cf.br: Unconditional branch
//! - cf.cond_br: Conditional branch
//! - cf.switch: Multi-way branch

use crate::context::TextMlirContext;
use crate::error::Result;
use goth_mir::mir::BlockId;

/// Emit an unconditional branch
pub fn emit_br(ctx: &mut TextMlirContext, target: BlockId) -> String {
    format!("{}cf.br ^bb{}\n", ctx.indent_str(), target.0)
}

/// Emit an unconditional branch with block arguments
pub fn emit_br_with_args(
    ctx: &mut TextMlirContext,
    target: BlockId,
    args: &[String],
) -> String {
    if args.is_empty() {
        emit_br(ctx, target)
    } else {
        format!(
            "{}cf.br ^bb{}({})\n",
            ctx.indent_str(),
            target.0,
            args.join(", ")
        )
    }
}

/// Emit a conditional branch
pub fn emit_cond_br(
    ctx: &mut TextMlirContext,
    condition: &str,
    then_block: BlockId,
    else_block: BlockId,
) -> String {
    format!(
        "{}cf.cond_br {}, ^bb{}, ^bb{}\n",
        ctx.indent_str(),
        condition,
        then_block.0,
        else_block.0
    )
}

/// Emit a conditional branch with block arguments
pub fn emit_cond_br_with_args(
    ctx: &mut TextMlirContext,
    condition: &str,
    then_block: BlockId,
    then_args: &[String],
    else_block: BlockId,
    else_args: &[String],
) -> String {
    let then_args_str = if then_args.is_empty() {
        String::new()
    } else {
        format!("({})", then_args.join(", "))
    };

    let else_args_str = if else_args.is_empty() {
        String::new()
    } else {
        format!("({})", else_args.join(", "))
    };

    format!(
        "{}cf.cond_br {}, ^bb{}{}, ^bb{}{}\n",
        ctx.indent_str(),
        condition,
        then_block.0,
        then_args_str,
        else_block.0,
        else_args_str
    )
}

/// Emit a switch (multi-way branch)
pub fn emit_switch(
    ctx: &mut TextMlirContext,
    scrutinee: &str,
    cases: &[(i64, BlockId)],
    default: BlockId,
) -> String {
    let cases_str: Vec<String> = cases.iter()
        .map(|(val, block)| format!("{}: ^bb{}", val, block.0))
        .collect();

    format!(
        "{}cf.switch {} [{}], ^bb{}\n",
        ctx.indent_str(),
        scrutinee,
        cases_str.join(", "),
        default.0
    )
}

/// Emit an assert operation (for debugging/contracts)
pub fn emit_assert(
    ctx: &mut TextMlirContext,
    condition: &str,
    message: &str,
) -> String {
    format!(
        "{}cf.assert {}, \"{}\"\n",
        ctx.indent_str(),
        condition,
        message.escape_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_br() {
        let mut ctx = TextMlirContext::new();
        let code = emit_br(&mut ctx, BlockId::new(1));
        assert_eq!(code, "cf.br ^bb1\n");
    }

    #[test]
    fn test_emit_br_with_args() {
        let mut ctx = TextMlirContext::new();
        let code = emit_br_with_args(&mut ctx, BlockId::new(2), &["%0".into(), "%1".into()]);
        assert_eq!(code, "cf.br ^bb2(%0, %1)\n");
    }

    #[test]
    fn test_emit_cond_br() {
        let mut ctx = TextMlirContext::new();
        let code = emit_cond_br(&mut ctx, "%cond", BlockId::new(1), BlockId::new(2));
        assert_eq!(code, "cf.cond_br %cond, ^bb1, ^bb2\n");
    }

    #[test]
    fn test_emit_switch() {
        let mut ctx = TextMlirContext::new();
        let cases = vec![(0, BlockId::new(1)), (1, BlockId::new(2))];
        let code = emit_switch(&mut ctx, "%x", &cases, BlockId::new(3));
        assert!(code.contains("cf.switch %x"));
        assert!(code.contains("0: ^bb1"));
        assert!(code.contains("1: ^bb2"));
        assert!(code.contains("^bb3"));
    }
}
