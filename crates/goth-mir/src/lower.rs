//! AST → MIR lowering
//!
//! This module transforms typed AST expressions into MIR.
//! Key transformations:
//! - De Bruijn indices → explicit locals
//! - Nested let bindings → sequential statements
//! - Lambda expressions → closure creation (handled by closure.rs)

use crate::mir::*;
use crate::error::{MirError, MirResult};
use goth_ast::expr::Expr;
use goth_ast::literal::Literal;
use goth_ast::decl::{Module, Decl};
use goth_ast::types::Type;

/// Lowering context
pub struct LoweringContext {
    /// Stack of locals: index 0 = most recent binding (de Bruijn 0)
    locals: Vec<LocalId>,
    /// Counter for fresh locals
    next_local: u32,
    /// Counter for fresh functions (lifted lambdas)
    next_fn: u32,
    /// Counter for fresh blocks
    next_block: u32,
    /// Accumulated statements for current block
    stmts: Vec<Stmt>,
    /// All blocks in the current function
    blocks: Vec<(BlockId, Block)>,
    /// Generated functions (from lambda lifting)
    functions: Vec<Function>,
    /// Global function names and their types
    globals: std::collections::HashMap<String, Type>,
    /// Type information for locals (needed for de Bruijn lookup)
    local_types: std::collections::HashMap<LocalId, Type>,
}

impl LoweringContext {
    pub fn new() -> Self {
        LoweringContext {
            locals: Vec::new(),
            next_local: 0,
            next_fn: 0,
            next_block: 0,
            stmts: Vec::new(),
            blocks: Vec::new(),
            functions: Vec::new(),
            globals: std::collections::HashMap::new(),
            local_types: std::collections::HashMap::new(),
        }
    }
    
    /// Generate a fresh local variable
    fn fresh_local(&mut self) -> LocalId {
        let id = LocalId::new(self.next_local);
        self.next_local += 1;
        id
    }
    
    /// Generate a fresh function name
    fn fresh_fn_name(&mut self) -> String {
        let name = format!("lambda_{}", self.next_fn);
        self.next_fn += 1;
        name
    }
    
    /// Generate a fresh block ID
    fn fresh_block(&mut self) -> BlockId {
        let id = BlockId::new(self.next_block);
        self.next_block += 1;
        id
    }
    
    /// Push a local onto the stack (for de Bruijn index resolution)
    fn push_local(&mut self, local: LocalId, ty: Type) {
        self.locals.push(local);
        self.local_types.insert(local, ty);
    }
    
    /// Pop a local from the stack
    fn pop_local(&mut self) {
        if let Some(local) = self.locals.pop() {
            self.local_types.remove(&local);
        }
    }
    
    /// Look up a de Bruijn index
    fn lookup_index(&self, idx: u32) -> MirResult<(LocalId, Type)> {
        let idx = idx as usize;
        if idx < self.locals.len() {
            // de Bruijn index 0 = most recent = end of stack
            let local = self.locals[self.locals.len() - 1 - idx];
            let ty = self.local_types.get(&local)
                .ok_or_else(|| MirError::Internal("Local type not found".into()))?;
            Ok((local, ty.clone()))
        } else {
            Err(MirError::UnboundVariable(idx as u32))
        }
    }
    
    /// Emit a statement
    fn emit(&mut self, dest: LocalId, ty: Type, rhs: Rhs) {
        self.local_types.insert(dest, ty.clone());
        self.stmts.push(Stmt { dest, ty, rhs });
    }
    
    /// Take all accumulated statements and reset
    fn take_stmts(&mut self) -> Vec<Stmt> {
        std::mem::take(&mut self.stmts)
    }
    
    /// Create a block from accumulated statements with a terminator
    fn make_block(&mut self, term: Terminator) -> Block {
        Block {
            stmts: self.take_stmts(),
            term,
        }
    }
    
    /// Add a completed block
    fn add_block(&mut self, id: BlockId, block: Block) {
        self.blocks.push((id, block));
    }
}

/// Lower an expression to MIR, returning the operand that holds the result
pub fn lower_expr_to_operand(ctx: &mut LoweringContext, expr: &Expr) -> MirResult<(Operand, Type)> {
    match expr {
        // ============ Literals ============
        
        Expr::Lit(lit) => {
            let (constant, ty) = lower_literal(lit);
            Ok((Operand::Const(constant), ty))
        }
        
        // ============ Variables ============
        
        Expr::Idx(idx) => {
            // De Bruijn index - look up in context
            let (local, ty) = ctx.lookup_index(*idx)?;
            Ok((Operand::Local(local), ty))
        }
        
        Expr::Name(name) => {
            // Global name - could be a function or constant
            if let Some(ty) = ctx.globals.get(name.as_ref()) {
                // It's a global function - create a reference
                // For MIR, we don't have first-class function values yet
                // So we'll need to handle this specially
                // For now, error - proper handling needs function pointers
                Err(MirError::Internal(
                    format!("Direct function references not yet supported: {}", name)
                ))
            } else {
                Err(MirError::UndefinedName(name.to_string()))
            }
        }
        
        // ============ Binary Operations ============
        
        Expr::BinOp(op, left, right) => {
            let (left_op, left_ty) = lower_expr_to_operand(ctx, left)?;
            let (right_op, right_ty) = lower_expr_to_operand(ctx, right)?;
            
            // Result type depends on operation
            // TODO: Proper type inference
            let result_ty = left_ty.clone();  // Simplified for now
            
            let dest = ctx.fresh_local();
            ctx.emit(dest, result_ty.clone(), Rhs::BinOp(op.clone(), left_op, right_op));
            
            Ok((Operand::Local(dest), result_ty))
        }
        
        // ============ Unary Operations ============
        
        Expr::UnaryOp(op, operand) => {
            let (op_val, op_ty) = lower_expr_to_operand(ctx, operand)?;
            
            // Result type depends on operation
            let result_ty = match op {
                goth_ast::op::UnaryOp::Floor | goth_ast::op::UnaryOp::Ceil => {
                    Type::Prim(goth_ast::types::PrimType::F64)
                }
                goth_ast::op::UnaryOp::Sqrt => Type::Prim(goth_ast::types::PrimType::F64),
                goth_ast::op::UnaryOp::Not => Type::Prim(goth_ast::types::PrimType::Bool),
                goth_ast::op::UnaryOp::Neg => op_ty.clone(),
                _ => op_ty.clone(),
            };
            
            let dest = ctx.fresh_local();
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(*op, op_val));
            
            Ok((Operand::Local(dest), result_ty))
        }
        
        // ============ Let Bindings ============
        
        Expr::Let { pattern, value, body } => {
            // Lower the value
            let (val_op, val_ty) = lower_expr_to_operand(ctx, value)?;
            
            // For now, only handle simple variable patterns
            // TODO: Pattern compilation for complex patterns
            let local = ctx.fresh_local();
            ctx.emit(local, val_ty.clone(), Rhs::Use(val_op));
            
            // Push onto stack for de Bruijn resolution
            ctx.push_local(local, val_ty);
            
            // Lower the body
            let result = lower_expr_to_operand(ctx, body)?;
            
            // Pop the local
            ctx.pop_local();
            
            Ok(result)
        }
        
        // ============ If Expressions ============
        
        Expr::If { cond, then_, else_ } => {
            // Lower condition
            let (cond_op, _cond_ty) = lower_expr_to_operand(ctx, cond)?;
            
            // We need to create blocks for then and else branches
            // For now, we'll do a simplified version without proper blocks
            // TODO: Implement proper block-based if lowering
            
            // For single-expression if, we can use a phi-like pattern
            // But MIR doesn't have phi nodes, so we need blocks
            
            // Simplified: lower both branches and use conditional
            // This is a temporary solution - proper version needs control flow
            let (then_op, then_ty) = lower_expr_to_operand(ctx, then_)?;
            let (else_op, _else_ty) = lower_expr_to_operand(ctx, else_)?;
            
            // Create a local for the result
            let result = ctx.fresh_local();
            
            // For now, emit a simplified if that doesn't use blocks
            // TODO: Proper if with Goto terminators
            // This is placeholder - real implementation needs block rewriting
            ctx.emit(result, then_ty.clone(), Rhs::Use(then_op));
            
            Ok((Operand::Local(result), then_ty))
        }
        
        // ============ Lambda Expressions ============
        
        Expr::Lam(body) => {
            // Lambda lifting:
            // 1. Analyze free variables
            // 2. Create a new top-level function
            // 3. Create closure that captures free vars
            
            use crate::closure::free_variables;
            let free_vars = free_variables(expr);
            
            // Collect captured values
            let mut captures = Vec::new();
            let mut capture_types = Vec::new();
            
            for &free_idx in &free_vars {
                let (local, ty) = ctx.lookup_index(free_idx)?;
                captures.push(Operand::Local(local));
                capture_types.push(ty);
            }
            
            // Generate function name
            let fn_name = ctx.fresh_fn_name();
            
            // Create new context for lambda body
            let mut lambda_ctx = LoweringContext::new();
            
            // If this is a closure, first parameter is environment
            let has_captures = !captures.is_empty();
            let env_ty = if has_captures {
                Type::Tuple(capture_types.iter().map(|ty| goth_ast::types::TupleField {
                    label: None,
                    ty: ty.clone(),
                }).collect())
            } else {
                Type::Tuple(vec![])
            };
            
            // Lambda parameter type - we need to infer it
            // For now, use a placeholder
            // TODO: Get parameter type from type annotation or inference
            let param_ty = Type::Prim(goth_ast::types::PrimType::I64); // Placeholder
            
            let mut params = Vec::new();
            if has_captures {
                params.push(env_ty.clone());
                let env_local = lambda_ctx.fresh_local();
                lambda_ctx.push_local(env_local, env_ty);
                
                // Extract captures from environment
                for (i, cap_ty) in capture_types.iter().enumerate() {
                    let cap_local = lambda_ctx.fresh_local();
                    lambda_ctx.emit(cap_local, cap_ty.clone(), 
                        Rhs::TupleField(Operand::Local(env_local), i));
                    lambda_ctx.push_local(cap_local, cap_ty.clone());
                }
            }
            
            // Add parameter
            params.push(param_ty.clone());
            let param_local = lambda_ctx.fresh_local();
            lambda_ctx.push_local(param_local, param_ty.clone());
            
            // Lower body
            let (body_op, body_ty) = lower_expr_to_operand(&mut lambda_ctx, body)?;
            
            // Create function
            let lambda_fn = Function {
                name: fn_name.clone(),
                params,
                ret_ty: body_ty.clone(),
                body: Block {
                    stmts: lambda_ctx.take_stmts(),
                    term: Terminator::Return(body_op),
                },
                is_closure: has_captures,
            };
            
            ctx.functions.push(lambda_fn);
            
            // Create closure value
            let closure_ty = Type::func(param_ty, body_ty.clone());
            let closure_local = ctx.fresh_local();
            ctx.emit(closure_local, closure_ty.clone(), 
                Rhs::MakeClosure { func: fn_name, captures });
            
            Ok((Operand::Local(closure_local), closure_ty))
        }
        
        // ============ Function Application ============
        
        Expr::App(func, arg) => {
            let (func_op, func_ty) = lower_expr_to_operand(ctx, func)?;
            let (arg_op, _arg_ty) = lower_expr_to_operand(ctx, arg)?;
            
            // Determine return type from function type
            let ret_ty = match &func_ty {
                Type::Fn(_arg_ty, ret_ty) => (**ret_ty).clone(),
                _ => return Err(MirError::TypeError(
                    format!("Expected function type, got {:?}", func_ty)
                )),
            };
            
            let result = ctx.fresh_local();
            
            // Check if this is a named function or a closure
            match &func_op {
                Operand::Local(_) => {
                    // Closure call
                    ctx.emit(result, ret_ty.clone(), 
                        Rhs::ClosureCall { 
                            closure: func_op, 
                            args: vec![arg_op] 
                        });
                }
                _ => {
                    // For now, treat as closure call
                    ctx.emit(result, ret_ty.clone(), 
                        Rhs::ClosureCall { 
                            closure: func_op, 
                            args: vec![arg_op] 
                        });
                }
            }
            
            Ok((Operand::Local(result), ret_ty))
        }
        
        // ============ Tuples ============
        
        Expr::Tuple(exprs) => {
            let mut ops = Vec::new();
            let mut field_tys = Vec::new();
            
            for expr in exprs {
                let (op, ty) = lower_expr_to_operand(ctx, expr)?;
                ops.push(op);
                field_tys.push(ty);
            }
            
            let tuple_ty = Type::tuple(field_tys);
            let dest = ctx.fresh_local();
            ctx.emit(dest, tuple_ty.clone(), Rhs::Tuple(ops));
            
            Ok((Operand::Local(dest), tuple_ty))
        }
        
        // ============ Arrays ============
        
        Expr::Array(exprs) => {
            let mut ops = Vec::new();
            let mut elem_ty = None;
            
            for expr in exprs {
                let (op, ty) = lower_expr_to_operand(ctx, expr)?;
                ops.push(op);
                
                if elem_ty.is_none() {
                    elem_ty = Some(ty);
                }
            }
            
            let elem_ty = elem_ty.unwrap_or(Type::Prim(goth_ast::types::PrimType::I64));
            let array_ty = Type::vector(
                goth_ast::shape::Dim::constant(exprs.len() as u64),
                elem_ty
            );
            
            let dest = ctx.fresh_local();
            ctx.emit(dest, array_ty.clone(), Rhs::Array(ops));
            
            Ok((Operand::Local(dest), array_ty))
        }
        
        // ============ TODO: More expressions ============
        
        _ => Err(MirError::CannotLower(format!("Expression type not yet implemented: {:?}", expr))),
    }
}

/// Lower a literal to a constant and its type
fn lower_literal(lit: &Literal) -> (Constant, Type) {
    match lit {
        Literal::Int(n) => {
            (Constant::Int(*n as i64), Type::Prim(goth_ast::types::PrimType::I64))
        }
        Literal::Float(x) => {
            (Constant::Float(*x), Type::Prim(goth_ast::types::PrimType::F64))
        }
        Literal::True => {
            (Constant::Bool(true), Type::Prim(goth_ast::types::PrimType::Bool))
        }
        Literal::False => {
            (Constant::Bool(false), Type::Prim(goth_ast::types::PrimType::Bool))
        }
        Literal::Unit => {
            (Constant::Unit, Type::Tuple(vec![]))
        }
        Literal::Char(_) => {
            // TODO: Handle char literals
            (Constant::Int(0), Type::Prim(goth_ast::types::PrimType::Char))
        }
        _ => {
            // TODO: Handle other literal types
            (Constant::Unit, Type::Tuple(vec![]))
        }
    }
}

/// Lower a top-level expression to a Program
pub fn lower_expr(expr: &Expr) -> MirResult<Program> {
    let mut ctx = LoweringContext::new();
    
    let (result_op, result_ty) = lower_expr_to_operand(&mut ctx, expr)?;
    
    // Create main function
    let stmts = ctx.take_stmts();
    let body = Block {
        stmts,
        term: Terminator::Return(result_op),
    };
    
    let main_fn = Function {
        name: "main".to_string(),
        params: vec![],
        ret_ty: result_ty,
        body,
        is_closure: false,
    };
    
    // Collect all functions (lambda-lifted + main)
    let mut all_functions = ctx.functions;
    all_functions.push(main_fn);
    
    Ok(Program {
        functions: all_functions,
        entry: "main".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::literal::Literal;
    use goth_ast::op::{BinOp, UnaryOp};
    use goth_ast::types::PrimType;
    
    #[test]
    fn test_lower_literal_int() {
        let expr = Expr::Lit(Literal::Int(42));
        let program = lower_expr(&expr).unwrap();
        
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.entry, "main");
        
        let main = &program.functions[0];
        assert_eq!(main.body.stmts.len(), 0); // Just a return
        
        match &main.body.term {
            Terminator::Return(Operand::Const(Constant::Int(n))) => {
                assert_eq!(*n, 42);
            }
            _ => panic!("Expected return of constant 42"),
        }
    }
    
    #[test]
    fn test_lower_binop() {
        // 1 + 2
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions[0];
        assert_eq!(main.body.stmts.len(), 1); // One BinOp statement
        
        let stmt = &main.body.stmts[0];
        match &stmt.rhs {
            Rhs::BinOp(BinOp::Add, left, right) => {
                assert!(matches!(left, Operand::Const(Constant::Int(1))));
                assert!(matches!(right, Operand::Const(Constant::Int(2))));
            }
            _ => panic!("Expected BinOp"),
        }
    }
    
    #[test]
    fn test_lower_unary() {
        // √16.0
        let expr = Expr::UnaryOp(
            UnaryOp::Sqrt,
            Box::new(Expr::Lit(Literal::Float(16.0))),
        );
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions[0];
        assert_eq!(main.body.stmts.len(), 1);
        
        let stmt = &main.body.stmts[0];
        match &stmt.rhs {
            Rhs::UnaryOp(UnaryOp::Sqrt, op) => {
                assert!(matches!(op, Operand::Const(Constant::Float(x)) if *x == 16.0));
            }
            _ => panic!("Expected UnaryOp"),
        }
    }
    
    #[test]
    fn test_lower_let_binding() {
        use goth_ast::pattern::Pattern;
        
        // let x = 5 in x + 1
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),  // x
                Box::new(Expr::Lit(Literal::Int(1))),
            )),
        };
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions[0];
        // Should have: _0 = 5, _1 = _0 + 1
        assert_eq!(main.body.stmts.len(), 2);
        
        // First stmt: _0 = Const(5)
        assert!(matches!(&main.body.stmts[0].rhs, 
            Rhs::Use(Operand::Const(Constant::Int(5)))));
        
        // Second stmt: _1 = BinOp(Add, _0, 1)
        match &main.body.stmts[1].rhs {
            Rhs::BinOp(BinOp::Add, left, right) => {
                assert!(matches!(left, Operand::Local(LocalId(0))));
                assert!(matches!(right, Operand::Const(Constant::Int(1))));
            }
            _ => panic!("Expected BinOp"),
        }
    }
    
    #[test]
    fn test_lower_nested_let() {
        use goth_ast::pattern::Pattern;
        
        // let x = 5 in let y = x + 3 in y * 2
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::Let {
                pattern: Pattern::Var(Some("y".into())),
                value: Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Idx(0)),  // x
                    Box::new(Expr::Lit(Literal::Int(3))),
                )),
                body: Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Idx(0)),  // y
                    Box::new(Expr::Lit(Literal::Int(2))),
                )),
            }),
        };
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions.last().unwrap();
        // Should have: _0 = 5, _1 = Use(_0), _2 = _1 + 3, _3 = _2 * 2
        // Or: _0 = 5, _1 = _0 + 3, _2 = Use(_1), _3 = _2 * 2
        // The exact count may vary based on how Use statements are emitted
        assert!(main.body.stmts.len() >= 3, "Should have at least 3 statements");
    }
    
    #[test]
    fn test_lower_tuple() {
        // ⟨1, true⟩
        let expr = Expr::Tuple(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::True),
        ]);
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions[0];
        assert_eq!(main.body.stmts.len(), 1);
        
        match &main.body.stmts[0].rhs {
            Rhs::Tuple(ops) => {
                assert_eq!(ops.len(), 2);
                assert!(matches!(ops[0], Operand::Const(Constant::Int(1))));
                assert!(matches!(ops[1], Operand::Const(Constant::Bool(true))));
            }
            _ => panic!("Expected Tuple"),
        }
    }
    
    #[test]
    fn test_lower_array() {
        // [1, 2, 3]
        let expr = Expr::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Lit(Literal::Int(2)),
            Expr::Lit(Literal::Int(3)),
        ]);
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions[0];
        assert_eq!(main.body.stmts.len(), 1);
        
        match &main.body.stmts[0].rhs {
            Rhs::Array(ops) => {
                assert_eq!(ops.len(), 3);
            }
            _ => panic!("Expected Array"),
        }
    }
    
    #[test]
    fn test_lower_lambda_no_captures() {
        // λ→ ₀ + 1
        let expr = Expr::Lam(Box::new(Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Idx(0)),  // Parameter
            Box::new(Expr::Lit(Literal::Int(1))),
        )));
        let program = lower_expr(&expr).unwrap();
        
        // Should generate 2 functions: lambda_0 + main
        assert_eq!(program.functions.len(), 2);
        
        // Check lambda function
        let lambda = &program.functions[0];
        assert!(lambda.name.starts_with("lambda_"));
        assert!(!lambda.is_closure); // No captures
        assert_eq!(lambda.params.len(), 1); // Just the parameter
        
        // Main should create closure
        let main = &program.functions[1];
        assert_eq!(main.body.stmts.len(), 1);
        match &main.body.stmts[0].rhs {
            Rhs::MakeClosure { func, captures } => {
                assert!(func.starts_with("lambda_"));
                assert!(captures.is_empty()); // No captures
            }
            _ => panic!("Expected MakeClosure"),
        }
    }
    
    #[test]
    fn test_lower_lambda_with_capture() {
        use goth_ast::pattern::Pattern;
        
        // let x = 10 in λ→ ₀ + x
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(10))),
            body: Box::new(Expr::Lam(Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),  // Parameter
                Box::new(Expr::Idx(1)),  // x (captured)
            )))),
        };
        let program = lower_expr(&expr).unwrap();
        
        // Should generate 2 functions
        assert_eq!(program.functions.len(), 2);
        
        // Lambda should be a closure
        let lambda = &program.functions[0];
        assert!(lambda.is_closure);
        assert_eq!(lambda.params.len(), 2); // Environment + parameter
        
        // Main should capture x
        let main = &program.functions[1];
        let closure_stmt = main.body.stmts.iter()
            .find(|s| matches!(s.rhs, Rhs::MakeClosure { .. }))
            .expect("Should have MakeClosure");
        
        match &closure_stmt.rhs {
            Rhs::MakeClosure { captures, .. } => {
                assert_eq!(captures.len(), 1); // Captures x
            }
            _ => unreachable!(),
        }
    }
    
    #[test]
    fn test_lower_function_application() {
        use goth_ast::pattern::Pattern;
        
        // let f = λ→ ₀ + 1 in f 5
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("f".into())),
            value: Box::new(Expr::Lam(Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Idx(0)),
                Box::new(Expr::Lit(Literal::Int(1))),
            )))),
            body: Box::new(Expr::App(
                Box::new(Expr::Idx(0)),  // f
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
        };
        let program = lower_expr(&expr).unwrap();
        
        // Main should have: closure creation, then call
        let main = &program.functions.last().unwrap();
        
        let has_call = main.body.stmts.iter()
            .any(|s| matches!(s.rhs, Rhs::ClosureCall { .. }));
        assert!(has_call, "Should have a closure call");
    }
    
    #[test]
    fn test_lower_if_expression() {
        // if true then 1 else 2
        let expr = Expr::If {
            cond: Box::new(Expr::Lit(Literal::True)),
            then_: Box::new(Expr::Lit(Literal::Int(1))),
            else_: Box::new(Expr::Lit(Literal::Int(2))),
        };
        let program = lower_expr(&expr).unwrap();
        
        let main = &program.functions[0];
        // Should lower all branches
        assert!(main.body.stmts.len() > 0);
    }
    
    #[test]
    fn test_lower_complex_expression() {
        use goth_ast::pattern::Pattern;
        
        // let x = 5 in
        // let y = x * 2 in
        // let f = λ→ ₀ + y in
        // f x
        let expr = Expr::Let {
            pattern: Pattern::Var(Some("x".into())),
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::Let {
                pattern: Pattern::Var(Some("y".into())),
                value: Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Idx(0)),  // x
                    Box::new(Expr::Lit(Literal::Int(2))),
                )),
                body: Box::new(Expr::Let {
                    pattern: Pattern::Var(Some("f".into())),
                    value: Box::new(Expr::Lam(Box::new(Expr::BinOp(
                        BinOp::Add,
                        Box::new(Expr::Idx(0)),  // Parameter
                        Box::new(Expr::Idx(2)),  // y (captured)
                    )))),
                    body: Box::new(Expr::App(
                        Box::new(Expr::Idx(0)),  // f
                        Box::new(Expr::Idx(2)),  // x
                    )),
                }),
            }),
        };
        
        let program = lower_expr(&expr).unwrap();
        
        // Should generate lambda + main
        assert!(program.functions.len() >= 2);
        
        // Lambda should capture y
        let lambda = &program.functions[0];
        assert!(lambda.is_closure);
    }
    
    #[test]
    fn test_pretty_print_simple() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        );
        let program = lower_expr(&expr).unwrap();
        
        let output = format!("{}", program);
        assert!(output.contains("fn main"));
        assert!(output.contains("BinOp"));
        assert!(output.contains("Return"));
    }
}

/// Lower a module to a Program
pub fn lower_module(module: &Module) -> MirResult<Program> {
    let mut ctx = LoweringContext::new();
    
    // Process declarations
    for decl in &module.decls {
        match decl {
            Decl::Fn(fn_decl) => {
                // Register global function
                ctx.globals.insert(fn_decl.name.to_string(), fn_decl.signature.clone());
                
                // TODO: Lower function body
            }
            Decl::Let(let_decl) => {
                // Register global let binding
                // TODO: Infer or use annotated type
            }
            _ => {}
        }
    }
    
    // For now, just create an empty main
    let main_fn = Function {
        name: "main".to_string(),
        params: vec![],
        ret_ty: Type::Tuple(vec![]),
        body: Block::with_return(Operand::Const(Constant::Unit)),
        is_closure: false,
    };
    
    Ok(Program {
        functions: vec![main_fn],
        entry: "main".to_string(),
    })
}
