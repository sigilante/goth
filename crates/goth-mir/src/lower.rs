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
    /// Global constant values (from top-level let bindings)
    global_constants: std::collections::HashMap<String, (Constant, Type)>,
    /// Enum constructors: name -> (tag, has_payload, enum_name)
    enum_constructors: std::collections::HashMap<String, (u32, bool, String)>,
    /// Type information for locals (needed for de Bruijn lookup)
    pub local_types: std::collections::HashMap<LocalId, Type>,
    /// Pending entry block (set by if expressions)
    pending_entry_block: Option<Block>,
    /// Entry block ID for the function (first block created)
    entry_block_id: Option<BlockId>,
    /// Named local variables (from let bindings with named patterns)
    named_locals: std::collections::HashMap<String, (LocalId, Type)>,
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
            global_constants: std::collections::HashMap::new(),
            enum_constructors: std::collections::HashMap::new(),
            local_types: std::collections::HashMap::new(),
            pending_entry_block: None,
            entry_block_id: None,
            named_locals: std::collections::HashMap::new(),
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

/// Check if a name is a known primitive
fn is_primitive(name: &str) -> bool {
    matches!(name,
        // Sequence generation
        "iota" | "ι" | "⍳" |
        "range" | "…" |

        // Array/tensor operations
        "len" | "length" |
        "sum" | "Σ" |
        "prod" | "Π" |
        "map" | "↦" |
        "filter" | "▸" |
        "fold" |
        "reverse" | "⌽" |
        "transpose" | "⍉" |
        "shape" | "ρ" |
        "reshape" |
        "take" | "↑" |
        "drop" | "↓" |
        "concat" | "⧺" |
        "index" |
        "replicate" |
        "zip" |
        "norm" |

        // Math operations
        "dot" | "·" |
        "matmul" |
        "sqrt" | "√" |
        "abs" |
        "floor" | "ceil" | "round" |
        "sin" | "cos" | "tan" |
        "asin" | "acos" | "atan" |
        "sinh" | "cosh" | "tanh" |
        "exp" | "ln" | "log" | "log10" | "log2" |
        "gamma" | "Γ" |
        "sign" |

        // Type conversions
        "toString" | "str" |
        "toInt" | "toFloat" | "toBool" | "toChar" |
        "parseInt" | "parseFloat" |

        // String operations
        "chars" |
        "strConcat" |
        "strLen" |
        "lines" |
        "words" |
        "bytes" |
        "strEq" |
        "startsWith" |
        "endsWith" |
        "contains" |
        "joinStrings" |

        // I/O operations
        "print" |
        "write" |
        "flush" |
        "readLine" |
        "readKey" |
        "rawModeEnter" |
        "rawModeExit" |
        "sleep" |
        "readFile" |
        "writeFile"
    )
}

/// Get the name of a primitive from an expression, if it is one
fn get_primitive_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Name(name) if is_primitive(name) => Some(name),
        Expr::Prim(name) => Some(name),
        _ => None,
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
            // First check for locally bound names (from let bindings)
            if let Some((local, ty)) = ctx.named_locals.get(name.as_ref()) {
                return Ok((Operand::Local(*local), ty.clone()));
            }

            // Check if it's a primitive
            if is_primitive(name) {
                // Primitives are handled at application site
                // Return a marker type for now
                let prim_ty = Type::Prim(goth_ast::types::PrimType::I64); // Placeholder
                Ok((Operand::Const(Constant::Int(0)), prim_ty)) // Will be replaced at App
            } else if let Some((constant, ty)) = ctx.global_constants.get(name.as_ref()) {
                // It's a global constant - inline the value
                Ok((Operand::Const(constant.clone()), ty.clone()))
            } else if let Some((tag, has_payload, _enum_name)) = ctx.enum_constructors.get(name.as_ref()).cloned() {
                // It's an enum constructor
                if has_payload {
                    // Constructor with payload - needs to be handled at application site
                    // Return a marker that indicates pending constructor application
                    let ty = Type::Prim(goth_ast::types::PrimType::I64); // Variant pointer
                    Ok((Operand::Const(Constant::Int(tag as i64)), ty))
                } else {
                    // Nullary constructor - emit MakeVariant directly
                    let result = ctx.fresh_local();
                    let ty = Type::Prim(goth_ast::types::PrimType::I64); // Variant pointer
                    ctx.emit(result, ty.clone(),
                        Rhs::MakeVariant {
                            tag,
                            constructor: name.to_string(),
                            payload: None,
                        });
                    Ok((Operand::Local(result), ty))
                }
            } else if let Some(_ty) = ctx.globals.get(name.as_ref()) {
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

        Expr::Prim(name) => {
            // Primitive reference - handled at application site
            let prim_ty = Type::Prim(goth_ast::types::PrimType::I64); // Placeholder
            Ok((Operand::Const(Constant::Int(0)), prim_ty))
        }
        
        // ============ Binary Operations ============
        
        Expr::BinOp(op, left, right) => {
            let (left_op, left_ty) = lower_expr_to_operand(ctx, left)?;
            let (right_op, _right_ty) = lower_expr_to_operand(ctx, right)?;

            // Result type depends on operation category
            let result_ty = match op {
                // Comparison operators always return Bool
                goth_ast::op::BinOp::Eq
                | goth_ast::op::BinOp::Neq
                | goth_ast::op::BinOp::Lt
                | goth_ast::op::BinOp::Gt
                | goth_ast::op::BinOp::Leq
                | goth_ast::op::BinOp::Geq => {
                    Type::Prim(goth_ast::types::PrimType::Bool)
                }
                // Logical operators always return Bool
                goth_ast::op::BinOp::And
                | goth_ast::op::BinOp::Or => {
                    Type::Prim(goth_ast::types::PrimType::Bool)
                }
                // Arithmetic and other operators preserve operand type
                _ => left_ty.clone(),
            };

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
                goth_ast::op::UnaryOp::Sqrt
                | goth_ast::op::UnaryOp::Gamma
                | goth_ast::op::UnaryOp::Ln
                | goth_ast::op::UnaryOp::Exp
                | goth_ast::op::UnaryOp::Sin
                | goth_ast::op::UnaryOp::Cos
                | goth_ast::op::UnaryOp::Abs => Type::Prim(goth_ast::types::PrimType::F64),
                goth_ast::op::UnaryOp::Not => Type::Prim(goth_ast::types::PrimType::Bool),
                goth_ast::op::UnaryOp::Neg => op_ty.clone(),
                // Reduce operations (Sum, Prod) return element type, not array type
                goth_ast::op::UnaryOp::Sum | goth_ast::op::UnaryOp::Prod => {
                    // Extract element type from tensor type
                    match &op_ty {
                        Type::Tensor(_, elem_ty) => (**elem_ty).clone(),
                        _ => op_ty.clone(), // Fallback to same type for scalars
                    }
                }
                _ => op_ty.clone(),
            };

            let dest = ctx.fresh_local();
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(*op, op_val));

            Ok((Operand::Local(dest), result_ty))
        }
        
        // ============ Let Bindings ============
        
        Expr::Let { pattern, value, body, type_: _ } => {
            use goth_ast::pattern::Pattern;

            // Lower the value
            let (val_op, val_ty) = lower_expr_to_operand(ctx, value)?;

            // For now, only handle simple variable patterns
            // TODO: Pattern compilation for complex patterns
            let local = ctx.fresh_local();
            ctx.emit(local, val_ty.clone(), Rhs::Use(val_op));

            // Extract name from pattern if it's a named Var
            let binding_name = match pattern {
                Pattern::Var(Some(name)) => Some(name.to_string()),
                _ => None,
            };

            // Store named binding if present
            if let Some(ref name) = binding_name {
                ctx.named_locals.insert(name.clone(), (local, val_ty.clone()));
            }

            // Push onto stack for de Bruijn resolution
            ctx.push_local(local, val_ty);

            // Lower the body
            let result = lower_expr_to_operand(ctx, body)?;

            // Pop the local and remove named binding
            ctx.pop_local();
            if let Some(name) = binding_name {
                ctx.named_locals.remove(&name);
            }

            Ok(result)
        }
        
        // ============ If Expressions ============

        Expr::If { cond, then_, else_ } => {
            // Lower condition in current block
            let (cond_op, _cond_ty) = lower_expr_to_operand(ctx, cond)?;

            // Create block IDs
            let cond_block_id = ctx.fresh_block();
            let then_block_id = ctx.fresh_block();
            let else_block_id = ctx.fresh_block();
            let join_block_id = ctx.fresh_block();

            // Result variable (will be written in both branches)
            let result = ctx.fresh_local();

            // Build the condition block from accumulated statements
            let current_stmts = ctx.take_stmts();
            let cond_block = Block {
                stmts: current_stmts,
                term: Terminator::If {
                    cond: cond_op,
                    then_block: then_block_id,
                    else_block: else_block_id,
                },
            };
            ctx.add_block(cond_block_id, cond_block);

            // Track the entry block (first condition block we create)
            if ctx.entry_block_id.is_none() {
                ctx.entry_block_id = Some(cond_block_id);
            }

            // === Lower THEN branch ===
            let blocks_before_then = ctx.blocks.len();
            let (then_op, then_ty) = lower_expr_to_operand(ctx, then_)?;
            let then_stmts = ctx.take_stmts();
            let then_created_cf = ctx.blocks.len() > blocks_before_then;

            if then_created_cf {
                // Then branch has nested control flow - create trampoline
                let nested_entry = ctx.blocks[blocks_before_then].0;
                ctx.add_block(then_block_id, Block {
                    stmts: then_stmts,
                    term: Terminator::Goto(nested_entry),
                });
                // Patch all nested join blocks (Unreachable) to flow to our join
                for i in blocks_before_then..ctx.blocks.len() {
                    if matches!(ctx.blocks[i].1.term, Terminator::Unreachable) {
                        ctx.blocks[i].1.stmts.push(Stmt {
                            dest: result,
                            ty: then_ty.clone(),
                            rhs: Rhs::Use(then_op.clone()),
                        });
                        ctx.blocks[i].1.term = Terminator::Goto(join_block_id);
                    }
                }
            } else {
                // Simple then branch
                let mut then_block_stmts = then_stmts;
                then_block_stmts.push(Stmt {
                    dest: result,
                    ty: then_ty.clone(),
                    rhs: Rhs::Use(then_op),
                });
                ctx.add_block(then_block_id, Block {
                    stmts: then_block_stmts,
                    term: Terminator::Goto(join_block_id),
                });
            }

            // === Lower ELSE branch ===
            let blocks_before_else = ctx.blocks.len();
            let (else_op, _else_ty) = lower_expr_to_operand(ctx, else_)?;
            let else_stmts = ctx.take_stmts();
            let else_created_cf = ctx.blocks.len() > blocks_before_else;

            if else_created_cf {
                // Else branch has nested control flow - create trampoline
                let nested_entry = ctx.blocks[blocks_before_else].0;
                ctx.add_block(else_block_id, Block {
                    stmts: else_stmts,
                    term: Terminator::Goto(nested_entry),
                });
                // Patch all nested join blocks (Unreachable) to flow to our join
                for i in blocks_before_else..ctx.blocks.len() {
                    if matches!(ctx.blocks[i].1.term, Terminator::Unreachable) {
                        ctx.blocks[i].1.stmts.push(Stmt {
                            dest: result,
                            ty: then_ty.clone(),
                            rhs: Rhs::Use(else_op.clone()),
                        });
                        ctx.blocks[i].1.term = Terminator::Goto(join_block_id);
                    }
                }
            } else {
                // Simple else branch
                let mut else_block_stmts = else_stmts;
                else_block_stmts.push(Stmt {
                    dest: result,
                    ty: then_ty.clone(),
                    rhs: Rhs::Use(else_op),
                });
                ctx.add_block(else_block_id, Block {
                    stmts: else_block_stmts,
                    term: Terminator::Goto(join_block_id),
                });
            }

            // Join block continues (terminator set by caller or at end)
            ctx.add_block(join_block_id, Block {
                stmts: vec![],
                term: Terminator::Unreachable, // Will be replaced by parent or final pass
            });

            // Register result type
            ctx.local_types.insert(result, then_ty.clone());

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
                blocks: lambda_ctx.blocks.clone(),
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
            // Check if this is a primitive application
            if let Some(prim_name) = get_primitive_name(func) {
                return lower_primitive_app(ctx, prim_name, arg);
            }

            // Check for curried primitive application (e.g., range start end)
            if let Expr::App(inner_func, first_arg) = func.as_ref() {
                if let Some(prim_name) = get_primitive_name(inner_func) {
                    return lower_primitive_app2(ctx, prim_name, first_arg, arg);
                }
            }

            // Check if this is an enum constructor with payload (e.g., Some 42)
            if let Expr::Name(name) = func.as_ref() {
                if let Some((tag, has_payload, _enum_name)) = ctx.enum_constructors.get(name.as_ref()).cloned() {
                    if has_payload {
                        let (payload_op, _payload_ty) = lower_expr_to_operand(ctx, arg)?;
                        let result = ctx.fresh_local();
                        let ty = Type::Prim(goth_ast::types::PrimType::I64); // Variant pointer
                        ctx.emit(result, ty.clone(),
                            Rhs::MakeVariant {
                                tag,
                                constructor: name.to_string(),
                                payload: Some(payload_op),
                            });
                        return Ok((Operand::Local(result), ty));
                    }
                }
            }

            // Check if this is a direct call to a global function
            if let Expr::Name(name) = func.as_ref() {
                if let Some(func_ty) = ctx.globals.get(name.as_ref()).cloned() {
                    // Direct function call
                    let (arg_op, arg_ty) = lower_expr_to_operand(ctx, arg)?;

                    // Get return type from function signature
                    let ret_ty = match &func_ty {
                        Type::Fn(_arg_ty, ret_ty) => (**ret_ty).clone(),
                        _ => return Err(MirError::TypeError(
                            format!("Expected function type for '{}', got {:?}", name, func_ty)
                        )),
                    };

                    let result = ctx.fresh_local();
                    ctx.emit(result, ret_ty.clone(),
                        Rhs::Call {
                            func: name.to_string(),
                            args: vec![arg_op],
                            arg_tys: vec![arg_ty],
                        });

                    return Ok((Operand::Local(result), ret_ty));
                }
            }

            // Check for curried global function call (e.g., add 1 2)
            if let Expr::App(inner_func, first_arg) = func.as_ref() {
                if let Expr::Name(name) = inner_func.as_ref() {
                    if let Some(func_ty) = ctx.globals.get(name.as_ref()).cloned() {
                        // Curried function call with 2 args
                        let (arg1_op, arg1_ty) = lower_expr_to_operand(ctx, first_arg)?;
                        let (arg2_op, arg2_ty) = lower_expr_to_operand(ctx, arg)?;

                        // Get return type (need to unwrap two function arrows)
                        let ret_ty = match &func_ty {
                            Type::Fn(_, ret1) => match ret1.as_ref() {
                                Type::Fn(_, ret2) => (**ret2).clone(),
                                other => other.clone(),
                            },
                            _ => return Err(MirError::TypeError(
                                format!("Expected function type for '{}', got {:?}", name, func_ty)
                            )),
                        };

                        let result = ctx.fresh_local();
                        ctx.emit(result, ret_ty.clone(),
                            Rhs::Call {
                                func: name.to_string(),
                                args: vec![arg1_op, arg2_op],
                                arg_tys: vec![arg1_ty, arg2_ty],
                            });

                        return Ok((Operand::Local(result), ret_ty));
                    }
                }
            }

            let (func_op, func_ty) = lower_expr_to_operand(ctx, func)?;
            let (arg_op, arg_ty) = lower_expr_to_operand(ctx, arg)?;

            // Determine return type from function type
            let ret_ty = match &func_ty {
                Type::Fn(_arg_ty, ret_ty) => (**ret_ty).clone(),
                // Handle polymorphic type variables - assume they represent functions
                // and use the type variable as return type (will be refined later)
                Type::Var(name) => Type::Var(name.clone()),
                // Handle effectful function types
                Type::Effectful(inner, _effects) => {
                    match inner.as_ref() {
                        Type::Fn(_arg_ty, ret_ty) => (**ret_ty).clone(),
                        Type::Var(name) => Type::Var(name.clone()),
                        _ => arg_ty.clone(), // Best effort: use argument type
                    }
                }
                // For Forall types, try to extract the inner function type
                Type::Forall(_params, inner) => {
                    match inner.as_ref() {
                        Type::Fn(_arg_ty, ret_ty) => (**ret_ty).clone(),
                        _ => arg_ty.clone(),
                    }
                }
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

        // ============ Array Fill ============

        Expr::ArrayFill { shape, value } => {
            // ArrayFill creates an array of given size filled with a value
            // For now, only support 1D arrays (single dimension)
            if shape.len() != 1 {
                return Err(MirError::CannotLower(
                    "Multi-dimensional ArrayFill not yet supported".to_string()
                ));
            }

            let (size_op, _size_ty) = lower_expr_to_operand(ctx, &shape[0])?;
            let (value_op, value_ty) = lower_expr_to_operand(ctx, value)?;

            // Result type is an array of the value type
            let array_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()), // Dynamic size
                value_ty.clone()
            );

            let dest = ctx.fresh_local();
            ctx.emit(dest, array_ty.clone(), Rhs::ArrayFill {
                size: size_op,
                value: value_op,
            });

            Ok((Operand::Local(dest), array_ty))
        }

        // ============ Field Access ============

        Expr::Field(base, access) => {
            let (base_op, base_ty) = lower_expr_to_operand(ctx, base)?;

            // Determine result type from base type
            let (field_idx, result_ty) = match access {
                goth_ast::expr::FieldAccess::Index(idx) => {
                    // Numeric index into tuple
                    let elem_ty = match &base_ty {
                        Type::Tuple(fields) => {
                            fields.get(*idx as usize)
                                .map(|f| f.ty.clone())
                                .unwrap_or(Type::Prim(goth_ast::types::PrimType::I64))
                        }
                        _ => Type::Prim(goth_ast::types::PrimType::I64),
                    };
                    (*idx as usize, elem_ty)
                }
                goth_ast::expr::FieldAccess::Named(name) => {
                    // Named field - find index in tuple
                    let (idx, elem_ty) = match &base_ty {
                        Type::Tuple(fields) => {
                            fields.iter().enumerate()
                                .find(|(_, f)| f.label.as_ref().map(|l| l.as_ref()) == Some(name.as_ref()))
                                .map(|(i, f)| (i, f.ty.clone()))
                                .unwrap_or((0, Type::Prim(goth_ast::types::PrimType::I64)))
                        }
                        _ => (0, Type::Prim(goth_ast::types::PrimType::I64)),
                    };
                    (idx, elem_ty)
                }
            };

            let dest = ctx.fresh_local();
            ctx.emit(dest, result_ty.clone(), Rhs::TupleField(base_op, field_idx));

            Ok((Operand::Local(dest), result_ty))
        }

        // ============ Array Indexing ============

        Expr::Index(arr, indices) => {
            let (arr_op, arr_ty) = lower_expr_to_operand(ctx, arr)?;

            // For now, only handle single index
            if indices.len() != 1 {
                return Err(MirError::CannotLower(
                    "Multi-dimensional indexing not yet supported".to_string()
                ));
            }

            let (idx_op, _) = lower_expr_to_operand(ctx, &indices[0])?;

            // Result type is element type of tensor
            let elem_ty = match &arr_ty {
                Type::Tensor(_, elem) => (**elem).clone(),
                _ => Type::Prim(goth_ast::types::PrimType::I64),
            };

            let dest = ctx.fresh_local();
            ctx.emit(dest, elem_ty.clone(), Rhs::Index(arr_op, idx_op));

            Ok((Operand::Local(dest), elem_ty))
        }

        // ============ Variants (Sum Types) ============

        Expr::Variant { constructor, payload } => {
            // Lower the payload if present
            let (payload_op, payload_ty) = if let Some(p) = payload {
                let (op, ty) = lower_expr_to_operand(ctx, p)?;
                (Some(op), Some(ty))
            } else {
                (None, None)
            };

            // Tag index is computed from constructor name hash for now
            // In a full implementation, this would be looked up from enum definition
            let tag = constructor_tag(constructor);

            // Build variant type - single arm for now (would need full enum info for complete type)
            let variant_ty = Type::Variant(vec![
                goth_ast::types::VariantArm {
                    name: constructor.clone(),
                    payload: payload_ty,
                }
            ]);

            let dest = ctx.fresh_local();
            ctx.emit(dest, variant_ty.clone(), Rhs::MakeVariant {
                tag,
                constructor: constructor.to_string(),
                payload: payload_op,
            });

            Ok((Operand::Local(dest), variant_ty))
        }

        // ============ Match Expressions ============

        Expr::Match { scrutinee, arms } => {
            lower_match_expr(ctx, scrutinee, arms)
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
        Literal::Char(c) => {
            // Char as 32-bit unicode scalar
            (Constant::Int(*c as i64), Type::Prim(goth_ast::types::PrimType::Char))
        }
        Literal::String(s) => {
            (Constant::String(s.to_string()), Type::Prim(goth_ast::types::PrimType::String))
        }
        _ => {
            // TODO: Handle other literal types (Array, Tensor)
            (Constant::Unit, Type::Tuple(vec![]))
        }
    }
}

/// Compute a tag index from a constructor name
/// In a full implementation, this would be looked up from enum definitions
fn constructor_tag(name: &str) -> u32 {
    // Use simple hash-based assignment for now
    // Common constructors get predictable tags
    match name {
        // Option
        "None" => 0,
        "Some" => 1,
        // Either
        "Left" => 0,
        "Right" => 1,
        // Bool-like
        "False" => 0,
        "True" => 1,
        // List
        "Nil" => 0,
        "Cons" => 1,
        // Result
        "Ok" => 0,
        "Err" => 1,
        // Default: hash the name
        _ => {
            let mut hash: u32 = 0;
            for c in name.chars() {
                hash = hash.wrapping_mul(31).wrapping_add(c as u32);
            }
            hash % 256 // Keep tags small
        }
    }
}

/// Lower a match expression to MIR with control flow
fn lower_match_expr(
    ctx: &mut LoweringContext,
    scrutinee: &Expr,
    arms: &[goth_ast::expr::MatchArm],
) -> MirResult<(Operand, Type)> {
    use goth_ast::pattern::Pattern;

    // Lower the scrutinee
    let (scrut_op, scrut_ty) = lower_expr_to_operand(ctx, scrutinee)?;

    // Create result variable
    let result = ctx.fresh_local();

    // Create join block for after the match
    let join_block_id = ctx.fresh_block();

    // Check if this is a variant match (has Pattern::Variant arms)
    let is_variant_match = arms.iter().any(|arm| matches!(arm.pattern, Pattern::Variant { .. }));

    if is_variant_match {
        // Get the tag of the scrutinee
        let tag_local = ctx.fresh_local();
        ctx.emit(tag_local, Type::Prim(goth_ast::types::PrimType::I64), Rhs::GetTag(scrut_op.clone()));

        // Build switch cases for each variant arm
        let mut cases = Vec::new();
        let mut arm_blocks = Vec::new();

        for arm in arms {
            let arm_block_id = ctx.fresh_block();
            arm_blocks.push((arm_block_id, arm));

            if let Pattern::Variant { constructor, payload: _ } = &arm.pattern {
                // Look up tag from registered enum constructors, fall back to heuristic
                let tag = if let Some((tag, _, _)) = ctx.enum_constructors.get(constructor.as_ref()) {
                    *tag
                } else {
                    constructor_tag(constructor)
                };
                cases.push((Constant::Int(tag as i64), arm_block_id));
            }
        }

        // Default block (unreachable for exhaustive matches)
        let default_block_id = ctx.fresh_block();

        // Save current statements and create the switch terminator
        let current_stmts = ctx.take_stmts();
        let switch_block = Block {
            stmts: current_stmts,
            term: Terminator::Switch {
                scrutinee: Operand::Local(tag_local),
                cases,
                default: default_block_id,
            },
        };
        ctx.pending_entry_block = Some(switch_block);

        // Lower each arm
        let mut result_ty = None;
        for (arm_block_id, arm) in arm_blocks {
            // Extract payload if pattern has one
            if let Pattern::Variant { constructor: _, payload } = &arm.pattern {
                if let Some(payload_pattern) = payload {
                    // Get the payload from the scrutinee
                    let payload_local = ctx.fresh_local();
                    let payload_ty = Type::Prim(goth_ast::types::PrimType::I64); // Placeholder type
                    ctx.emit(payload_local, payload_ty.clone(), Rhs::GetPayload(scrut_op.clone()));

                    // Bind the payload to a local for the arm body
                    if let Pattern::Var(_) = payload_pattern.as_ref() {
                        ctx.push_local(payload_local, payload_ty);
                    }
                }
            }

            // Lower the arm body
            let (arm_op, arm_ty) = lower_expr_to_operand(ctx, &arm.body)?;

            if result_ty.is_none() {
                result_ty = Some(arm_ty.clone());
            }

            // Pop any bound locals
            if let Pattern::Variant { payload: Some(_), .. } = &arm.pattern {
                ctx.pop_local();
            }

            // Create arm block
            let arm_stmts = ctx.take_stmts();
            let mut block_stmts = arm_stmts;
            block_stmts.push(Stmt {
                dest: result,
                ty: arm_ty,
                rhs: Rhs::Use(arm_op),
            });
            ctx.add_block(arm_block_id, Block {
                stmts: block_stmts,
                term: Terminator::Goto(join_block_id),
            });
        }

        // Default block (unreachable)
        ctx.add_block(default_block_id, Block {
            stmts: vec![],
            term: Terminator::Unreachable,
        });

        // Join block
        ctx.add_block(join_block_id, Block {
            stmts: vec![],
            term: Terminator::Unreachable, // Will be replaced by caller
        });

        let result_ty = result_ty.unwrap_or(Type::Tuple(vec![]));
        ctx.local_types.insert(result, result_ty.clone());

        Ok((Operand::Local(result), result_ty))
    } else {
        // Non-variant match (e.g., literal patterns) - use if-else chain
        //
        // Strategy: Generate nested if-else for each literal pattern,
        // with variable/wildcard patterns as the default case.
        //
        // match x {
        //   0 → a;
        //   1 → b;
        //   n → c;   // variable pattern becomes default
        // }
        //
        // Becomes:
        //   if x == 0 then a
        //   else if x == 1 then b
        //   else c  (with n bound to x)

        // Separate literal patterns from default (var/wildcard) patterns
        let mut literal_arms: Vec<(&goth_ast::expr::MatchArm, Constant)> = Vec::new();
        let mut default_arm: Option<&goth_ast::expr::MatchArm> = None;
        let mut var_binding: Option<Option<Box<str>>> = None;

        for arm in arms {
            match &arm.pattern {
                Pattern::Lit(lit) => {
                    let (constant, _) = lower_literal(lit);
                    literal_arms.push((arm, constant));
                }
                Pattern::Var(name) => {
                    // Variable pattern matches anything and binds the value
                    if default_arm.is_none() {
                        default_arm = Some(arm);
                        var_binding = Some(name.clone());
                    }
                }
                Pattern::Wildcard => {
                    // Wildcard matches anything but doesn't bind
                    if default_arm.is_none() {
                        default_arm = Some(arm);
                    }
                }
                _ => {
                    // Other patterns (tuple, array, etc.) not yet supported
                    return Err(MirError::CannotLower(
                        format!("Pattern type {:?} not yet supported in non-variant match", arm.pattern)
                    ));
                }
            }
        }

        // Need at least a default arm for exhaustiveness
        let default_arm = match default_arm {
            Some(arm) => arm,
            None => {
                // If no explicit default, use the last arm if it exists
                // (This handles matches that are exhaustive by literal coverage)
                // For now, we require an explicit default
                return Err(MirError::CannotLower(
                    "Non-variant match requires a default (wildcard or variable) pattern".to_string()
                ));
            }
        };

        // If no literal arms, just lower the default
        if literal_arms.is_empty() {
            // Bind variable if needed
            if let Some(name) = &var_binding {
                if name.is_some() {
                    // Create a local for the bound variable
                    let bound_local = ctx.fresh_local();
                    ctx.emit(bound_local, scrut_ty.clone(), Rhs::Use(scrut_op.clone()));
                    ctx.push_local(bound_local, scrut_ty.clone());
                }
            }

            let (body_op, body_ty) = lower_expr_to_operand(ctx, &default_arm.body)?;

            // Pop binding if we pushed one
            if let Some(Some(_)) = &var_binding {
                ctx.pop_local();
            }

            return Ok((body_op, body_ty));
        }

        // Generate if-else chain for literal patterns
        // We'll build this from the inside out (last pattern first)

        // First, lower the default arm body
        // Store the scrutinee for potential variable binding
        let scrut_local = match &scrut_op {
            Operand::Local(id) => *id,
            Operand::Const(_) => {
                // Store constant in a local for comparison
                let local = ctx.fresh_local();
                ctx.emit(local, scrut_ty.clone(), Rhs::Use(scrut_op.clone()));
                local
            }
        };

        // Create all the block IDs we need upfront
        let mut check_block_ids: Vec<BlockId> = Vec::new();
        let mut arm_block_ids: Vec<BlockId> = Vec::new();

        for _ in 0..literal_arms.len() {
            check_block_ids.push(ctx.fresh_block());
            arm_block_ids.push(ctx.fresh_block());
        }
        let default_block_id = ctx.fresh_block();
        let join_block_id = ctx.fresh_block();

        // Save current statements - these go in the entry/first check block
        let entry_stmts = ctx.take_stmts();

        // Lower each literal arm body and create arm blocks
        let mut result_ty: Option<Type> = None;

        for (i, (arm, _)) in literal_arms.iter().enumerate() {
            // Lower the arm body
            let (arm_op, arm_ty) = lower_expr_to_operand(ctx, &arm.body)?;

            if result_ty.is_none() {
                result_ty = Some(arm_ty.clone());
            }

            let arm_stmts = ctx.take_stmts();
            let mut block_stmts = arm_stmts;
            block_stmts.push(Stmt {
                dest: result,
                ty: arm_ty,
                rhs: Rhs::Use(arm_op),
            });

            ctx.add_block(arm_block_ids[i], Block {
                stmts: block_stmts,
                term: Terminator::Goto(join_block_id),
            });
        }

        // Lower the default arm body
        if let Some(Some(_)) = &var_binding {
            // Bind the scrutinee to a local variable
            ctx.push_local(scrut_local, scrut_ty.clone());
        }

        let (default_op, default_ty) = lower_expr_to_operand(ctx, &default_arm.body)?;

        if let Some(Some(_)) = &var_binding {
            ctx.pop_local();
        }

        if result_ty.is_none() {
            result_ty = Some(default_ty.clone());
        }

        let default_stmts = ctx.take_stmts();
        let mut default_block_stmts = default_stmts;
        default_block_stmts.push(Stmt {
            dest: result,
            ty: default_ty,
            rhs: Rhs::Use(default_op),
        });

        ctx.add_block(default_block_id, Block {
            stmts: default_block_stmts,
            term: Terminator::Goto(join_block_id),
        });

        // Now create the check blocks with comparisons
        // Each check block: compare scrutinee with literal, branch to arm or next check
        for (i, (_, constant)) in literal_arms.iter().enumerate() {
            let cmp_local = ctx.fresh_local();
            let cmp_stmt = Stmt {
                dest: cmp_local,
                ty: Type::Prim(goth_ast::types::PrimType::Bool),
                rhs: Rhs::BinOp(
                    goth_ast::op::BinOp::Eq,
                    Operand::Local(scrut_local),
                    Operand::Const(constant.clone()),
                ),
            };

            let next_block = if i + 1 < literal_arms.len() {
                check_block_ids[i + 1]
            } else {
                default_block_id
            };

            let block_stmts = if i == 0 {
                // First check block gets the entry statements
                let mut stmts = entry_stmts.clone();
                stmts.push(cmp_stmt);
                stmts
            } else {
                vec![cmp_stmt]
            };

            ctx.add_block(check_block_ids[i], Block {
                stmts: block_stmts,
                term: Terminator::If {
                    cond: Operand::Local(cmp_local),
                    then_block: arm_block_ids[i],
                    else_block: next_block,
                },
            });
        }

        // Join block
        ctx.add_block(join_block_id, Block {
            stmts: vec![],
            term: Terminator::Unreachable, // Will be replaced by caller
        });

        // Set the first check block as the pending entry
        ctx.pending_entry_block = Some(Block {
            stmts: vec![],
            term: Terminator::Goto(check_block_ids[0]),
        });

        let result_ty = result_ty.unwrap_or(Type::Tuple(vec![]));
        ctx.local_types.insert(result, result_ty.clone());

        Ok((Operand::Local(result), result_ty))
    }
}

/// Lower a single-argument primitive application
fn lower_primitive_app(ctx: &mut LoweringContext, prim_name: &str, arg: &Expr) -> MirResult<(Operand, Type)> {
    let (arg_op, arg_ty) = lower_expr_to_operand(ctx, arg)?;
    let dest = ctx.fresh_local();

    match prim_name {
        // Sequence generation
        "iota" | "ι" | "⍳" => {
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::Prim(goth_ast::types::PrimType::I64)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Iota(arg_op));
            Ok((Operand::Local(dest), result_ty))
        }

        // Aggregations
        "sum" | "Σ" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::TensorReduce {
                tensor: arg_op,
                op: ReduceOp::Sum,
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "prod" | "Π" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::TensorReduce {
                tensor: arg_op,
                op: ReduceOp::Prod,
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Tensor operations
        "len" | "length" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "len".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "reverse" | "⌽" => {
            let result_ty = arg_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "reverse".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "transpose" | "⍉" => {
            let result_ty = arg_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "transpose".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "shape" | "ρ" => {
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::Prim(goth_ast::types::PrimType::I64)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "shape".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Math functions
        "sqrt" | "√" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Sqrt,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "abs" => {
            let result_ty = arg_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "abs".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "floor" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Floor,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "ceil" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Ceil,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }

        // I/O
        "print" => {
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "print".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Conversion
        "toString" | "str" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::Char); // String type
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "toString".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "toInt" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "toInt".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "toFloat" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "toFloat".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "toBool" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::Bool);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "toBool".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "toChar" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::Char);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "toChar".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "parseInt" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "parseInt".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "parseFloat" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "parseFloat".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // String operations
        "chars" => {
            // String → [n]Char (returns the string as char tensor)
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "chars".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "strLen" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "strLen".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "lines" => {
            // String → [m]String (split by newlines)
            let string_ty = Type::vector(
                goth_ast::shape::Dim::Var("k".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("m".into()),
                string_ty
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "lines".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "words" => {
            // String → [m]String (split by whitespace)
            let string_ty = Type::vector(
                goth_ast::shape::Dim::Var("k".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("m".into()),
                string_ty
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "words".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "bytes" => {
            // String → [m]I64 (UTF-8 byte values)
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("m".into()),
                Type::Prim(goth_ast::types::PrimType::I64)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "bytes".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "joinStrings" => {
            // [n]String → String (join array of strings)
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("m".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "joinStrings".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Additional math functions
        "sin" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Sin,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "cos" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Cos,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "tan" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Tan,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "asin" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Asin,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "acos" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Acos,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "atan" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Atan,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "sinh" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Sinh,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "cosh" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Cosh,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "tanh" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Tanh,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "exp" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Exp,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "ln" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Ln,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "log" | "log10" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Log10,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "log2" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Log2,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "gamma" | "Γ" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Gamma,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "sign" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Sign,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "round" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::UnaryOp(
                goth_ast::op::UnaryOp::Round,
                arg_op
            ));
            Ok((Operand::Local(dest), result_ty))
        }
        "norm" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "norm".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // I/O operations
        "write" => {
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "write".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "flush" => {
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "flush".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "readLine" => {
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "readLine".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "readKey" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "readKey".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "rawModeEnter" => {
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "rawModeEnter".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "rawModeExit" => {
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "rawModeExit".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "sleep" => {
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "sleep".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "readFile" => {
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "readFile".to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Default: generic primitive call
        _ => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64); // Placeholder
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: prim_name.to_string(),
                args: vec![arg_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
    }
}

/// Lower a two-argument primitive application
fn lower_primitive_app2(ctx: &mut LoweringContext, prim_name: &str, arg1: &Expr, arg2: &Expr) -> MirResult<(Operand, Type)> {
    let (arg1_op, arg1_ty) = lower_expr_to_operand(ctx, arg1)?;
    let (arg2_op, arg2_ty) = lower_expr_to_operand(ctx, arg2)?;
    let dest = ctx.fresh_local();

    match prim_name {
        "range" | "…" => {
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::Prim(goth_ast::types::PrimType::I64)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Range(arg1_op, arg2_op));
            Ok((Operand::Local(dest), result_ty))
        }

        "map" | "↦" => {
            let result_ty = arg1_ty.clone(); // Tensor type preserved
            ctx.emit(dest, result_ty.clone(), Rhs::TensorMap {
                tensor: arg1_op,
                func: arg2_op,
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "filter" | "▸" => {
            let result_ty = arg1_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::TensorFilter {
                tensor: arg1_op,
                pred: arg2_op,
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "take" | "↑" => {
            let result_ty = arg2_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "take".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "drop" | "↓" => {
            let result_ty = arg2_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "drop".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "index" => {
            // Indexing: result type is element type of tensor
            let result_ty = match &arg1_ty {
                Type::Tensor(_, elem) => (**elem).clone(),
                _ => Type::Prim(goth_ast::types::PrimType::I64),
            };
            ctx.emit(dest, result_ty.clone(), Rhs::Index(arg1_op, arg2_op));
            Ok((Operand::Local(dest), result_ty))
        }

        "concat" | "⧺" => {
            let result_ty = arg1_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "concat".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "dot" | "·" => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::F64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "dot".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "matmul" => {
            let result_ty = arg1_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "matmul".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        "reshape" => {
            let result_ty = arg2_ty.clone();
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "reshape".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // String operations (two-argument)
        "strConcat" => {
            // String → String → String
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("p".into()),
                Type::Prim(goth_ast::types::PrimType::Char)
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "strConcat".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "strEq" => {
            // String → String → Bool
            let result_ty = Type::Prim(goth_ast::types::PrimType::Bool);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "strEq".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "startsWith" => {
            // String → String → Bool
            let result_ty = Type::Prim(goth_ast::types::PrimType::Bool);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "startsWith".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "endsWith" => {
            // String → String → Bool
            let result_ty = Type::Prim(goth_ast::types::PrimType::Bool);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "endsWith".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "contains" => {
            // String → String → Bool
            let result_ty = Type::Prim(goth_ast::types::PrimType::Bool);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "contains".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Array operations (two-argument)
        "replicate" => {
            // I64 → α → [n]α
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                arg2_ty.clone()
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "replicate".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
        "zip" => {
            // [n]α → [n]β → [n]⟨α, β⟩
            let elem1 = match &arg1_ty {
                Type::Tensor(_, elem) => (**elem).clone(),
                _ => arg1_ty.clone(),
            };
            let elem2 = match &arg2_ty {
                Type::Tensor(_, elem) => (**elem).clone(),
                _ => arg2_ty.clone(),
            };
            let result_ty = Type::vector(
                goth_ast::shape::Dim::Var("n".into()),
                Type::tuple(vec![elem1, elem2])
            );
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "zip".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // File I/O (two-argument)
        "writeFile" => {
            // String → String → Unit
            let result_ty = Type::Tuple(vec![]);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: "writeFile".to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }

        // Default: generic primitive call
        _ => {
            let result_ty = Type::Prim(goth_ast::types::PrimType::I64);
            ctx.emit(dest, result_ty.clone(), Rhs::Prim {
                name: prim_name.to_string(),
                args: vec![arg1_op, arg2_op],
            });
            Ok((Operand::Local(dest), result_ty))
        }
    }
}

/// Lower a top-level expression to a Program
pub fn lower_expr(expr: &Expr) -> MirResult<Program> {
    let mut ctx = LoweringContext::new();

    let (result_op, result_ty) = lower_expr_to_operand(&mut ctx, expr)?;

    // Determine body block
    let body = if let Some(entry_id) = ctx.entry_block_id {
        // Control flow was created - find and extract the entry block
        let entry_idx = ctx.blocks.iter().position(|(id, _)| *id == entry_id)
            .expect("entry block not found");
        let (_, entry_block) = ctx.blocks.remove(entry_idx);

        // Update ALL join blocks (those with Unreachable) to return the result
        // The last join block (highest ID with Unreachable) gets the Return
        // Others need to flow to it
        let mut last_join_idx = None;
        for (i, (_, block)) in ctx.blocks.iter().enumerate() {
            if matches!(block.term, Terminator::Unreachable) {
                last_join_idx = Some(i);
            }
        }
        if let Some(idx) = last_join_idx {
            ctx.blocks[idx].1.term = Terminator::Return(result_op);
        }

        entry_block
    } else {
        let stmts = ctx.take_stmts();
        Block {
            stmts,
            term: Terminator::Return(result_op),
        }
    };

    let main_fn = Function {
        name: "main".to_string(),
        params: vec![],
        ret_ty: result_ty,
        body,
        blocks: ctx.blocks,
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
            type_: None,
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
            type_: None,
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::Let {
                pattern: Pattern::Var(Some("y".into())),
                type_: None,
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
            type_: None,
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
            type_: None,
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
        // Should have If terminator
        assert!(matches!(main.body.term, Terminator::If { .. }),
            "Entry block should have If terminator");
        // Should have additional blocks for then, else, and join
        assert_eq!(main.blocks.len(), 3,
            "Should have 3 blocks (then, else, join)");
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
            type_: None,
            value: Box::new(Expr::Lit(Literal::Int(5))),
            body: Box::new(Expr::Let {
                pattern: Pattern::Var(Some("y".into())),
                type_: None,
                value: Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Idx(0)),  // x
                    Box::new(Expr::Lit(Literal::Int(2))),
                )),
                body: Box::new(Expr::Let {
                    pattern: Pattern::Var(Some("f".into())),
                    type_: None,
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

/// Try to extract a constant from a simple literal expression
fn try_extract_constant(expr: &Expr) -> Option<(Constant, Type)> {
    match expr {
        Expr::Lit(lit) => Some(lower_literal(lit)),
        // Could add constant folding for simple binary operations later
        _ => None,
    }
}

/// Lower a module to a Program
pub fn lower_module(module: &Module) -> MirResult<Program> {
    let mut ctx = LoweringContext::new();
    let mut lowered_fns = Vec::new();

    // First pass: register all function signatures and global constants
    for decl in &module.decls {
        match decl {
            Decl::Fn(fn_decl) => {
                ctx.globals.insert(fn_decl.name.to_string(), fn_decl.signature.clone());
            }
            Decl::Let(let_decl) => {
                // Try to extract constant value from the expression
                if let Some((constant, ty)) = try_extract_constant(&let_decl.value) {
                    // Use explicit type annotation if available
                    let ty = let_decl.type_.clone().unwrap_or(ty);
                    ctx.global_constants.insert(let_decl.name.to_string(), (constant, ty));
                }
            }
            Decl::Enum(enum_decl) => {
                // Register each variant as a constructor
                for (tag, variant) in enum_decl.variants.iter().enumerate() {
                    let has_payload = variant.payload.is_some();
                    ctx.enum_constructors.insert(
                        variant.name.to_string(),
                        (tag as u32, has_payload, enum_decl.name.to_string())
                    );
                }
            }
            _ => {}
        }
    }

    // Second pass: lower function bodies
    for decl in &module.decls {
        match decl {
            Decl::Fn(fn_decl) => {
                // Create fresh context for this function
                let mut fn_ctx = LoweringContext::new();
                fn_ctx.globals = ctx.globals.clone();
                fn_ctx.global_constants = ctx.global_constants.clone();
                fn_ctx.enum_constructors = ctx.enum_constructors.clone();

                // Extract parameter types from signature
                let mut param_types = Vec::new();
                let mut current_ty = &fn_decl.signature;
                while let Type::Fn(arg_ty, ret_ty) = current_ty {
                    param_types.push((**arg_ty).clone());
                    current_ty = ret_ty;
                }
                let ret_ty = current_ty.clone();

                // Allocate locals for parameters (in order)
                for _ in 0..param_types.len() {
                    fn_ctx.fresh_local();
                }
                // Push onto local stack in FORWARD order so that:
                // ₀ = last param (standard de Bruijn, matching evaluator)
                // For f : A → B → C applied as (f a b):
                //   ₀ = b (most recently applied)
                //   ₁ = a (first applied)
                // Since lookup_index does: locals[len - 1 - idx]
                for i in 0..param_types.len() {
                    fn_ctx.push_local(LocalId::new(i as u32), param_types[i].clone());
                }

                // Lower body
                let (body_op, _) = lower_expr_to_operand(&mut fn_ctx, &fn_decl.body)?;

                let body = if let Some(entry_id) = fn_ctx.entry_block_id {
                    // Control flow exists - extract entry block
                    let entry_idx = fn_ctx.blocks.iter().position(|(id, _)| *id == entry_id)
                        .expect("entry block not found");
                    let (_, entry_block) = fn_ctx.blocks.remove(entry_idx);

                    // Update the last Unreachable block to Return
                    let mut last_join_idx = None;
                    for (i, (_, block)) in fn_ctx.blocks.iter().enumerate() {
                        if matches!(block.term, Terminator::Unreachable) {
                            last_join_idx = Some(i);
                        }
                    }
                    if let Some(idx) = last_join_idx {
                        fn_ctx.blocks[idx].1.term = Terminator::Return(body_op);
                    }

                    entry_block
                } else {
                    Block {
                        stmts: fn_ctx.take_stmts(),
                        term: Terminator::Return(body_op),
                    }
                };

                lowered_fns.push(Function {
                    name: fn_decl.name.to_string(),
                    params: param_types,
                    ret_ty,
                    body,
                    blocks: fn_ctx.blocks,
                    is_closure: false,
                });

                // Collect any lifted lambdas
                lowered_fns.extend(fn_ctx.functions);
            }
            Decl::Let(_let_decl) => {
                // Global constants are handled in the first pass
            }
            _ => {}
        }
    }

    // Find entry point (main function)
    let entry = if lowered_fns.iter().any(|f| f.name == "main") {
        "main".to_string()
    } else if let Some(f) = lowered_fns.first() {
        f.name.clone()
    } else {
        // Create empty main if no functions
        lowered_fns.push(Function {
            name: "main".to_string(),
            params: vec![],
            ret_ty: Type::Tuple(vec![]),
            body: Block::with_return(Operand::Const(Constant::Unit)),
            blocks: vec![],
            is_closure: false,
        });
        "main".to_string()
    };

    Ok(Program {
        functions: lowered_fns,
        entry,
    })
}
