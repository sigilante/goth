//! Goth CLI and REPL
//!
//! Usage:
//!   goth              - Start REPL
//!   goth <file.goth>  - Run a file
//!   goth -e <expr>    - Evaluate expression

use clap::Parser;
use colored::Colorize;
use goth_ast::ser::{to_json, to_json_compact, from_json, expr_to_json};
use goth_ast::pretty;
use goth_eval::prelude::*;
use goth_parse::prelude::*;
use goth_check::TypeChecker;
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result as RlResult};
use std::fs;
use std::path::PathBuf;

/// Count the arity of a function type (number of arguments)
/// For example: F → F → F has arity 2
fn count_function_arity(ty: &goth_ast::types::Type) -> u32 {
    let mut arity = 0u32;
    let mut current = ty;
    
    loop {
        match current {
            goth_ast::types::Type::Fn(_arg, ret) => {
                arity += 1;
                current = ret;
            }
            _ => break,
        }
    }
    
    // If no function arrows found, it's a value (arity 0), but for closures we need at least 1
    arity.max(1)
}

#[derive(Parser, Debug)]
#[command(name = "goth")]
#[command(author = "Goth Language")]
#[command(version = "0.1.0")]
#[command(about = "The Goth programming language", long_about = None)]
struct Args {
    /// File to execute
    #[arg()]
    file: Option<PathBuf>,

    /// Arguments to pass to the main function
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    program_args: Vec<String>,

    /// Evaluate expression
    #[arg(short, long)]
    eval: Option<String>,

    /// Enable trace output
    #[arg(short, long)]
    trace: bool,

    /// Parse only (don't evaluate)
    #[arg(short, long)]
    parse_only: bool,

    /// Show AST
    #[arg(short, long)]
    ast: bool,

    /// Type check expressions before evaluation
    #[arg(long, short = 'c')]
    check: bool,

    /// Don't look for or execute main function (just load declarations)
    #[arg(long)]
    no_main: bool,

    // ============ AST-First LLM Workflow ============

    /// Output JSON AST instead of evaluating (for LLM consumption)
    #[arg(long)]
    to_json: bool,

    /// Read JSON AST from file instead of Goth source (for LLM-generated code)
    #[arg(long)]
    from_json: Option<PathBuf>,

    /// Render AST to Goth syntax (use with --from-json to see pretty output)
    #[arg(long)]
    render: bool,

    /// Output compact JSON (no pretty-printing)
    #[arg(long)]
    compact: bool,
}

fn main() {
    let args = Args::parse();

    // ============ AST-First LLM Workflow ============

    // Handle --from-json: Read JSON AST, validate, optionally render/eval
    if let Some(json_path) = args.from_json {
        // When using --from-json, the `file` positional arg (if present) becomes the first program arg
        let mut effective_args = args.program_args.clone();
        if let Some(ref file) = args.file {
            effective_args.insert(0, file.to_string_lossy().to_string());
        }
        run_from_json(&json_path, args.check, args.render, args.trace, &effective_args);
        return;
    }

    // Handle --to-json with file: Parse Goth, emit JSON AST
    if args.to_json {
        if let Some(ref file) = args.file {
            run_to_json_file(file, args.compact);
            return;
        } else if let Some(ref expr) = args.eval {
            run_to_json_expr(expr, args.compact);
            return;
        } else {
            eprintln!("{}: --to-json requires a file or -e expression", "Error".red().bold());
            return;
        }
    }

    // ============ Standard Workflow ============

    if let Some(expr) = args.eval {
        // Evaluate expression from command line
        run_expr(&expr, args.trace, args.parse_only, args.ast, args.check);
        return;
    } else if let Some(file) = args.file {
        // Run file
        run_file(&file, args.trace, args.parse_only, args.ast, args.no_main, &args.program_args);
    } else {
        // Start REPL
        if let Err(e) = run_repl(args.trace) {
            eprintln!("{}: {}", "Error".red().bold(), e);
        }
    }
}

fn run_expr(source: &str, trace: bool, parse_only: bool, show_ast: bool, check: bool) {
    use colored::Colorize;

    // Parse
    let parsed = match parse_expr(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}: {}", "Parse error".red().bold(), e);
            return;
        }
    };

    // Show AST if requested
    if show_ast {
        println!("{}", "AST:".cyan().bold());
        println!("{:#?}", parsed);
    }

    // Resolve
    let resolved = resolve_expr(parsed);

    // Type check if --check flag is set
    if check {
        let mut type_checker = TypeChecker::new();
        match type_checker.infer(&resolved) {
            Ok(ty) => {
                println!("{}: {}", "Type".cyan(), ty);
            }
            Err(e) => {
                eprintln!("{}: {}", "Type error".red().bold(), e);
                return;  // Don't evaluate if type check fails
            }
        }
    }

    if parse_only {
        return;
    }

    // Evaluate
    let mut evaluator = if trace {
        Evaluator::new().with_trace(true)
    } else {
        Evaluator::new()
    };

    match evaluator.eval(&resolved) {
        Ok(value) => {
            println!("{}", value);
        }
        Err(e) => {
            eprintln!("{}: {}", "Error".red().bold(), e);
        }
    }
}

fn run_file(path: &PathBuf, trace: bool, parse_only: bool, show_ast: bool, no_main: bool, program_args: &[String]) {
    // Use the loader to handle `use` declarations (resolves imports)
    match load_file(path) {
        Ok(module) => {
            if show_ast {
                println!("{} {:#?}", "Parsed AST:".cyan().bold(), module);
            }
            if parse_only {
                println!("{} module '{}' with {} declarations",
                    "Parsed:".green().bold(),
                    module.name.as_ref().map(|s| s.as_ref()).unwrap_or("anonymous"),
                    module.decls.len());
                return;
            }
            let module = resolve_module(module);
            if show_ast {
                println!("{} {:#?}", "Resolved AST:".cyan().bold(), module);
            }

            // Type check the module
            let mut type_checker = TypeChecker::new();
            if let Err(e) = type_checker.check_module(&module) {
                eprintln!("{}: {}", "Type error".red().bold(), e);
                return;
            }

            if no_main {
                // Just load declarations like before (verbose mode)
                run_module(&module, trace);
            } else {
                // Load declarations silently and execute main
                run_module_with_main(&module, trace, program_args);
            }
        }
        Err(e) => eprintln!("{}: {}", "Load error".red().bold(), e),
    }
}

fn run_module(module: &goth_ast::decl::Module, trace: bool) {
    use goth_ast::decl::Decl;
    
    let mut evaluator = Evaluator::new();
    if trace {
        evaluator = evaluator.with_trace(true);
    }

    for decl in &module.decls {
        match decl {
            Decl::Let(let_decl) => {
                match evaluator.eval(&let_decl.value) {
                    Ok(value) => {
                        evaluator.define(let_decl.name.to_string(), value.clone());
                        println!("{} {} = {}", "let".cyan(), let_decl.name, value);
                    }
                    Err(e) => {
                        eprintln!("{}: in '{}': {}", "Error".red().bold(), let_decl.name, e);
                    }
                }
            }
            Decl::Fn(fn_decl) => {
                let arity = count_function_arity(&fn_decl.signature);
                let closure = Value::closure_with_contracts(
                    arity, 
                    fn_decl.body.clone(), 
                    Env::with_globals(evaluator.globals()),
                    fn_decl.preconditions.clone(),
                    fn_decl.postconditions.clone()
                );
                evaluator.define(fn_decl.name.to_string(), closure);
                println!("{} {} : {}", "fn".cyan(), fn_decl.name, fn_decl.signature);
            }
            Decl::Type(type_decl) => {
                println!("{} {} = {}", "type".cyan(), type_decl.name, type_decl.definition);
            }
            _ => {}
        }
    }
}

/// Run a module by loading declarations and executing main function
fn run_module_with_main(module: &goth_ast::decl::Module, trace: bool, program_args: &[String]) {
    use goth_ast::decl::Decl;
    use goth_ast::expr::Expr;

    let mut evaluator = Evaluator::new();
    if trace {
        evaluator = evaluator.with_trace(true);
    }

    // Load all declarations silently
    for decl in &module.decls {
        match decl {
            Decl::Let(let_decl) => {
                match evaluator.eval(&let_decl.value) {
                    Ok(value) => {
                        evaluator.define(let_decl.name.to_string(), value);
                    }
                    Err(e) => {
                        eprintln!("{}: in '{}': {}", "Error".red().bold(), let_decl.name, e);
                        return;
                    }
                }
            }
            Decl::Fn(fn_decl) => {
                let arity = count_function_arity(&fn_decl.signature);
                let closure = Value::closure_with_contracts(
                    arity,
                    fn_decl.body.clone(),
                    Env::with_globals(evaluator.globals()),
                    fn_decl.preconditions.clone(),
                    fn_decl.postconditions.clone()
                );
                evaluator.define(fn_decl.name.to_string(), closure);
            }
            _ => {}
        }
    }

    // Look for main function
    let has_main = evaluator.globals().borrow().contains_key("main");
    if !has_main {
        eprintln!("{}: no 'main' function found", "Error".red().bold());
        eprintln!("  {} Define a main function or use --no-main to load declarations only", "hint:".yellow());
        return;
    }

    // Build an expression that calls main with the arguments
    // Start with just referencing the main function
    let mut call_expr = Expr::name("main");

    // Apply each argument
    if program_args.is_empty() {
        // If main takes no args, just call it (or pass unit)
        call_expr = Expr::app(call_expr, Expr::Tuple(vec![]));
    } else {
        for arg in program_args {
            let arg_expr = parse_arg_to_expr(arg);
            call_expr = Expr::app(call_expr, arg_expr);
        }
    }

    // Evaluate the call expression
    match evaluator.eval(&call_expr) {
        Ok(value) => {
            // Print result unless it's unit
            if !matches!(value, Value::Unit) {
                println!("{}", value);
            }
        }
        Err(e) => {
            eprintln!("{}: {}", "Error".red().bold(), e);
        }
    }
}

/// Parse a CLI argument string into a Goth expression
fn parse_arg_to_expr(arg: &str) -> goth_ast::expr::Expr {
    use goth_ast::expr::Expr;
    use goth_ast::literal::Literal;

    // Try to parse as integer first
    if let Ok(n) = arg.parse::<i128>() {
        return Expr::Lit(Literal::Int(n));
    }

    // Try to parse as float
    if let Ok(f) = arg.parse::<f64>() {
        return Expr::Lit(Literal::Float(f));
    }

    // Try to parse as boolean
    match arg {
        "true" | "⊤" => return Expr::Lit(Literal::True),
        "false" | "⊥" => return Expr::Lit(Literal::False),
        _ => {}
    }

    // Otherwise, treat as string
    Expr::Lit(Literal::String(arg.into()))
}

fn run_repl(trace: bool) -> RlResult<()> {
    print_banner();
    
    let mut rl = DefaultEditor::new()?;
    let history_path = dirs::data_dir()
        .map(|p| p.join("goth").join("history.txt"));
    
    if let Some(ref path) = history_path {
        let _ = fs::create_dir_all(path.parent().unwrap());
        let _ = rl.load_history(path);
    }

    let mut evaluator = Evaluator::new();
    let mut type_checker = TypeChecker::new();
    if trace {
        evaluator = evaluator.with_trace(true);
    }
    
    let mut line_count = 0;
    let mut accumulated = String::new();

    loop {
        let prompt = if accumulated.is_empty() {
            format!("{} ", format!("𝖌𝖔𝖙𝖍[{}]›", line_count).cyan())
        } else {
            // Continuation prompt - colorize dots to distinguish from input
            let main_prompt = format!("𝖌𝖔𝖙𝖍[{}]›", line_count);
            let width = main_prompt.len();
            let dots = format!("{}", ".".repeat(width).dimmed());
            format!("{} ", dots)
        };
        
        match rl.readline(&prompt) {
            Ok(line) => {
                // Accumulate the line
                if !accumulated.is_empty() {
                    accumulated.push('\n');
                }
                accumulated.push_str(&line);
                
                // Check if input is complete
                if !is_complete(&accumulated) {
                    continue; // Read more lines
                }
                
                let input = accumulated.trim();
                if input.is_empty() {
                    accumulated.clear();
                    continue;
                }

                let _ = rl.add_history_entry(input);

                // Handle REPL commands
                if input.starts_with(':') {
                    handle_command(input, &mut evaluator, &mut type_checker, trace);
                    accumulated.clear();
                    continue;
                }

                // Try to parse and evaluate
                match parse_and_eval(input, &mut evaluator, &mut type_checker) {
                    Ok(Some(value)) => {
                        print_value(&value);
                        // Bind result to _
                        evaluator.define("_", value);
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("{}: {}", "Error".red().bold(), e),
                }

                accumulated.clear();
                line_count += 1;
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", "^C".yellow());
                accumulated.clear();
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("{}", "Goodbye!".cyan());
                break;
            }
            Err(err) => {
                eprintln!("{}: {:?}", "Error".red().bold(), err);
                break;
            }
        }
    }

    if let Some(ref path) = history_path {
        let _ = rl.save_history(path);
    }

    Ok(())
}

fn parse_and_eval(input: &str, evaluator: &mut Evaluator, type_checker: &mut TypeChecker) -> Result<Option<Value>, String> {
    // Try parsing as let declaration first
    if input.starts_with("let ") {
        // Check if it's a top-level let (no 'in')
        // Must check for both " in " and " in\n" patterns
        let has_in_body = input.contains(" in ") || input.contains(" in\n");
        if !has_in_body {
            // Parse as declaration
            match parse_module(input, "repl") {
                Ok(module) => {
                    let module = resolve_module(module);
                    for decl in module.decls {
                        if let goth_ast::decl::Decl::Let(let_decl) = decl {
                            let value = evaluator.eval(&let_decl.value)
                                .map_err(|e| e.to_string())?;
                            evaluator.define(let_decl.name.to_string(), value.clone());
                            
                            // Also infer type and add to type checker
                            if let Ok(ty) = type_checker.infer(&let_decl.value) {
                                type_checker.define(let_decl.name.to_string(), ty);
                            }
                            
                            println!("{} {} = {}", "let".cyan(), let_decl.name, value);
                        }
                    }
                    return Ok(None);
                }
                Err(_) => {} // Fall through to expression parsing
            }
        }
    }

    // Try parsing as function definition
    if input.starts_with("╭─") || input.starts_with("/-") {
        match parse_module(input, "repl") {
            Ok(module) => {
                let module = resolve_module(module);
                for decl in module.decls {
                    if let goth_ast::decl::Decl::Fn(fn_decl) = decl {
                        let arity = count_function_arity(&fn_decl.signature);
                        let closure = Value::closure_with_contracts(
                            arity,
                            fn_decl.body.clone(),
                            Env::with_globals(evaluator.globals()),
                            fn_decl.preconditions.clone(),
                            fn_decl.postconditions.clone()
                        );
                        evaluator.define(fn_decl.name.to_string(), closure);
                        // Also add to type checker!
                        type_checker.define(fn_decl.name.to_string(), fn_decl.signature.clone());
                        println!("{} {} : {}", "fn".cyan(), fn_decl.name, fn_decl.signature);
                    }
                }
                return Ok(None);
            }
            Err(e) => return Err(format!("Parse error: {}", e)),
        }
    }

    // Parse as expression
    let expr = parse_expr(input).map_err(|e| format!("Parse error: {}", e))?;
    let expr = resolve_expr(expr);

    // Type check before evaluation (show type or warning, don't block)
    match type_checker.infer(&expr) {
        Ok(ty) => {
            // Show inferred type in dim style (like GHCi)
            println!("{} {}", "::".dimmed(), format!("{}", ty).dimmed());
        }
        Err(e) => {
            // Show type error as warning but continue with evaluation
            eprintln!("{}: {}", "Type warning".yellow(), e);
        }
    }

    let value = evaluator.eval(&expr).map_err(|e| e.to_string())?;
    Ok(Some(value))
}

fn handle_command(cmd: &str, evaluator: &mut Evaluator, type_checker: &mut TypeChecker, _trace: bool) {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    let command = parts[0];
    let arg = parts.get(1).map(|s| s.trim());

    match command {
        ":help" | ":h" | ":?" => print_help(),
        ":quit" | ":q" => std::process::exit(0),
        ":ast" => {
            if let Some(expr_str) = arg {
                match parse_expr(expr_str) {
                    Ok(expr) => {
                        println!("{}", "Parsed:".cyan().bold());
                        println!("{:#?}", expr);
                        let resolved = resolve_expr(expr);
                        println!("{}", "Resolved:".cyan().bold());
                        println!("{:#?}", resolved);
                    }
                    Err(e) => eprintln!("{}: {}", "Parse error".red().bold(), e),
                }
            } else {
                eprintln!("Usage: :ast <expression>");
            }
        }
        ":type" | ":t" => {
            if let Some(expr_str) = arg {
                match parse_expr(expr_str) {
                    Ok(expr) => {
                        let resolved = resolve_expr(expr);
                        match type_checker.infer(&resolved) {
                            Ok(ty) => println!("{}", ty),
                            Err(e) => eprintln!("{}: {}", "Type error".red().bold(), e),
                        }
                    }
                    Err(e) => eprintln!("{}: {}", "Parse error".red().bold(), e),
                }
            } else {
                eprintln!("Usage: :type <expression>");
            }
        }
        ":clear" => {
            *evaluator = Evaluator::new();
            *type_checker = TypeChecker::new();
            println!("{}", "Environment cleared.".yellow());
        }
        ":load" | ":l" => {
            if let Some(path) = arg {
                match fs::read_to_string(path) {
                    Ok(source) => {
                        match parse_module(&source, "loaded") {
                            Ok(module) => {
                                let module = resolve_module(module);
                                for decl in module.decls {
                                    match decl {
                                        goth_ast::decl::Decl::Let(let_decl) => {
                                            if let Ok(value) = evaluator.eval(&let_decl.value) {
                                                evaluator.define(let_decl.name.to_string(), value);
                                                println!("{} {}", "Loaded:".green(), let_decl.name);
                                            }
                                        }
                                        goth_ast::decl::Decl::Fn(fn_decl) => {
                                            let arity = count_function_arity(&fn_decl.signature);
                                            let closure = Value::closure_with_contracts(
                                                arity,
                                                fn_decl.body.clone(),
                                                Env::with_globals(evaluator.globals()),
                                                fn_decl.preconditions.clone(),
                                                fn_decl.postconditions.clone()
                                            );
                                            evaluator.define(fn_decl.name.to_string(), closure);
                                            println!("{} {}", "Loaded:".green(), fn_decl.name);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            Err(e) => eprintln!("{}: {}", "Parse error".red().bold(), e),
                        }
                    }
                    Err(e) => eprintln!("{}: {}", "File error".red().bold(), e),
                }
            } else {
                eprintln!("Usage: :load <file.goth>");
            }
        }
        _ => eprintln!("{}: unknown command '{}'. Try :help", "Error".red().bold(), command),
    }
}

fn print_value(value: &Value) {
    match value {
        Value::Int(n) => println!("{}", n.to_string().green()),
        Value::Float(f) => println!("{}", format!("{}", f).green()),
        Value::Bool(true) => println!("{}", "⊤".green().bold()),
        Value::Bool(false) => println!("{}", "⊥".green().bold()),
        Value::Char(c) => println!("{}", format!("'{}'", c).yellow()),
        Value::Unit => println!("{}", "⟨⟩".dimmed()),
        Value::Tensor(t) => println!("{}", format!("{}", t).blue()),
        Value::Tuple(vs) => {
            print!("{}", "⟨".dimmed());
            for (i, v) in vs.iter().enumerate() {
                if i > 0 { print!("{}", ", ".dimmed()); }
                print_value_inline(v);
            }
            println!("{}", "⟩".dimmed());
        }
        Value::Closure(c) => println!("{}", format!("<λ/{}>", c.arity).magenta()),
        Value::Primitive(p) => println!("{}", format!("<prim:{:?}>", p).magenta()),
        Value::Partial { remaining, .. } => println!("{}", format!("<partial/{}>", remaining).magenta()),
        Value::Variant { tag, payload } => {
            print!("{}", tag.cyan());
            if let Some(p) = payload {
                print!(" ");
                print_value_inline(p);
            }
            println!();
        }
        Value::Error(msg) => println!("{}: {}", "Error".red().bold(), msg),
        _ => println!("{}", format!("{}", value).white()),
    }
}

fn print_value_inline(value: &Value) {
    match value {
        Value::Int(n) => print!("{}", n.to_string().green()),
        Value::Float(f) => print!("{}", format!("{}", f).green()),
        Value::Bool(true) => print!("{}", "⊤".green().bold()),
        Value::Bool(false) => print!("{}", "⊥".green().bold()),
        Value::Char(c) => print!("{}", format!("'{}'", c).yellow()),
        Value::Unit => print!("{}", "⟨⟩".dimmed()),
        Value::Tensor(t) => print!("{}", format!("{}", t).blue()),
        _ => print!("{}", value),
    }
}

/// Check if input is syntactically complete
fn is_complete(input: &str) -> bool {
    let input = input.trim();
    
    // Empty input is complete
    if input.is_empty() {
        return true;
    }
    
    // Check balanced delimiters
    let mut parens = 0;
    let mut brackets = 0;
    let mut braces = 0;
    let mut angles = 0;
    let mut in_string = false;
    let mut in_char = false;
    let mut escape_next = false;
    
    for ch in input.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        
        if in_string {
            if ch == '\\' {
                escape_next = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        
        if in_char {
            if ch == '\\' {
                escape_next = true;
            } else if ch == '\'' {
                in_char = false;
            }
            continue;
        }
        
        match ch {
            '"' => in_string = true,
            '\'' => in_char = true,
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            '{' => braces += 1,
            '}' => braces -= 1,
            '⟨' => angles += 1,
            '⟩' => angles -= 1,
            _ => {}
        }
    }
    
    // If any delimiter is unbalanced, incomplete
    if parens != 0 || brackets != 0 || braces != 0 || angles != 0 || in_string || in_char {
        return false;
    }
    
    // Check for trailing operators that expect continuation
    let trailing_ops = [
        "+", "-", "*", "×", "/", "^",
        "↦", "⊗", "⊕", "▸", "⤇", "∘",
        "-:", "*:", "+:", "|>", "=>", ".:",
        "=", "<", ">", "≤", "≥", "≠",
        "∧", "∨", "&&", "||",
    ];
    
    for op in trailing_ops {
        if input.ends_with(op) {
            return false;
        }
    }
    
    // Check for incomplete let binding
    // - `let x = 42` (no 'in') -> complete (top-level binding)
    // - `let x = 42 in` (ends with 'in') -> incomplete (needs body)
    // - `let x = 42 in expr` -> complete
    if input.starts_with("let ") && input.contains('=') {
        let trimmed = input.trim_end();
        // If input ends with " in", it's incomplete (waiting for body)
        if trimmed.ends_with(" in") {
            return false;
        }
    }
    
    // Check for incomplete if-then (without else)
    if (input.contains(" if ") || input.contains("\nif ")) 
        && (input.contains(" then ") || input.contains("\nthen ")) 
        && !input.contains(" else ") && !input.contains("\nelse ") {
        return false;
    }
    
    // Check for incomplete match (without closing brace or without any arms)
    if input.starts_with("match ") || input.contains(" match ") {
        // Simple heuristic: needs at least one → and proper closing
        if !input.contains('→') && !input.contains("->") {
            return false;
        }
    }
    
    // Check for unclosed function declaration (╭─ without ╰─)
    if input.contains("╭─") || input.contains("/-") {
        if !(input.contains("╰─") || input.contains("\\-")) {
            return false;
        }
    }
    
    true
}

// ============ AST-First LLM Workflow Handlers ============

/// Read JSON AST from file, validate, optionally render/eval
fn run_from_json(path: &PathBuf, check: bool, render: bool, trace: bool, program_args: &[String]) {
    // Read JSON file
    let json_content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("{}: reading {}: {}", "Error".red().bold(), path.display(), e);
            return;
        }
    };

    // Parse JSON to Module
    let module = match from_json(&json_content) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}: invalid JSON AST: {}", "Error".red().bold(), e);
            return;
        }
    };

    println!("{}: parsed {} declaration(s)", "OK".green().bold(), module.decls.len());

    // JSON AST may contain both Names (for globals) and Idx (for lambdas)
    // Resolve any remaining Names to indices
    let module = resolve_module(module);

    // Type check if requested
    if check {
        let mut type_checker = TypeChecker::new();
        match type_checker.check_module(&module) {
            Ok(_bindings) => println!("{}: type check passed", "OK".green().bold()),
            Err(e) => {
                eprintln!("{}: {}", "Type error".red().bold(), e);
                return;
            }
        }
    }

    // Render to Goth syntax if requested
    if render {
        println!();
        println!("{}", "─".repeat(40).dimmed());
        let output = pretty::print_module(&module);
        println!("{}", output);
        println!("{}", "─".repeat(40).dimmed());
    } else {
        // If not rendering, evaluate
        run_module_with_main(&module, trace, program_args);
    }
}

/// Parse Goth source file, emit JSON AST
fn run_to_json_file(path: &PathBuf, compact: bool) {
    // Load and parse the file
    let module = match load_file(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}: {}", "Load error".red().bold(), e);
            return;
        }
    };

    // Serialize to JSON
    let json = if compact {
        match to_json_compact(&module) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("{}: {}", "JSON error".red().bold(), e);
                return;
            }
        }
    } else {
        match to_json(&module) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("{}: {}", "JSON error".red().bold(), e);
                return;
            }
        }
    };

    println!("{}", json);
}

/// Parse Goth expression, emit JSON AST
fn run_to_json_expr(source: &str, compact: bool) {
    // Parse expression
    let expr = match parse_expr(source) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{}: {}", "Parse error".red().bold(), e);
            return;
        }
    };

    // Serialize to JSON
    let json = if compact {
        match serde_json::to_string(&expr) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("{}: {}", "JSON error".red().bold(), e);
                return;
            }
        }
    } else {
        match expr_to_json(&expr) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("{}: {}", "JSON error".red().bold(), e);
                return;
            }
        }
    };

    println!("{}", json);
}

fn print_banner() {
    println!("{}", r#"
   ╔═══════════════════════════════════════╗
   ║             𝖌𝖔𝖙𝖍  v0.1.0              ║
   ║   Functional • Tensors • Refinements  ║
   ╚═══════════════════════════════════════╝
"#.cyan());
    println!("  Type {} for help, {} to quit\n", ":help".yellow(), ":quit".yellow());
}

fn print_help() {
    println!("{}", "
Commands:
  :help, :h, :?    Show this help
  :quit, :q        Exit REPL
  :ast <expr>      Show AST for expression
  :type <expr>     Show type of expression
  :clear           Clear environment
  :load <file>     Load definitions from file

Syntax:
  λ→ ₀ + 1         Lambda (or \\-> _0 + 1)
  [1, 2, 3]        Array
  ⟨x, y⟩           Tuple (or (x, y))
  Σ [1,2,3]        Sum (or +/ [1,2,3])
  xs ↦ f           Map (or xs -: f)
  xs ▸ p           Filter
  f ∘ g            Compose (or f .: g)

Examples:
  1 + 2 * 3
  let x = 5 in x * x
  [1,2,3,4] ↦ λ→ ₀ * 2
  Σ [1,2,3,4,5]
".cyan());
}
