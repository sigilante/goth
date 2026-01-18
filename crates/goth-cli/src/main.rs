//! Goth CLI and REPL
//!
//! Usage:
//!   goth              - Start REPL
//!   goth <file.goth>  - Run a file
//!   goth -e <expr>    - Evaluate expression

use clap::Parser;
use colored::Colorize;
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
}

fn main() {
    let args = Args::parse();

    if let Some(expr) = args.eval {
        // Evaluate expression from command line
        run_expr(&expr, args.trace, args.parse_only, args.ast, args.check);
        return;
    } else if let Some(file) = args.file {
        // Run file
        run_file(&file, args.trace, args.parse_only, args.ast);
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

fn run_file(path: &PathBuf, trace: bool, parse_only: bool, show_ast: bool) {
    match fs::read_to_string(path) {
        Ok(source) => {
            let name = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("main");
            
            match parse_module(&source, name) {
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
                    run_module(&module, trace);
                }
                Err(e) => eprintln!("{}: {}", "Parse error".red().bold(), e),
            }
        }
        Err(e) => eprintln!("{}: {}", "File error".red().bold(), e),
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
        if !input.contains(" in ") {
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
    
    // Check for incomplete let binding (let x = ... without in)
    if input.starts_with("let ") && input.contains('=') {
        // Check for 'in' keyword - could be " in " or "\nin " (at line start)
        if !input.contains(" in ") && !input.contains("\nin ") {
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
