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
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result as RlResult};
use std::fs;
use std::path::PathBuf;

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
}

fn main() {
    let args = Args::parse();

    if let Some(expr) = args.eval {
        // Evaluate expression from command line
        run_expr(&expr, args.trace, args.parse_only, args.ast);
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

fn run_expr(source: &str, trace: bool, parse_only: bool, show_ast: bool) {
    match parse_expr(source) {
        Ok(expr) => {
            if show_ast {
                println!("{} {:#?}", "Parsed AST:".cyan().bold(), expr);
            }
            if parse_only {
                println!("{} {}", "Parsed:".green().bold(), expr);
                return;
            }
            
            // Resolve names to de Bruijn indices
            let expr = resolve_expr(expr);
            if show_ast {
                println!("{} {:#?}", "Resolved AST:".cyan().bold(), expr);
            }
            
            let mut evaluator = Evaluator::new();
            if trace {
                evaluator = evaluator.with_trace(true);
            }
            match evaluator.eval(&expr) {
                Ok(value) => print_value(&value),
                Err(e) => eprintln!("{}: {}", "Error".red().bold(), e),
            }
        }
        Err(e) => eprintln!("{}: {}", "Parse error".red().bold(), e),
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
                let closure = Value::closure(1, fn_decl.body.clone(), Env::new());
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
    if trace {
        evaluator = evaluator.with_trace(true);
    }
    
    let mut line_count = 0;

    loop {
        let prompt = format!("{} ", format!("goth[{}]›", line_count).cyan());
        
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                // Handle REPL commands
                if line.starts_with(':') {
                    handle_command(line, &mut evaluator, trace);
                    continue;
                }

                // Try to parse and evaluate
                match parse_and_eval(line, &mut evaluator) {
                    Ok(Some(value)) => {
                        print_value(&value);
                        // Bind result to _
                        evaluator.define("_", value);
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("{}: {}", "Error".red().bold(), e),
                }

                line_count += 1;
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", "^C".yellow());
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

fn parse_and_eval(input: &str, evaluator: &mut Evaluator) -> Result<Option<Value>, String> {
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
                        let closure = Value::closure(1, fn_decl.body.clone(), Env::new());
                        evaluator.define(fn_decl.name.to_string(), closure);
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

fn handle_command(cmd: &str, evaluator: &mut Evaluator, _trace: bool) {
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
                        match evaluator.eval(&resolved) {
                            Ok(value) => println!("{}", value.type_name().cyan()),
                            Err(e) => eprintln!("{}: {}", "Error".red().bold(), e),
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
                                            let closure = Value::closure(1, fn_decl.body.clone(), Env::new());
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

fn print_banner() {
    println!("{}", r#"
   ╔═══════════════════════════════════════╗
   ║             𝔊𝔬𝔱𝔥  v0.1.0              ║
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
