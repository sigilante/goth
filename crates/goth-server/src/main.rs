//! goth-server - HTTP server for the Goth programming language
//!
//! Provides a REST API for evaluating Goth expressions and serving
//! a web-based interface.
//!
//! Usage:
//!   goth-server                      # Start server on localhost:3000
//!   goth-server --port 8080          # Custom port
//!   goth-server --host 0.0.0.0       # Listen on all interfaces
//!   goth-server --static ./web       # Serve static files from ./web

use axum::{
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{info, Level};

use goth_ast::decl::Decl;
use goth_ast::types::Type;
use goth_eval::prelude::{Env, Evaluator, Value};
use goth_parse::prelude::*;

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "goth-server")]
#[command(author = "Goth Authors")]
#[command(version = "0.1.0")]
#[command(about = "HTTP server for the Goth programming language")]
struct Args {
    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Directory to serve static files from (optional)
    #[arg(long, short = 's')]
    r#static: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Count the arity of a function type
fn count_function_arity(ty: &Type) -> u32 {
    let mut arity = 0u32;
    let mut current = ty;

    loop {
        match current {
            Type::Fn(_arg, ret) => {
                arity += 1;
                current = ret;
            }
            _ => break,
        }
    }

    // If no function arrows found, it's a value (arity 0), but for closures we need at least 1
    arity.max(1)
}

/// Evaluate Goth source code and return (value, type_name) or error
fn eval_source(source: &str) -> Result<(Value, String), String> {
    let trimmed = source.trim();

    // First, try to parse as a standalone expression
    // This handles cases like "2 + 3" or "let x = 10 in x * x"
    if let Ok(expr) = parse_expr(trimmed) {
        let resolved_expr = resolve_expr(expr);
        let mut evaluator = Evaluator::new();
        match evaluator.eval(&resolved_expr) {
            Ok(value) => {
                let type_name = value.type_name().to_string();
                return Ok((value, type_name));
            }
            Err(_) => {
                // Fall through to module parsing
            }
        }
    }

    // If expression parsing fails, try to parse as module
    // This handles multi-line definitions like functions
    let module = parse_module(source, "web").map_err(|e| format!("Parse error: {}", e))?;

    // Resolve (de Bruijn indices)
    let resolved = resolve_module(module);

    // Evaluate each declaration
    let mut evaluator = Evaluator::new();
    let mut last_value: Option<Value> = None;

    for decl in &resolved.decls {
        match decl {
            Decl::Let(let_decl) => {
                let value = evaluator
                    .eval(&let_decl.value)
                    .map_err(|e| format!("Evaluation error in '{}': {}", let_decl.name, e))?;
                evaluator.define(let_decl.name.to_string(), value.clone());
                last_value = Some(value);
            }
            Decl::Fn(fn_decl) => {
                let arity = count_function_arity(&fn_decl.signature);
                let closure = Value::closure_with_contracts(
                    arity,
                    fn_decl.body.clone(),
                    Env::with_globals(evaluator.globals()),
                    fn_decl.preconditions.clone(),
                    fn_decl.postconditions.clone(),
                );
                evaluator.define(fn_decl.name.to_string(), closure);
            }
            Decl::Use(_) => {
                // Use declarations are already resolved by parse/resolve
            }
            _ => {
                // Skip other declarations for now
            }
        }
    }

    // If the source ends with an expression (not a declaration), evaluate it
    if !trimmed.is_empty() {
        let lines: Vec<&str> = trimmed.lines().collect();
        if let Some(last_line) = lines.last() {
            let last_trimmed = last_line.trim();
            // If it's not starting with let, #, use, or ╭─, try to evaluate as expression
            if !last_trimmed.starts_with("let ")
                && !last_trimmed.starts_with('#')
                && !last_trimmed.starts_with("use ")
                && !last_trimmed.starts_with("╭─")
                && !last_trimmed.starts_with("╰─")
                && !last_trimmed.is_empty()
            {
                // Try to parse and evaluate the last line as an expression
                if let Ok(expr) = parse_expr(last_trimmed) {
                    let resolved_expr = resolve_expr(expr);
                    if let Ok(value) = evaluator.eval(&resolved_expr) {
                        let type_name = value.type_name().to_string();
                        return Ok((value, type_name));
                    }
                }
            }
        }
    }

    // Return the last defined value, or Unit if no declarations
    match last_value {
        Some(v) => {
            let type_name = v.type_name().to_string();
            Ok((v, type_name))
        }
        None => Ok((Value::Unit, "Unit".to_string())),
    }
}

// ============================================================================
// API Types
// ============================================================================

/// Request to evaluate a Goth expression
#[derive(Debug, Deserialize)]
struct EvalRequest {
    /// Goth source code to evaluate
    source: String,
}

/// Response from evaluation
#[derive(Debug, Serialize)]
struct EvalResponse {
    /// Whether evaluation succeeded
    success: bool,
    /// Result value (if success)
    result: Option<String>,
    /// Result type (if success)
    result_type: Option<String>,
    /// Error message (if failure)
    error: Option<String>,
    /// Evaluation time in milliseconds
    time_ms: f64,
}

/// Request to parse Goth source
#[derive(Debug, Deserialize)]
struct ParseRequest {
    source: String,
}

/// Response from parsing
#[derive(Debug, Serialize)]
struct ParseResponse {
    success: bool,
    /// AST as JSON (if success)
    ast: Option<serde_json::Value>,
    error: Option<String>,
}

/// Request to type-check Goth source
#[derive(Debug, Deserialize)]
struct TypeCheckRequest {
    source: String,
}

/// Response from type-checking
#[derive(Debug, Serialize)]
struct TypeCheckResponse {
    success: bool,
    /// Inferred type (if success)
    inferred_type: Option<String>,
    /// Effects (if any)
    effects: Vec<String>,
    error: Option<String>,
}

/// Server info response
#[derive(Debug, Serialize)]
struct InfoResponse {
    name: String,
    version: String,
    endpoints: Vec<EndpointInfo>,
}

#[derive(Debug, Serialize)]
struct EndpointInfo {
    method: String,
    path: String,
    description: String,
}

// ============================================================================
// API Handlers
// ============================================================================

/// GET / - Server info and available endpoints
async fn info_handler() -> Json<InfoResponse> {
    Json(InfoResponse {
        name: "goth-server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        endpoints: vec![
            EndpointInfo {
                method: "GET".to_string(),
                path: "/".to_string(),
                description: "Server info and endpoints".to_string(),
            },
            EndpointInfo {
                method: "GET".to_string(),
                path: "/health".to_string(),
                description: "Health check".to_string(),
            },
            EndpointInfo {
                method: "POST".to_string(),
                path: "/api/eval".to_string(),
                description: "Evaluate Goth expression".to_string(),
            },
            EndpointInfo {
                method: "POST".to_string(),
                path: "/api/parse".to_string(),
                description: "Parse Goth source to AST".to_string(),
            },
            EndpointInfo {
                method: "POST".to_string(),
                path: "/api/typecheck".to_string(),
                description: "Type-check Goth source".to_string(),
            },
            EndpointInfo {
                method: "GET".to_string(),
                path: "/api/stdlib".to_string(),
                description: "List standard library functions".to_string(),
            },
        ],
    })
}

/// GET /health - Health check
async fn health_handler() -> &'static str {
    "ok"
}

/// POST /api/eval - Evaluate Goth expression
async fn eval_handler(Json(req): Json<EvalRequest>) -> Json<EvalResponse> {
    let start = Instant::now();

    match eval_source(&req.source) {
        Ok((value, type_name)) => Json(EvalResponse {
            success: true,
            result: Some(format!("{}", value)),
            result_type: Some(type_name),
            error: None,
            time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }),
        Err(e) => Json(EvalResponse {
            success: false,
            result: None,
            result_type: None,
            error: Some(e),
            time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }),
    }
}

/// POST /api/parse - Parse Goth source to AST
async fn parse_handler(Json(req): Json<ParseRequest>) -> Json<ParseResponse> {
    match parse_module(&req.source, "web") {
        Ok(module) => {
            // Serialize AST to JSON
            let ast_json = serde_json::to_value(&module).unwrap_or(serde_json::Value::Null);
            Json(ParseResponse {
                success: true,
                ast: Some(ast_json),
                error: None,
            })
        }
        Err(e) => Json(ParseResponse {
            success: false,
            ast: None,
            error: Some(format!("{}", e)),
        }),
    }
}

/// POST /api/typecheck - Type-check Goth source
async fn typecheck_handler(Json(req): Json<TypeCheckRequest>) -> Json<TypeCheckResponse> {
    // Parse first
    let module = match parse_module(&req.source, "web") {
        Ok(m) => m,
        Err(e) => {
            return Json(TypeCheckResponse {
                success: false,
                inferred_type: None,
                effects: vec![],
                error: Some(format!("Parse error: {}", e)),
            });
        }
    };

    // Resolve
    let resolved = resolve_module(module);

    // Type check
    let mut checker = goth_check::TypeChecker::new();
    match checker.check_module(&resolved) {
        Ok(types) => {
            // Format the type bindings as a string
            let type_strs: Vec<String> = types
                .iter()
                .map(|(name, ty)| format!("{}: {}", name, ty))
                .collect();
            let type_info = if type_strs.is_empty() {
                "()".to_string()
            } else {
                type_strs.join(", ")
            };

            Json(TypeCheckResponse {
                success: true,
                inferred_type: Some(type_info),
                effects: vec![],
                error: None,
            })
        }
        Err(e) => Json(TypeCheckResponse {
            success: false,
            inferred_type: None,
            effects: vec![],
            error: Some(format!("{}", e)),
        }),
    }
}

/// GET /api/stdlib - List standard library functions
async fn stdlib_handler() -> Json<HashMap<String, Vec<String>>> {
    let mut stdlib = HashMap::new();

    stdlib.insert(
        "math".to_string(),
        vec![
            "abs", "sqrt", "floor", "ceil", "round", "sin", "cos", "tan", "exp", "ln", "log10",
            "log2", "pow",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    stdlib.insert(
        "tensor".to_string(),
        vec![
            "sum", "prod", "len", "shape", "reverse", "concat", "dot", "norm", "matmul",
            "transpose", "iota", "range",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    stdlib.insert(
        "string".to_string(),
        vec![
            "toString", "chars", "strConcat", "strLen", "lines", "words", "bytes", "strEq",
            "startsWith", "endsWith", "contains",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    stdlib.insert(
        "io".to_string(),
        vec!["print", "println", "readLine", "readFile", "writeFile"]
            .into_iter()
            .map(String::from)
            .collect(),
    );

    stdlib.insert(
        "conversion".to_string(),
        vec![
            "toInt", "toFloat", "toBool", "toChar", "parseInt", "parseFloat",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    );

    Json(stdlib)
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    // Build router
    let mut app = Router::new()
        // Info endpoints
        .route("/", get(info_handler))
        .route("/health", get(health_handler))
        // API endpoints
        .route("/api/eval", post(eval_handler))
        .route("/api/parse", post(parse_handler))
        .route("/api/typecheck", post(typecheck_handler))
        .route("/api/stdlib", get(stdlib_handler))
        // Middleware
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http());

    // Add static file serving if directory specified
    if let Some(static_dir) = args.r#static {
        info!("Serving static files from: {:?}", static_dir);
        app = app.nest_service("/static", ServeDir::new(&static_dir));
        // Also serve index.html at root if it exists
        app = app.fallback_service(ServeDir::new(&static_dir));
    }

    // Build address
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("Invalid address");

    info!("Starting goth-server on http://{}", addr);
    info!("API endpoints:");
    info!("  POST /api/eval      - Evaluate expression");
    info!("  POST /api/parse     - Parse to AST");
    info!("  POST /api/typecheck - Type check");
    info!("  GET  /api/stdlib    - List stdlib functions");

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
