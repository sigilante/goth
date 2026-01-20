//! gothic - The Goth compiler
//!
//! Compiles Goth source files to native executables via LLVM.
//!
//! Usage:
//!   gothic input.goth              # Compile to ./input
//!   gothic input.goth -o output    # Compile to ./output
//!   gothic input.goth --emit-llvm  # Output LLVM IR only
//!   gothic input.goth --emit-mir   # Output MIR only

use clap::Parser;
use colored::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use goth_parse::prelude::*;
use goth_mir::lower_module;
use goth_llvm::emit_program;

#[derive(Parser, Debug)]
#[command(name = "gothic")]
#[command(author = "Goth Authors")]
#[command(version = "0.1.0")]
#[command(about = "The Goth compiler - compile Goth source to native executables")]
struct Args {
    /// Input source file (.goth)
    input: PathBuf,

    /// Output file (default: input name without extension)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Emit LLVM IR instead of compiling
    #[arg(long)]
    emit_llvm: bool,

    /// Emit MIR instead of compiling
    #[arg(long)]
    emit_mir: bool,

    /// Emit MLIR instead of compiling
    #[arg(long)]
    emit_mlir: bool,

    /// Keep intermediate files (.ll)
    #[arg(long)]
    keep_temps: bool,

    /// Optimization level (0-3)
    #[arg(short = 'O', default_value = "2")]
    opt_level: u8,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run_compiler(&args) {
        eprintln!("{}: {}", "error".red().bold(), e);
        std::process::exit(1);
    }
}

fn run_compiler(args: &Args) -> Result<(), String> {
    let input_name = args.input.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    if args.verbose {
        eprintln!("{} {}", "Compiling".green().bold(), args.input.display());
    }

    // Parse and resolve imports
    if args.verbose {
        eprintln!("  {} Parsing and resolving imports...", "→".cyan());
    }
    let module = load_file(&args.input)
        .map_err(|e| format!("Load error: {}", e))?;

    // Resolve (de Bruijn indices)
    if args.verbose {
        eprintln!("  {} Resolving...", "→".cyan());
    }
    let resolved = resolve_module(module);

    // Lower to MIR
    if args.verbose {
        eprintln!("  {} Lowering to MIR...", "→".cyan());
    }
    let mir = lower_module(&resolved)
        .map_err(|e| format!("MIR error: {:?}", e))?;

    // If --emit-mir, just output MIR and stop
    if args.emit_mir {
        let mir_output = goth_mir::print_program(&mir);
        if let Some(ref output) = args.output {
            fs::write(output, &mir_output)
                .map_err(|e| format!("Failed to write {}: {}", output.display(), e))?;
            if args.verbose {
                eprintln!("{} {}", "Wrote".green(), output.display());
            }
        } else {
            println!("{}", mir_output);
        }
        return Ok(());
    }

    // If --emit-mlir, output MLIR
    if args.emit_mlir {
        let mlir_output = goth_mlir::emit_program(&mir)
            .map_err(|e| format!("MLIR error: {:?}", e))?;
        if let Some(ref output) = args.output {
            fs::write(output, &mlir_output)
                .map_err(|e| format!("Failed to write {}: {}", output.display(), e))?;
            if args.verbose {
                eprintln!("{} {}", "Wrote".green(), output.display());
            }
        } else {
            println!("{}", mlir_output);
        }
        return Ok(());
    }

    // Emit LLVM IR
    if args.verbose {
        eprintln!("  {} Emitting LLVM IR...", "→".cyan());
    }
    let llvm_ir = emit_program(&mir)
        .map_err(|e| format!("LLVM error: {:?}", e))?;

    // If --emit-llvm, just output LLVM IR and stop
    if args.emit_llvm {
        if let Some(ref output) = args.output {
            fs::write(output, &llvm_ir)
                .map_err(|e| format!("Failed to write {}: {}", output.display(), e))?;
            if args.verbose {
                eprintln!("{} {}", "Wrote".green(), output.display());
            }
        } else {
            println!("{}", llvm_ir);
        }
        return Ok(());
    }

    // Compile to native executable
    let output_path = args.output.clone()
        .unwrap_or_else(|| PathBuf::from(input_name));

    compile_llvm_ir(&llvm_ir, &output_path, args)?;

    if args.verbose {
        eprintln!("{} {}", "Compiled".green().bold(), output_path.display());
    }

    Ok(())
}

/// Compile LLVM IR to native executable using clang
fn compile_llvm_ir(llvm_ir: &str, output: &Path, args: &Args) -> Result<(), String> {
    // Write LLVM IR to temp file
    let ll_path = output.with_extension("ll");
    fs::write(&ll_path, llvm_ir)
        .map_err(|e| format!("Failed to write {}: {}", ll_path.display(), e))?;

    // Write runtime to temp file
    let runtime_path = output.with_extension("runtime.c");
    fs::write(&runtime_path, GOTH_RUNTIME_C)
        .map_err(|e| format!("Failed to write {}: {}", runtime_path.display(), e))?;

    if args.verbose {
        eprintln!("  {} Compiling with clang...", "→".cyan());
    }

    // Compile with clang
    let opt_flag = format!("-O{}", args.opt_level.min(3));
    let status = Command::new("clang")
        .args([
            &opt_flag,
            "-o", output.to_str().unwrap(),
            ll_path.to_str().unwrap(),
            runtime_path.to_str().unwrap(),
            "-lm",  // Link math library
        ])
        .stderr(if args.verbose { Stdio::inherit() } else { Stdio::null() })
        .status()
        .map_err(|e| format!("Failed to run clang: {}. Is clang installed?", e))?;

    if !status.success() {
        return Err(format!("clang failed with exit code: {:?}", status.code()));
    }

    // Clean up temp files unless --keep-temps
    if !args.keep_temps {
        let _ = fs::remove_file(&ll_path);
        let _ = fs::remove_file(&runtime_path);
    }

    Ok(())
}

/// Embedded C runtime for Goth
const GOTH_RUNTIME_C: &str = r#"
// Goth Runtime Library
// Provides IO and tensor operations for compiled Goth programs

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Global argc/argv storage
static int g_argc = 0;
static char** g_argv = NULL;

void goth_init(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
}

int goth_argc() {
    return g_argc;
}

char* goth_argv(int i) {
    if (i >= 0 && i < g_argc) {
        return g_argv[i];
    }
    return "";
}

// Print functions
void goth_print_i64(int64_t x) {
    printf("%ld", x);
}

void goth_print_f64(double x) {
    printf("%g", x);
}

void goth_print_bool(int x) {
    printf("%s", x ? "true" : "false");
}

void goth_print_newline() {
    printf("\n");
}

// Tensor representation:
// Heap-allocated array with length prefix
// [len: i64][elem0][elem1]...

// Create iota array [0, 1, 2, ..., n-1]
int64_t* goth_iota(int64_t n) {
    int64_t* arr = (int64_t*)malloc((n + 1) * sizeof(int64_t));
    arr[0] = n;  // Store length
    for (int64_t i = 0; i < n; i++) {
        arr[i + 1] = i;
    }
    return arr;
}

// Create range array [start, start+1, ..., end-1]
int64_t* goth_range(int64_t start, int64_t end) {
    int64_t n = end > start ? end - start : 0;
    int64_t* arr = (int64_t*)malloc((n + 1) * sizeof(int64_t));
    arr[0] = n;
    for (int64_t i = 0; i < n; i++) {
        arr[i + 1] = start + i;
    }
    return arr;
}

// Get array length
int64_t goth_len(int64_t* arr) {
    return arr[0];
}

// Sum of i64 array
int64_t goth_sum_i64(int64_t* arr, int64_t len) {
    int64_t sum = 0;
    for (int64_t i = 0; i < len; i++) {
        sum += arr[i + 1];
    }
    return sum;
}

// Sum of f64 array
double goth_sum_f64(double* arr, int64_t len) {
    double sum = 0.0;
    int64_t* p = (int64_t*)arr;
    double* data = (double*)(p + 1);
    for (int64_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}

// Product of i64 array
int64_t goth_prod_i64(int64_t* arr, int64_t len) {
    int64_t prod = 1;
    for (int64_t i = 0; i < len; i++) {
        prod *= arr[i + 1];
    }
    return prod;
}

// Index into i64 array
int64_t goth_index_i64(int64_t* arr, int64_t idx) {
    // Bounds check could go here
    return arr[idx + 1];
}

// Reverse array
int64_t* goth_reverse(int64_t* arr, int64_t len) {
    int64_t* result = (int64_t*)malloc((len + 1) * sizeof(int64_t));
    result[0] = len;
    for (int64_t i = 0; i < len; i++) {
        result[i + 1] = arr[len - i];
    }
    return result;
}

// Min of i64 array
int64_t goth_min_i64(int64_t* arr, int64_t len) {
    if (len == 0) return 0;
    int64_t min = arr[1];
    for (int64_t i = 1; i < len; i++) {
        if (arr[i + 1] < min) min = arr[i + 1];
    }
    return min;
}

// Max of i64 array
int64_t goth_max_i64(int64_t* arr, int64_t len) {
    if (len == 0) return 0;
    int64_t max = arr[1];
    for (int64_t i = 1; i < len; i++) {
        if (arr[i + 1] > max) max = arr[i + 1];
    }
    return max;
}

// ============ String Operations ============

// String length (C-string)
int64_t goth_strlen(char* s) {
    return (int64_t)strlen(s);
}

// String concatenation (allocates new string)
char* goth_strconcat(char* a, char* b) {
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    char* result = (char*)malloc(len_a + len_b + 1);
    strcpy(result, a);
    strcat(result, b);
    return result;
}

// Print string
void goth_print_str(char* s) {
    printf("%s", s);
}

// Print string with newline
void goth_println_str(char* s) {
    printf("%s\n", s);
}

// ============ File I/O ============

// Read file contents as string (returns NULL on error)
char* goth_read_file(char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* contents = (char*)malloc(size + 1);
    if (!contents) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(contents, 1, size, f);
    contents[read] = '\0';
    fclose(f);

    return contents;
}

// Write string to file (returns 0 on success, -1 on error)
int64_t goth_write_file(char* path, char* contents) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    size_t len = strlen(contents);
    size_t written = fwrite(contents, 1, len, f);
    fclose(f);

    return (written == len) ? 0 : -1;
}

// Read line from stdin (allocates new string)
char* goth_read_line() {
    char* line = NULL;
    size_t len = 0;
    ssize_t read = getline(&line, &len, stdin);

    if (read == -1) {
        free(line);
        return strdup("");
    }

    // Remove trailing newline
    if (read > 0 && line[read - 1] == '\n') {
        line[read - 1] = '\0';
    }

    return line;
}

// ============ Memory Management ============

// Free allocated memory
void goth_free(void* ptr) {
    free(ptr);
}

// Allocate memory
void* goth_alloc(int64_t size) {
    return malloc((size_t)size);
}

// ============ Higher-Order Functions ============

// Function pointer type for i64 -> i64
typedef int64_t (*fn_i64_i64)(int64_t);

// Map: apply function to each element of array
int64_t* goth_map_i64(int64_t* arr, fn_i64_i64 fn, int64_t len) {
    int64_t* result = (int64_t*)malloc((len + 1) * sizeof(int64_t));
    result[0] = len;
    for (int64_t i = 0; i < len; i++) {
        result[i + 1] = fn(arr[i + 1]);
    }
    return result;
}

// Function pointer type for i64 -> bool (i64)
typedef int64_t (*fn_i64_bool)(int64_t);

// Filter: keep elements where predicate returns true
int64_t* goth_filter_i64(int64_t* arr, fn_i64_bool pred, int64_t len) {
    // First pass: count matching elements
    int64_t count = 0;
    for (int64_t i = 0; i < len; i++) {
        if (pred(arr[i + 1])) count++;
    }

    // Second pass: copy matching elements
    int64_t* result = (int64_t*)malloc((count + 1) * sizeof(int64_t));
    result[0] = count;
    int64_t j = 1;
    for (int64_t i = 0; i < len; i++) {
        if (pred(arr[i + 1])) {
            result[j++] = arr[i + 1];
        }
    }
    return result;
}

// ============ Matrix/Vector Operations (F64) ============

#include <math.h>

// Dot product: [n]F64 → [n]F64 → F64
double goth_dot_f64(double* a, double* b, int64_t len) {
    double sum = 0.0;
    for (int64_t i = 0; i < len; i++) {
        sum += a[i + 1] * b[i + 1];
    }
    return sum;
}

// Vector norm (Euclidean): [n]F64 → F64
double goth_norm_f64(double* arr, int64_t len) {
    double sum = 0.0;
    for (int64_t i = 0; i < len; i++) {
        double v = arr[i + 1];
        sum += v * v;
    }
    return sqrt(sum);
}

// Matrix multiplication: [m n]F64 → [n p]F64 → [m p]F64
// Matrices stored row-major: arr[0] = total_elements, then data
// For [m n] matrix: index (i, j) = arr[1 + i*n + j]
double* goth_matmul_f64(double* a, double* b, int64_t m, int64_t n, int64_t p) {
    // Result is [m p] matrix
    int64_t result_size = m * p;
    double* result = (double*)malloc((result_size + 1) * sizeof(double));
    int64_t* len_ptr = (int64_t*)result;
    len_ptr[0] = result_size;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; k++) {
                // a[i, k] * b[k, j]
                sum += a[1 + i*n + k] * b[1 + k*p + j];
            }
            result[1 + i*p + j] = sum;
        }
    }
    return result;
}

// Matrix transpose: [m n]F64 → [n m]F64
double* goth_transpose_f64(double* arr, int64_t m, int64_t n) {
    int64_t size = m * n;
    double* result = (double*)malloc((size + 1) * sizeof(double));
    int64_t* len_ptr = (int64_t*)result;
    len_ptr[0] = size;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            // Transpose: result[j, i] = arr[i, j]
            result[1 + j*m + i] = arr[1 + i*n + j];
        }
    }
    return result;
}

// Print f64 array (for debugging)
void goth_print_array_f64(double* arr, int64_t len) {
    printf("[");
    for (int64_t i = 0; i < len; i++) {
        if (i > 0) printf(", ");
        printf("%g", arr[i + 1]);
    }
    printf("]");
}

// Print i64 array (for debugging)
void goth_print_array_i64(int64_t* arr, int64_t len) {
    printf("[");
    for (int64_t i = 0; i < len; i++) {
        if (i > 0) printf(", ");
        printf("%ld", arr[i + 1]);
    }
    printf("]");
}
"#;
