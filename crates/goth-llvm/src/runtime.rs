//! Runtime function declarations for Goth
//!
//! These are external C functions that provide IO and tensor operations.

/// Generate LLVM IR declarations for runtime functions
pub fn emit_runtime_declarations() -> String {
    let mut out = String::new();

    // Standard C library functions
    out.push_str("; External declarations\n");
    out.push_str("declare i32 @printf(i8* nocapture readonly, ...) nounwind\n");
    out.push_str("declare i32 @puts(i8* nocapture readonly) nounwind\n");
    out.push_str("declare i8* @malloc(i64) nounwind\n");
    out.push_str("declare void @free(i8*) nounwind\n");
    out.push_str("declare i64 @atol(i8* nocapture readonly) nounwind\n");
    out.push_str("declare double @atof(i8* nocapture readonly) nounwind\n");
    out.push_str("\n");

    // Math library functions
    out.push_str("; Math functions\n");
    out.push_str("declare double @sqrt(double) nounwind readnone\n");
    out.push_str("declare double @floor(double) nounwind readnone\n");
    out.push_str("declare double @ceil(double) nounwind readnone\n");
    out.push_str("declare double @round(double) nounwind readnone\n");
    out.push_str("declare double @sin(double) nounwind readnone\n");
    out.push_str("declare double @cos(double) nounwind readnone\n");
    out.push_str("declare double @tan(double) nounwind readnone\n");
    out.push_str("declare double @asin(double) nounwind readnone\n");
    out.push_str("declare double @acos(double) nounwind readnone\n");
    out.push_str("declare double @atan(double) nounwind readnone\n");
    out.push_str("declare double @sinh(double) nounwind readnone\n");
    out.push_str("declare double @cosh(double) nounwind readnone\n");
    out.push_str("declare double @tanh(double) nounwind readnone\n");
    out.push_str("declare double @exp(double) nounwind readnone\n");
    out.push_str("declare double @log(double) nounwind readnone\n");
    out.push_str("declare double @log10(double) nounwind readnone\n");
    out.push_str("declare double @log2(double) nounwind readnone\n");
    out.push_str("declare double @fabs(double) nounwind readnone\n");
    out.push_str("declare double @copysign(double, double) nounwind readnone\n");
    out.push_str("declare double @pow(double, double) nounwind readnone\n");
    out.push_str("declare double @tgamma(double) nounwind readnone\n");
    out.push_str("declare double @lgamma(double) nounwind readnone\n");
    out.push_str("\n");

    // Goth runtime functions (implemented in runtime.c)
    out.push_str("; Goth runtime\n");
    out.push_str("declare void @goth_print_i64(i64)\n");
    out.push_str("declare void @goth_print_f64(double)\n");
    out.push_str("declare void @goth_print_bool(i1)\n");
    out.push_str("declare void @goth_print_newline()\n");
    out.push_str("declare i8* @goth_argv(i32)\n");
    out.push_str("declare i32 @goth_argc()\n");
    out.push_str("\n");

    // Tensor runtime (heap-allocated arrays)
    out.push_str("; Tensor operations\n");
    out.push_str("declare i8* @goth_iota(i64)\n");
    out.push_str("declare i8* @goth_range(i64, i64)\n");
    out.push_str("declare i64 @goth_sum_i64(i8*, i64)\n");
    out.push_str("declare double @goth_sum_f64(i8*, i64)\n");
    out.push_str("declare i64 @goth_prod_i64(i8*, i64)\n");
    out.push_str("declare i64 @goth_min_i64(i8*, i64)\n");
    out.push_str("declare i64 @goth_max_i64(i8*, i64)\n");
    out.push_str("declare i64 @goth_len(i8*)\n");
    out.push_str("declare i8* @goth_reverse(i8*, i64)\n");
    out.push_str("declare i64 @goth_index_i64(i8*, i64)\n");
    out.push_str("declare i8* @goth_map_i64(i8*, i8*, i64)\n");
    out.push_str("declare i8* @goth_filter_i64(i8*, i8*, i64)\n");
    out.push_str("\n");

    // Matrix/vector operations (F64)
    out.push_str("; Matrix/vector operations\n");
    out.push_str("declare double @goth_dot_f64(i8*, i8*, i64)\n");
    out.push_str("declare double @goth_norm_f64(i8*, i64)\n");
    out.push_str("declare i8* @goth_matmul_f64(i8*, i8*, i64, i64, i64)\n");
    out.push_str("declare i8* @goth_transpose_f64(i8*, i64, i64)\n");
    out.push_str("declare void @goth_print_array_f64(i8*, i64)\n");
    out.push_str("declare void @goth_print_array_i64(i8*, i64)\n");
    out.push_str("\n");

    out
}

/// Generate format string constants
pub fn emit_format_strings() -> String {
    let mut out = String::new();

    out.push_str("; Format strings\n");
    out.push_str("@.fmt_i64 = private unnamed_addr constant [5 x i8] c\"%ld\\0A\\00\"\n");
    out.push_str("@.fmt_f64 = private unnamed_addr constant [5 x i8] c\"%lf\\0A\\00\"\n");
    out.push_str("@.fmt_i64_no_nl = private unnamed_addr constant [4 x i8] c\"%ld\\00\"\n");
    out.push_str("@.fmt_f64_no_nl = private unnamed_addr constant [4 x i8] c\"%lf\\00\"\n");
    out.push_str("@.true_str = private unnamed_addr constant [5 x i8] c\"true\\00\"\n");
    out.push_str("@.false_str = private unnamed_addr constant [6 x i8] c\"false\\00\"\n");
    out.push_str("\n");

    out
}
