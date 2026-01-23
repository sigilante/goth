//! Linalg dialect operations for tensor linear algebra
//!
//! The linalg dialect provides high-level tensor operations:
//! - linalg.generic: General loop nest with affine indexing
//! - linalg.reduce: Reduction operations along axes
//! - linalg.matmul: Matrix multiplication
//! - linalg.dot: Dot product
//! - linalg.fill: Fill tensor with scalar value
//!
//! These operations work on tensors and will be bufferized to memrefs
//! before lowering to LLVM.

use goth_ast::types::Type;
use crate::context::TextMlirContext;
use crate::types::type_to_mlir_string;
use crate::error::Result;

/// Affine map specification for linalg.generic
#[derive(Debug, Clone)]
pub struct AffineMap {
    /// Number of dimensions
    pub dims: usize,
    /// Number of symbols
    pub symbols: usize,
    /// Result expressions (e.g., "d0", "d1", "d0 + d1")
    pub results: Vec<String>,
}

impl AffineMap {
    /// Create an identity map for n dimensions
    pub fn identity(n: usize) -> Self {
        let results = (0..n).map(|i| format!("d{}", i)).collect();
        Self {
            dims: n,
            symbols: 0,
            results,
        }
    }

    /// Create a projection map that selects specific dimensions
    pub fn projection(total_dims: usize, selected: &[usize]) -> Self {
        let results = selected.iter().map(|&i| format!("d{}", i)).collect();
        Self {
            dims: total_dims,
            symbols: 0,
            results,
        }
    }

    /// Create a reduction map (eliminates specified dimensions)
    pub fn reduction(total_dims: usize, reduce_dim: usize) -> Self {
        let results: Vec<_> = (0..total_dims)
            .filter(|&i| i != reduce_dim)
            .map(|i| format!("d{}", i))
            .collect();
        Self {
            dims: total_dims,
            symbols: 0,
            results,
        }
    }
}

impl std::fmt::Display for AffineMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims: Vec<_> = (0..self.dims).map(|i| format!("d{}", i)).collect();
        let symbols: Vec<_> = (0..self.symbols).map(|i| format!("s{}", i)).collect();

        write!(f, "affine_map<(")?;
        write!(f, "{}", dims.join(", "))?;
        if !symbols.is_empty() {
            write!(f, ")[{}", symbols.join(", "))?;
        }
        write!(f, ") -> (")?;
        write!(f, "{}", self.results.join(", "))?;
        write!(f, ")>")
    }
}

/// Iterator type for linalg.generic
#[derive(Debug, Clone, Copy)]
pub enum IteratorType {
    /// Parallel iterator (can be parallelized)
    Parallel,
    /// Reduction iterator (must be sequential)
    Reduction,
}

impl std::fmt::Display for IteratorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IteratorType::Parallel => write!(f, "parallel"),
            IteratorType::Reduction => write!(f, "reduction"),
        }
    }
}

/// Emit linalg.generic for map operations
///
/// Transforms each element of input tensor(s) through a computation body.
pub fn emit_generic(
    ctx: &mut TextMlirContext,
    inputs: &[&str],
    outputs: &[&str],
    indexing_maps: &[AffineMap],
    iterator_types: &[IteratorType],
    result_types: &[&Type],
) -> Result<String> {
    let results: Vec<String> = result_types
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let ssa = ctx.fresh_ssa();
            if i == 0 { ssa } else { ctx.fresh_ssa() }
        })
        .collect();

    let result_ssa = if results.len() == 1 {
        results[0].clone()
    } else {
        format!("({})", results.join(", "))
    };

    let inputs_str = inputs.join(", ");
    let outputs_str = outputs.join(", ");
    let maps_str = indexing_maps
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let iters_str = iterator_types
        .iter()
        .map(|t| format!("#linalg.iterator_type<{}>", t))
        .collect::<Vec<_>>()
        .join(", ");
    let types_str = result_types
        .iter()
        .map(|t| type_to_mlir_string(t))
        .collect::<Result<Vec<_>>>()?
        .join(", ");

    Ok(format!(
        "{}{} = linalg.generic {{\n\
         {}  indexing_maps = [{}],\n\
         {}  iterator_types = [{}]\n\
         {}}} ins({} : {}) outs({} : {})\n",
        ctx.indent_str(),
        result_ssa,
        ctx.indent_str(),
        maps_str,
        ctx.indent_str(),
        iters_str,
        ctx.indent_str(),
        inputs_str,
        types_str,
        outputs_str,
        types_str,
    ))
}

/// Start a linalg.generic operation
pub fn emit_generic_start(
    ctx: &mut TextMlirContext,
    inputs: &[&str],
    outputs: &[&str],
    indexing_maps: &[AffineMap],
    iterator_types: &[IteratorType],
    input_types: &[&Type],
    output_types: &[&Type],
) -> Result<String> {
    let maps_str = indexing_maps
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let iters_str = iterator_types
        .iter()
        .map(|t| format!("#linalg.iterator_type<{}>", t))
        .collect::<Vec<_>>()
        .join(", ");

    let in_types: Vec<_> = input_types
        .iter()
        .map(|t| type_to_mlir_string(t))
        .collect::<Result<Vec<_>>>()?;
    let out_types: Vec<_> = output_types
        .iter()
        .map(|t| type_to_mlir_string(t))
        .collect::<Result<Vec<_>>>()?;

    Ok(format!(
        "{}linalg.generic {{\n\
         {}  indexing_maps = [{}],\n\
         {}  iterator_types = [{}]\n\
         {}}} ins({} : {}) outs({} : {}) {{\n",
        ctx.indent_str(),
        ctx.indent_str(),
        maps_str,
        ctx.indent_str(),
        iters_str,
        ctx.indent_str(),
        inputs.join(", "),
        in_types.join(", "),
        outputs.join(", "),
        out_types.join(", "),
    ))
}

/// End a linalg.generic body
pub fn emit_generic_end(ctx: &mut TextMlirContext) -> String {
    format!("{}}} : {}\n", ctx.indent_str(), "/* result types */")
}

/// Emit linalg.yield to yield values from linalg body
pub fn emit_yield(ctx: &mut TextMlirContext, values: &[&str]) -> String {
    format!(
        "{}linalg.yield {}\n",
        ctx.indent_str(),
        values.join(", ")
    )
}

/// Emit linalg.reduce for reduction operations
pub fn emit_reduce(
    ctx: &mut TextMlirContext,
    input: &str,
    init: &str,
    dimensions: &[i64],
    combiner: &str,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let ty_str = type_to_mlir_string(result_type)?;
    let dims_str = dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Ok(format!(
        "{}{} = linalg.reduce ins({} : {}) outs({} : {}) dimensions = [{}] combiner = @{}\n",
        ctx.indent_str(),
        ssa,
        input,
        ty_str,
        init,
        ty_str,
        dims_str,
        combiner
    ))
}

/// Emit linalg.reduce_sum (sum reduction)
pub fn emit_reduce_sum(
    ctx: &mut TextMlirContext,
    input: &str,
    init: &str,
    axis: Option<i64>,
    input_type: &Type,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_ty = type_to_mlir_string(input_type)?;
    let out_ty = type_to_mlir_string(result_type)?;

    let dims = axis.map(|a| format!(" dimensions = [{}]", a)).unwrap_or_default();

    Ok(format!(
        "{}{} = linalg.reduce {{ arith.addf }} ins({} : {}) outs({} : {}){}\n",
        ctx.indent_str(),
        ssa,
        input,
        in_ty,
        init,
        out_ty,
        dims
    ))
}

/// Emit linalg.reduce_prod (product reduction)
pub fn emit_reduce_prod(
    ctx: &mut TextMlirContext,
    input: &str,
    init: &str,
    axis: Option<i64>,
    input_type: &Type,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_ty = type_to_mlir_string(input_type)?;
    let out_ty = type_to_mlir_string(result_type)?;

    let dims = axis.map(|a| format!(" dimensions = [{}]", a)).unwrap_or_default();

    Ok(format!(
        "{}{} = linalg.reduce {{ arith.mulf }} ins({} : {}) outs({} : {}){}\n",
        ctx.indent_str(),
        ssa,
        input,
        in_ty,
        init,
        out_ty,
        dims
    ))
}

/// Emit linalg.reduce_min (minimum reduction)
pub fn emit_reduce_min(
    ctx: &mut TextMlirContext,
    input: &str,
    init: &str,
    axis: Option<i64>,
    input_type: &Type,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_ty = type_to_mlir_string(input_type)?;
    let out_ty = type_to_mlir_string(result_type)?;

    let dims = axis.map(|a| format!(" dimensions = [{}]", a)).unwrap_or_default();

    Ok(format!(
        "{}{} = linalg.reduce {{ arith.minimumf }} ins({} : {}) outs({} : {}){}\n",
        ctx.indent_str(),
        ssa,
        input,
        in_ty,
        init,
        out_ty,
        dims
    ))
}

/// Emit linalg.reduce_max (maximum reduction)
pub fn emit_reduce_max(
    ctx: &mut TextMlirContext,
    input: &str,
    init: &str,
    axis: Option<i64>,
    input_type: &Type,
    result_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_ty = type_to_mlir_string(input_type)?;
    let out_ty = type_to_mlir_string(result_type)?;

    let dims = axis.map(|a| format!(" dimensions = [{}]", a)).unwrap_or_default();

    Ok(format!(
        "{}{} = linalg.reduce {{ arith.maximumf }} ins({} : {}) outs({} : {}){}\n",
        ctx.indent_str(),
        ssa,
        input,
        in_ty,
        init,
        out_ty,
        dims
    ))
}

/// Emit linalg.matmul for matrix multiplication
pub fn emit_matmul(
    ctx: &mut TextMlirContext,
    lhs: &str,
    rhs: &str,
    output: &str,
    lhs_type: &Type,
    rhs_type: &Type,
    output_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let lhs_ty = type_to_mlir_string(lhs_type)?;
    let rhs_ty = type_to_mlir_string(rhs_type)?;
    let out_ty = type_to_mlir_string(output_type)?;

    Ok(format!(
        "{}{} = linalg.matmul ins({}, {} : {}, {}) outs({} : {})\n",
        ctx.indent_str(),
        ssa,
        lhs,
        rhs,
        lhs_ty,
        rhs_ty,
        output,
        out_ty
    ))
}

/// Emit linalg.dot for dot product
pub fn emit_dot(
    ctx: &mut TextMlirContext,
    lhs: &str,
    rhs: &str,
    output: &str,
    lhs_type: &Type,
    rhs_type: &Type,
    output_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let lhs_ty = type_to_mlir_string(lhs_type)?;
    let rhs_ty = type_to_mlir_string(rhs_type)?;
    let out_ty = type_to_mlir_string(output_type)?;

    Ok(format!(
        "{}{} = linalg.dot ins({}, {} : {}, {}) outs({} : {})\n",
        ctx.indent_str(),
        ssa,
        lhs,
        rhs,
        lhs_ty,
        rhs_ty,
        output,
        out_ty
    ))
}

/// Emit linalg.fill to fill tensor with scalar
pub fn emit_fill(
    ctx: &mut TextMlirContext,
    value: &str,
    output: &str,
    value_type: &Type,
    output_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let val_ty = type_to_mlir_string(value_type)?;
    let out_ty = type_to_mlir_string(output_type)?;

    Ok(format!(
        "{}{} = linalg.fill ins({} : {}) outs({} : {})\n",
        ctx.indent_str(),
        ssa,
        value,
        val_ty,
        output,
        out_ty
    ))
}

/// Emit linalg.map for elementwise operations on tensors
pub fn emit_map(
    ctx: &mut TextMlirContext,
    inputs: &[&str],
    output: &str,
    body_fn: &str,
    input_types: &[&Type],
    output_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_types: Vec<_> = input_types
        .iter()
        .map(|t| type_to_mlir_string(t))
        .collect::<Result<Vec<_>>>()?;
    let out_ty = type_to_mlir_string(output_type)?;

    Ok(format!(
        "{}{} = linalg.map {{ {} }} ins({} : {}) outs({} : {})\n",
        ctx.indent_str(),
        ssa,
        body_fn,
        inputs.join(", "),
        in_types.join(", "),
        output,
        out_ty
    ))
}

/// Emit linalg.transpose for tensor transposition
pub fn emit_transpose(
    ctx: &mut TextMlirContext,
    input: &str,
    output: &str,
    permutation: &[i64],
    input_type: &Type,
    output_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_ty = type_to_mlir_string(input_type)?;
    let out_ty = type_to_mlir_string(output_type)?;
    let perm_str = permutation
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Ok(format!(
        "{}{} = linalg.transpose ins({} : {}) outs({} : {}) permutation = [{}]\n",
        ctx.indent_str(),
        ssa,
        input,
        in_ty,
        output,
        out_ty,
        perm_str
    ))
}

/// Emit linalg.broadcast for broadcasting scalars to tensors
pub fn emit_broadcast(
    ctx: &mut TextMlirContext,
    input: &str,
    output: &str,
    dimensions: &[i64],
    input_type: &Type,
    output_type: &Type,
) -> Result<String> {
    let ssa = ctx.fresh_ssa();
    let in_ty = type_to_mlir_string(input_type)?;
    let out_ty = type_to_mlir_string(output_type)?;
    let dims_str = dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Ok(format!(
        "{}{} = linalg.broadcast ins({} : {}) outs({} : {}) dimensions = [{}]\n",
        ctx.indent_str(),
        ssa,
        input,
        in_ty,
        output,
        out_ty,
        dims_str
    ))
}

/// Builder for constructing linalg.generic operations step-by-step
pub struct GenericBuilder {
    inputs: Vec<String>,
    outputs: Vec<String>,
    input_types: Vec<String>,
    output_types: Vec<String>,
    indexing_maps: Vec<AffineMap>,
    iterator_types: Vec<IteratorType>,
    body_lines: Vec<String>,
}

impl GenericBuilder {
    /// Create a new generic builder
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_types: Vec::new(),
            output_types: Vec::new(),
            indexing_maps: Vec::new(),
            iterator_types: Vec::new(),
            body_lines: Vec::new(),
        }
    }

    /// Add an input tensor
    pub fn add_input(mut self, value: &str, ty: &str) -> Self {
        self.inputs.push(value.to_string());
        self.input_types.push(ty.to_string());
        self
    }

    /// Add an output tensor
    pub fn add_output(mut self, value: &str, ty: &str) -> Self {
        self.outputs.push(value.to_string());
        self.output_types.push(ty.to_string());
        self
    }

    /// Add an indexing map
    pub fn add_indexing_map(mut self, map: AffineMap) -> Self {
        self.indexing_maps.push(map);
        self
    }

    /// Add an iterator type
    pub fn add_iterator(mut self, iter_type: IteratorType) -> Self {
        self.iterator_types.push(iter_type);
        self
    }

    /// Add a body line
    pub fn add_body_line(mut self, line: &str) -> Self {
        self.body_lines.push(line.to_string());
        self
    }

    /// Build the linalg.generic operation
    pub fn build(self, ctx: &mut TextMlirContext, result_count: usize) -> String {
        let results: Vec<_> = (0..result_count).map(|_| ctx.fresh_ssa()).collect();

        let result_str = if results.len() == 1 {
            results[0].clone()
        } else if results.is_empty() {
            String::new()
        } else {
            format!("({})", results.join(", "))
        };

        let maps_str = self
            .indexing_maps
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let iters_str = self
            .iterator_types
            .iter()
            .map(|t| format!("#linalg.iterator_type<{}>", t))
            .collect::<Vec<_>>()
            .join(", ");

        let prefix = if result_str.is_empty() {
            String::new()
        } else {
            format!("{} = ", result_str)
        };

        let mut code = format!(
            "{}{}linalg.generic {{\n",
            ctx.indent_str(),
            prefix
        );
        code.push_str(&format!(
            "{}  indexing_maps = [{}],\n",
            ctx.indent_str(),
            maps_str
        ));
        code.push_str(&format!(
            "{}  iterator_types = [{}]\n",
            ctx.indent_str(),
            iters_str
        ));
        code.push_str(&format!(
            "{}}} ins({} : {}) outs({} : {}) {{\n",
            ctx.indent_str(),
            self.inputs.join(", "),
            self.input_types.join(", "),
            self.outputs.join(", "),
            self.output_types.join(", ")
        ));

        ctx.push_indent();
        for line in &self.body_lines {
            code.push_str(&format!("{}{}\n", ctx.indent_str(), line));
        }
        ctx.pop_indent();

        code.push_str(&format!(
            "{}}} -> {}\n",
            ctx.indent_str(),
            self.output_types.join(", ")
        ));

        code
    }
}

impl Default for GenericBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_ast::shape::{Shape, Dim};

    fn tensor_f64(dims: &[u64]) -> Type {
        let shape = Shape(dims.iter().map(|&d| Dim::Const(d)).collect());
        Type::Tensor(shape, Box::new(Type::Prim(PrimType::F64)))
    }

    fn matrix_f64(m: u64, n: u64) -> Type {
        tensor_f64(&[m, n])
    }

    #[test]
    fn test_affine_map_identity() {
        let map = AffineMap::identity(2);
        assert_eq!(map.to_string(), "affine_map<(d0, d1) -> (d0, d1)>");
    }

    #[test]
    fn test_affine_map_projection() {
        let map = AffineMap::projection(3, &[0, 2]);
        assert_eq!(map.to_string(), "affine_map<(d0, d1, d2) -> (d0, d2)>");
    }

    #[test]
    fn test_affine_map_reduction() {
        let map = AffineMap::reduction(3, 1);
        assert_eq!(map.to_string(), "affine_map<(d0, d1, d2) -> (d0, d2)>");
    }

    #[test]
    fn test_emit_matmul() {
        let mut ctx = TextMlirContext::new();
        let lhs_ty = matrix_f64(4, 8);
        let rhs_ty = matrix_f64(8, 16);
        let out_ty = matrix_f64(4, 16);

        let code = emit_matmul(
            &mut ctx,
            "%a",
            "%b",
            "%c",
            &lhs_ty,
            &rhs_ty,
            &out_ty,
        ).unwrap();

        assert!(code.contains("linalg.matmul"));
        assert!(code.contains("%a, %b"));
        assert!(code.contains("%c"));
    }

    #[test]
    fn test_emit_dot() {
        let mut ctx = TextMlirContext::new();
        let vec_ty = tensor_f64(&[8]);
        let scalar_ty = Type::Prim(PrimType::F64);

        let code = emit_dot(
            &mut ctx,
            "%a",
            "%b",
            "%c",
            &vec_ty,
            &vec_ty,
            &scalar_ty,
        ).unwrap();

        assert!(code.contains("linalg.dot"));
        assert!(code.contains("%a, %b"));
    }

    #[test]
    fn test_emit_fill() {
        let mut ctx = TextMlirContext::new();
        let scalar_ty = Type::Prim(PrimType::F64);
        let tensor_ty = tensor_f64(&[4, 4]);

        let code = emit_fill(
            &mut ctx,
            "%zero",
            "%out",
            &scalar_ty,
            &tensor_ty,
        ).unwrap();

        assert!(code.contains("linalg.fill"));
        assert!(code.contains("%zero"));
        assert!(code.contains("%out"));
    }

    #[test]
    fn test_emit_transpose() {
        let mut ctx = TextMlirContext::new();
        let input_ty = matrix_f64(4, 8);
        let output_ty = matrix_f64(8, 4);

        let code = emit_transpose(
            &mut ctx,
            "%in",
            "%out",
            &[1, 0],
            &input_ty,
            &output_ty,
        ).unwrap();

        assert!(code.contains("linalg.transpose"));
        assert!(code.contains("permutation = [1, 0]"));
    }

    #[test]
    fn test_generic_builder() {
        let mut ctx = TextMlirContext::new();

        let code = GenericBuilder::new()
            .add_input("%in", "tensor<4xf64>")
            .add_output("%out", "tensor<4xf64>")
            .add_indexing_map(AffineMap::identity(1))
            .add_indexing_map(AffineMap::identity(1))
            .add_iterator(IteratorType::Parallel)
            .add_body_line("^bb0(%arg0: f64):")
            .add_body_line("  %0 = arith.mulf %arg0, %arg0 : f64")
            .add_body_line("  linalg.yield %0 : f64")
            .build(&mut ctx, 1);

        assert!(code.contains("linalg.generic"));
        assert!(code.contains("indexing_maps"));
        assert!(code.contains("iterator_types"));
        assert!(code.contains("parallel"));
    }

    #[test]
    fn test_emit_reduce_sum() {
        let mut ctx = TextMlirContext::new();
        let input_ty = tensor_f64(&[4, 8]);
        let output_ty = tensor_f64(&[4]);

        let code = emit_reduce_sum(
            &mut ctx,
            "%in",
            "%init",
            Some(1),
            &input_ty,
            &output_ty,
        ).unwrap();

        assert!(code.contains("linalg.reduce"));
        assert!(code.contains("arith.addf"));
        assert!(code.contains("dimensions = [1]"));
    }
}
