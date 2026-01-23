//! MemRef dialect operations for memory management
//!
//! The memref dialect provides memory reference operations for working with
//! mutable, addressable memory:
//! - memref.alloc: Allocate memory
//! - memref.dealloc: Deallocate memory
//! - memref.load: Load value from memory
//! - memref.store: Store value to memory
//! - memref.copy: Copy memory regions
//! - memref.view: Create view into memory
//!
//! These operations are used after bufferization to lower tensors to memory.

use goth_ast::types::Type;
use crate::context::TextMlirContext;
use crate::types::type_to_mlir_string;
use crate::error::Result;

/// Memory space identifier
#[derive(Debug, Clone, Copy, Default)]
pub struct MemorySpace(pub u32);

impl std::fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            Ok(()) // Default memory space, don't print
        } else {
            write!(f, ", {}", self.0)
        }
    }
}

/// Emit memref.alloc to allocate memory
///
/// Allocates a memref with the given shape and element type.
pub fn emit_alloc(
    ctx: &mut TextMlirContext,
    dynamic_dims: &[&str],
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let dims_str = if dynamic_dims.is_empty() {
        String::new()
    } else {
        format!("({})", dynamic_dims.join(", "))
    };

    format!(
        "{}{} = memref.alloc{} : {}\n",
        ctx.indent_str(),
        ssa,
        dims_str,
        memref_type
    )
}

/// Emit memref.alloc with alignment
pub fn emit_alloc_aligned(
    ctx: &mut TextMlirContext,
    dynamic_dims: &[&str],
    memref_type: &str,
    alignment: u64,
) -> String {
    let ssa = ctx.fresh_ssa();

    let dims_str = if dynamic_dims.is_empty() {
        String::new()
    } else {
        format!("({})", dynamic_dims.join(", "))
    };

    format!(
        "{}{} = memref.alloc{} {{ alignment = {} }} : {}\n",
        ctx.indent_str(),
        ssa,
        dims_str,
        alignment,
        memref_type
    )
}

/// Emit memref.alloca for stack allocation
pub fn emit_alloca(
    ctx: &mut TextMlirContext,
    dynamic_dims: &[&str],
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let dims_str = if dynamic_dims.is_empty() {
        String::new()
    } else {
        format!("({})", dynamic_dims.join(", "))
    };

    format!(
        "{}{} = memref.alloca{} : {}\n",
        ctx.indent_str(),
        ssa,
        dims_str,
        memref_type
    )
}

/// Emit memref.dealloc to deallocate memory
pub fn emit_dealloc(ctx: &mut TextMlirContext, memref: &str) -> String {
    format!("{}memref.dealloc {} : memref<*>\n", ctx.indent_str(), memref)
}

/// Emit memref.dealloc with explicit type
pub fn emit_dealloc_typed(ctx: &mut TextMlirContext, memref: &str, memref_type: &str) -> String {
    format!(
        "{}memref.dealloc {} : {}\n",
        ctx.indent_str(),
        memref,
        memref_type
    )
}

/// Emit memref.load to load a value from memory
pub fn emit_load(
    ctx: &mut TextMlirContext,
    memref: &str,
    indices: &[&str],
    element_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let indices_str = if indices.is_empty() {
        String::new()
    } else {
        format!("[{}]", indices.join(", "))
    };

    format!(
        "{}{} = memref.load {}{} : memref<*x{}>\n",
        ctx.indent_str(),
        ssa,
        memref,
        indices_str,
        element_type
    )
}

/// Emit memref.load with explicit memref type
pub fn emit_load_typed(
    ctx: &mut TextMlirContext,
    memref: &str,
    indices: &[&str],
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let indices_str = if indices.is_empty() {
        String::new()
    } else {
        format!("[{}]", indices.join(", "))
    };

    format!(
        "{}{} = memref.load {}{} : {}\n",
        ctx.indent_str(),
        ssa,
        memref,
        indices_str,
        memref_type
    )
}

/// Emit memref.store to store a value to memory
pub fn emit_store(
    ctx: &mut TextMlirContext,
    value: &str,
    memref: &str,
    indices: &[&str],
    memref_type: &str,
) -> String {
    let indices_str = if indices.is_empty() {
        String::new()
    } else {
        format!("[{}]", indices.join(", "))
    };

    format!(
        "{}memref.store {}, {}{} : {}\n",
        ctx.indent_str(),
        value,
        memref,
        indices_str,
        memref_type
    )
}

/// Emit memref.copy to copy from source to destination
pub fn emit_copy(
    ctx: &mut TextMlirContext,
    source: &str,
    dest: &str,
    memref_type: &str,
) -> String {
    format!(
        "{}memref.copy {}, {} : {}, {}\n",
        ctx.indent_str(),
        source,
        dest,
        memref_type,
        memref_type
    )
}

/// Emit memref.dim to get dynamic dimension size
pub fn emit_dim(
    ctx: &mut TextMlirContext,
    memref: &str,
    index: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = memref.dim {}, {} : memref<*>\n",
        ctx.indent_str(),
        ssa,
        memref,
        index
    )
}

/// Emit memref.dim with explicit memref type
pub fn emit_dim_typed(
    ctx: &mut TextMlirContext,
    memref: &str,
    index: &str,
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = memref.dim {}, {} : {}\n",
        ctx.indent_str(),
        ssa,
        memref,
        index,
        memref_type
    )
}

/// Emit memref.rank to get memref rank
pub fn emit_rank(
    ctx: &mut TextMlirContext,
    memref: &str,
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = memref.rank {} : {}\n",
        ctx.indent_str(),
        ssa,
        memref,
        memref_type
    )
}

/// Emit memref.subview to create a view into a memref
pub fn emit_subview(
    ctx: &mut TextMlirContext,
    source: &str,
    offsets: &[&str],
    sizes: &[&str],
    strides: &[&str],
    source_type: &str,
    result_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let offsets_str = offsets.join(", ");
    let sizes_str = sizes.join(", ");
    let strides_str = strides.join(", ");

    format!(
        "{}{} = memref.subview {}[{}][{}][{}] : {} to {}\n",
        ctx.indent_str(),
        ssa,
        source,
        offsets_str,
        sizes_str,
        strides_str,
        source_type,
        result_type
    )
}

/// Emit memref.cast for memref type casting
pub fn emit_cast(
    ctx: &mut TextMlirContext,
    source: &str,
    source_type: &str,
    dest_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = memref.cast {} : {} to {}\n",
        ctx.indent_str(),
        ssa,
        source,
        source_type,
        dest_type
    )
}

/// Emit memref.reinterpret_cast for strided memref reshaping
pub fn emit_reinterpret_cast(
    ctx: &mut TextMlirContext,
    source: &str,
    offset: &str,
    sizes: &[&str],
    strides: &[&str],
    source_type: &str,
    result_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let sizes_str = sizes.join(", ");
    let strides_str = strides.join(", ");

    format!(
        "{}{} = memref.reinterpret_cast {} to offset: [{}], sizes: [{}], strides: [{}] : {} to {}\n",
        ctx.indent_str(),
        ssa,
        source,
        offset,
        sizes_str,
        strides_str,
        source_type,
        result_type
    )
}

/// Emit memref.expand_shape to add dimensions
pub fn emit_expand_shape(
    ctx: &mut TextMlirContext,
    source: &str,
    reassociation: &[Vec<i64>],
    source_type: &str,
    result_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let reassoc_str = reassociation
        .iter()
        .map(|group| {
            let indices: Vec<_> = group.iter().map(|i| i.to_string()).collect();
            format!("[{}]", indices.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "{}{} = memref.expand_shape {} [[{}]] : {} into {}\n",
        ctx.indent_str(),
        ssa,
        source,
        reassoc_str,
        source_type,
        result_type
    )
}

/// Emit memref.collapse_shape to remove dimensions
pub fn emit_collapse_shape(
    ctx: &mut TextMlirContext,
    source: &str,
    reassociation: &[Vec<i64>],
    source_type: &str,
    result_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    let reassoc_str = reassociation
        .iter()
        .map(|group| {
            let indices: Vec<_> = group.iter().map(|i| i.to_string()).collect();
            format!("[{}]", indices.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "{}{} = memref.collapse_shape {} [[{}]] : {} into {}\n",
        ctx.indent_str(),
        ssa,
        source,
        reassoc_str,
        source_type,
        result_type
    )
}

/// Emit memref.global for global memory declaration
pub fn emit_global(
    ctx: &mut TextMlirContext,
    name: &str,
    memref_type: &str,
    initial_value: Option<&str>,
    is_constant: bool,
) -> String {
    let sym_visibility = if is_constant { "private constant" } else { "private" };

    let init = initial_value
        .map(|v| format!(" = {}", v))
        .unwrap_or_default();

    format!(
        "{}memref.global {} @{} : {}{}\n",
        ctx.indent_str(),
        sym_visibility,
        name,
        memref_type,
        init
    )
}

/// Emit memref.get_global to access global memref
pub fn emit_get_global(
    ctx: &mut TextMlirContext,
    name: &str,
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = memref.get_global @{} : {}\n",
        ctx.indent_str(),
        ssa,
        name,
        memref_type
    )
}

/// Emit memref.assume_alignment for optimization hints
pub fn emit_assume_alignment(
    ctx: &mut TextMlirContext,
    memref: &str,
    alignment: u64,
) -> String {
    format!(
        "{}memref.assume_alignment {}, {} : memref<*>\n",
        ctx.indent_str(),
        memref,
        alignment
    )
}

/// Emit bufferization.to_memref for tensor to memref conversion
pub fn emit_to_memref(
    ctx: &mut TextMlirContext,
    tensor: &str,
    tensor_type: &str,
    memref_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = bufferization.to_memref {} : {} -> {}\n",
        ctx.indent_str(),
        ssa,
        tensor,
        tensor_type,
        memref_type
    )
}

/// Emit bufferization.to_tensor for memref to tensor conversion
pub fn emit_to_tensor(
    ctx: &mut TextMlirContext,
    memref: &str,
    memref_type: &str,
    tensor_type: &str,
) -> String {
    let ssa = ctx.fresh_ssa();

    format!(
        "{}{} = bufferization.to_tensor {} : {} -> {}\n",
        ctx.indent_str(),
        ssa,
        memref,
        memref_type,
        tensor_type
    )
}

/// Convert a Goth type to an MLIR memref type string
pub fn type_to_memref_string(ty: &Type) -> Result<String> {
    match ty {
        Type::Tensor(shape, elem) => {
            let elem_str = type_to_mlir_string(elem)?;
            let dims: Vec<String> = shape.0.iter().map(|d| {
                match d {
                    goth_ast::shape::Dim::Const(n) => n.to_string(),
                    goth_ast::shape::Dim::Var(_) => "?".to_string(),
                    goth_ast::shape::Dim::BinOp(_, _, _) => "?".to_string(),
                }
            }).collect();

            if dims.is_empty() {
                Ok(format!("memref<{}>", elem_str))
            } else {
                Ok(format!("memref<{}x{}>", dims.join("x"), elem_str))
            }
        }
        other => {
            let base = type_to_mlir_string(other)?;
            Ok(format!("memref<{}>", base))
        }
    }
}

/// Builder for constructing memref operations
pub struct MemRefBuilder {
    /// Shape dimensions (static or dynamic)
    dims: Vec<DimSize>,
    /// Element type string
    element_type: String,
    /// Memory space (0 = default)
    memory_space: MemorySpace,
}

/// Dimension size - static or dynamic
#[derive(Debug, Clone)]
pub enum DimSize {
    Static(i64),
    Dynamic,
}

impl MemRefBuilder {
    /// Create a new memref builder with element type
    pub fn new(element_type: &str) -> Self {
        Self {
            dims: Vec::new(),
            element_type: element_type.to_string(),
            memory_space: MemorySpace::default(),
        }
    }

    /// Add a static dimension
    pub fn with_static_dim(mut self, size: i64) -> Self {
        self.dims.push(DimSize::Static(size));
        self
    }

    /// Add a dynamic dimension
    pub fn with_dynamic_dim(mut self) -> Self {
        self.dims.push(DimSize::Dynamic);
        self
    }

    /// Set memory space
    pub fn with_memory_space(mut self, space: u32) -> Self {
        self.memory_space = MemorySpace(space);
        self
    }

    /// Build the memref type string
    pub fn build_type(&self) -> String {
        let dims_str: Vec<String> = self.dims.iter().map(|d| {
            match d {
                DimSize::Static(n) => n.to_string(),
                DimSize::Dynamic => "?".to_string(),
            }
        }).collect();

        let space_str = if self.memory_space.0 == 0 {
            String::new()
        } else {
            format!(", {}", self.memory_space.0)
        };

        if dims_str.is_empty() {
            format!("memref<{}{}>", self.element_type, space_str)
        } else {
            format!("memref<{}x{}{}>", dims_str.join("x"), self.element_type, space_str)
        }
    }

    /// Build an alloc operation with this type
    pub fn build_alloc(self, ctx: &mut TextMlirContext, dynamic_sizes: &[&str]) -> String {
        let ty = self.build_type();
        emit_alloc(ctx, dynamic_sizes, &ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goth_ast::types::PrimType;
    use goth_ast::shape::{Shape, Dim};

    fn tensor_i64_2d(m: u64, n: u64) -> Type {
        Type::Tensor(
            Shape(vec![Dim::Const(m), Dim::Const(n)]),
            Box::new(Type::Prim(PrimType::I64)),
        )
    }

    #[test]
    fn test_emit_alloc() {
        let mut ctx = TextMlirContext::new();
        let code = emit_alloc(&mut ctx, &[], "memref<4x8xf64>");
        assert!(code.contains("memref.alloc"));
        assert!(code.contains("memref<4x8xf64>"));
    }

    #[test]
    fn test_emit_alloc_dynamic() {
        let mut ctx = TextMlirContext::new();
        let code = emit_alloc(&mut ctx, &["%m", "%n"], "memref<?x?xf64>");
        assert!(code.contains("memref.alloc(%m, %n)"));
    }

    #[test]
    fn test_emit_load_store() {
        let mut ctx = TextMlirContext::new();

        let load = emit_load_typed(&mut ctx, "%mem", &["%i", "%j"], "memref<4x8xf64>");
        assert!(load.contains("memref.load"));

        let store = emit_store(&mut ctx, "%val", "%mem", &["%i", "%j"], "memref<4x8xf64>");
        assert!(store.contains("memref.store"));
        assert!(store.contains("%val, %mem"));
    }

    #[test]
    fn test_emit_copy() {
        let mut ctx = TextMlirContext::new();
        let code = emit_copy(&mut ctx, "%src", "%dst", "memref<4x8xf64>");
        assert!(code.contains("memref.copy"));
        assert!(code.contains("%src, %dst"));
    }

    #[test]
    fn test_emit_subview() {
        let mut ctx = TextMlirContext::new();
        let code = emit_subview(
            &mut ctx,
            "%mem",
            &["%off0", "%off1"],
            &["%sz0", "%sz1"],
            &["1", "1"],
            "memref<8x16xf64>",
            "memref<?x?xf64>",
        );
        assert!(code.contains("memref.subview"));
        assert!(code.contains("[%off0, %off1]"));
    }

    #[test]
    fn test_type_to_memref_string() {
        let ty = tensor_i64_2d(4, 8);
        let memref_str = type_to_memref_string(&ty).unwrap();
        assert_eq!(memref_str, "memref<4x8xi64>");
    }

    #[test]
    fn test_memref_builder() {
        let ty = MemRefBuilder::new("f64")
            .with_static_dim(4)
            .with_static_dim(8)
            .build_type();
        assert_eq!(ty, "memref<4x8xf64>");
    }

    #[test]
    fn test_memref_builder_dynamic() {
        let ty = MemRefBuilder::new("i32")
            .with_dynamic_dim()
            .with_static_dim(16)
            .build_type();
        assert_eq!(ty, "memref<?x16xi32>");
    }

    #[test]
    fn test_memref_builder_with_space() {
        let ty = MemRefBuilder::new("f32")
            .with_static_dim(4)
            .with_memory_space(1)
            .build_type();
        assert_eq!(ty, "memref<4xf32, 1>");
    }

    #[test]
    fn test_emit_global() {
        let mut ctx = TextMlirContext::new();
        let code = emit_global(&mut ctx, "weights", "memref<4x8xf64>", None, true);
        assert!(code.contains("memref.global"));
        assert!(code.contains("constant"));
        assert!(code.contains("@weights"));
    }

    #[test]
    fn test_emit_bufferization() {
        let mut ctx = TextMlirContext::new();

        let to_memref = emit_to_memref(&mut ctx, "%t", "tensor<4x8xf64>", "memref<4x8xf64>");
        assert!(to_memref.contains("bufferization.to_memref"));

        let to_tensor = emit_to_tensor(&mut ctx, "%m", "memref<4x8xf64>", "tensor<4x8xf64>");
        assert!(to_tensor.contains("bufferization.to_tensor"));
    }
}
