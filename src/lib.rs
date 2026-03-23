#![doc = include_str!("../README.md")]

pub mod clustering;
pub mod error;
pub mod factory;
pub mod idx;
pub mod index;
pub mod io;
pub mod metric;
pub mod pairwise;
pub mod result;

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod gpu;

pub use clustering::{Clustering, ClusteringParameters};
pub use error::{Error, Result};
pub use factory::{index_factory, IndexBuilder};
pub use idx::Idx;
pub use index::{
    BinaryIndex, Index, IndexBinary, IndexFlat, IndexFlat1D, IndexIDMap, IndexIDMap2, IndexIVF,
    IndexIVFFlat, IndexIVFScalarQuantizer, IndexImpl, IndexLSH, IndexPreTransform, IndexRefineFlat,
    IndexReplicas, IndexScalarQuantizer, IndexShards, IvfIndex, QuantizerType,
};
pub use io::{read_index, read_index_binary, write_index, write_index_binary};
pub use metric::MetricType;
pub use pairwise::{
    get_distance_compute_blas_database_bs, get_distance_compute_blas_query_bs,
    get_distance_compute_blas_threshold, inner_products, l2_sqr_ny, norm_l2_sqr, norms_l2,
    norms_l2_sqr, pairwise_l2_sqr, pairwise_l2_sqr_with_stride, renorm_l2,
    set_distance_compute_blas_database_bs, set_distance_compute_blas_query_bs,
    set_distance_compute_blas_threshold,
};
pub use result::{BinarySearchResult, RangeSearchResult, SearchResult};

#[cfg(all(target_os = "linux", feature = "cuda"))]
pub use gpu::{GpuIndexImpl, GpuResources};
