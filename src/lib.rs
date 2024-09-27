#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

/// `c_api/AutoTune_c.h` not implemented yet
pub mod autotune;
/// `c_api/clone_index_c.h` implemented in `crate::mod::index`
pub mod clone_index;
/// c_api/Clustering.h implemented, need test case
pub mod clustering;
/// `Result<T>` and `Error`
pub mod error;
/// `c_api/impl/AuxIndexStructures_c.h`
pub mod impl_aux_index_structure;
/// `c_api/Index_c.h`
pub mod index;
/// `c_api/IndexBinary.h`
pub mod index_binary;
/// `c_api/index_factory_c.h`
pub mod index_factory;
/// `c_api/IndexFlat_c.h`
pub mod index_flat;
/// `c_api/index_io_c.h` implemented in `crate::mod::index`
pub mod index_io;
/// `c_api/IndexIVF_c.h`
pub mod index_ivf;
/// `c_api/IndexIVFFlat_c.h`
pub mod index_ivf_flat;
/// `c_api/IndexLSH_c.h`
pub mod index_lsh;
/// `c_api/IndexPreTransform_c.h`
pub mod index_pre_transform;
/// `c_api/IndexReplicas_c.h`
pub mod index_replicas;
/// `c_api/IndexScaleQuantizer_c.h`
pub mod index_scalar_quantizer;
/// `c_api/IndexShards_c.h`
pub mod index_shards;
/// macros used to generate some code
pub(crate) mod macros;
/// `c_api/MetaIndexes_c.h`
pub mod meta_indexes;
/// collected trait from modules
pub mod traits;
/// `c_api/utils/distances_c.h`, not all functions are implemented
pub mod utils_distances;
/// `c_api/VectorTransform_c.h`
pub mod vector_transform;

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
/// `CUDA`
pub mod cuda;

pub mod prelude {
    #[cfg(feature = "cuda")]
    pub use crate::cuda::prelude::*;
    pub use crate::impl_aux_index_structure::*;
    pub use crate::index::FaissMetricType;
    pub use crate::index_factory::faiss_index_factory;
    pub use crate::index_flat::*;
    pub use crate::traits::*;
}
