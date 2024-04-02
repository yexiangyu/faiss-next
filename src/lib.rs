pub mod autotune;
pub mod aux_index_structures;
pub mod clone_index;
pub mod clustering;
pub mod error;
pub mod index;
pub mod index_factory;
pub mod index_flat;
pub mod index_io;
pub mod index_ivf;
pub mod index_ivf_flat;
pub mod index_lsh;
pub mod index_pre_transform;
pub mod index_replicas;
pub mod index_scalar_quantizer;
pub mod index_shards;
pub mod macros;
pub mod meta_indexes;
pub mod metric;
pub mod vector_transform;

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod prelude {
    pub use super::aux_index_structures::DistanceComputerTrait;
    pub use super::aux_index_structures::IDSelectorTrait;
    pub use super::index::IndexTrait;
    pub use super::index_flat::IndexFlatTrait;
    pub use super::index_ivf::IndexIVFTrait;
    pub use super::index_scalar_quantizer::IndexScalarQuantizerTrait;
    pub use super::vector_transform::{LinearTransformTrait, VectorTransformTrait};
}
