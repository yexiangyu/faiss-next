#![doc = include_str!("../README.md")]

pub mod cluster;
pub mod error;
pub mod index;
pub mod macros;
pub mod transformer;

pub mod prelude {
    pub use crate::cluster::{
        faiss_kmeans_clustering, ClusteringResult, FaissClustering, FaissClusteringIterationStats,
    };
    #[cfg(feature = "gpu")]
    pub use crate::index::gpu::GpuIndex;
    pub use crate::index::{FaissIndex, FaissMetricType, IDSelector, Index, IndexInner};
}
