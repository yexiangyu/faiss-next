pub use crate::bindings::FaissMetricType;
pub use crate::error::{Error, Result};
pub use crate::traits::{FaissIVFIndex, FaissIndex, FaissIndexBinary, FaissVectorTransform};

pub use crate::index::Index;
pub use crate::index_factory::index_factory;
pub use crate::index_flat::IndexFlat;

#[cfg(feature = "cuda")]
pub use crate::cuda::prelude::*;
