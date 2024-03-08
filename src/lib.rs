pub mod autotune;
pub mod clustering;
pub mod error;
pub mod index;
pub mod vector_transform;

macro_rules! rc {
    ($blk: block) => {
        match unsafe { $blk } {
            0 => Ok(()),
            r => {
                let e = crate::error::Error::from(r);
                tracing::error!("error={:?}", e);
                Err(crate::error::Error::from(r))
            }
        }
    };
}

pub(crate) use rc;

pub mod prelude {
    pub use crate::index::common::*;
    pub use crate::index::factory::*;
    pub use crate::index::flat::*;
    pub use crate::index::ivf::*;
    pub use crate::index::ivf_flat::*;
    pub use crate::index::lsh::*;
    pub use crate::index::meta::*;
    pub use crate::index::metric::*;
    pub use crate::index::pre_transform::*;
    pub use crate::index::replicas::*;
    pub use crate::index::shards::*;
    pub use crate::vector_transform::*;
}
