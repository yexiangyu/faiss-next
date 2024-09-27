/// `faiss/c_api/gpu/DeviceUtils_c.h`
pub mod device_utils;
/// `faiss/c_api/gpu/GpuAutoTune_c.h``
pub mod gpu_autotune;
/// `faiss/c_api/gpu/GpuClonerOptions_c.h`
pub mod gpu_cloner_options;
/// `faiss/c_api/gpu/GpuIndex_c.h`
pub mod gpu_index;
/// `faiss/c_api/gpu/GpuIndicesOptions_c.h``
pub mod gpu_indices_options;
/// `faiss/c_api/gpu/GpuResources_c.h`
pub mod gpu_resources;
/// `faiss/c_api/gpu/StandardGpuResources_c.h``
pub mod standard_gpu_resources;

/// `faiss/gpu/GpuDistance.h`
pub mod gpu_distance;

pub mod prelude {
    pub use super::gpu_cloner_options::*;
    pub use super::gpu_index::*;
    pub use super::gpu_resources::FaissGpuResourcesProviderTrait;
    pub use super::standard_gpu_resources::*;
}
