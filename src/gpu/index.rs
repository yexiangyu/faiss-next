use crate::index::IndexTrait;
use faiss_next_sys as sys;

pub struct GpuIndexConfig {
    inner: *mut sys::FaissGpuIndexConfig,
}

impl GpuIndexConfig {
    pub fn device(&self) -> i32 {
        unsafe { sys::faiss_GpuIndexConfig_device(self.inner) }
    }
}

pub trait GpuIndexTrait: IndexTrait {}
