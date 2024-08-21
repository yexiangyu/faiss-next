use faiss_next_sys as ffi;

use crate::{
    error::*,
    index::{FaissIndexOwned, FaissIndexTrait},
    macros::impl_faiss_drop,
};
use std::ptr::null_mut;

#[derive(Debug)]
pub struct FaissGpuIndexConfig {
    inner: *mut ffi::FaissGpuIndexConfig,
}

impl FaissGpuIndexConfig {
    pub fn device(&self) -> i32 {
        unsafe { ffi::faiss_GpuIndexConfig_device(self.inner) }
    }
}

pub trait FaissGpuIndexTrait: FaissIndexTrait {
    fn to_cpu(&self) -> Result<FaissIndexOwned> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_index_gpu_to_cpu(self.inner(), &mut inner) })?;
        Ok(FaissIndexOwned { inner })
    }
}

#[derive(Debug)]
pub struct FaissGpuIndexOwned {
    pub inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissGpuIndexOwned, faiss_Index_free);
impl FaissIndexTrait for FaissGpuIndexOwned {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}
impl FaissGpuIndexTrait for FaissGpuIndexOwned {}
