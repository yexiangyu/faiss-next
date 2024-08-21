use std::ptr::null_mut;

use crate::{cuda::gpu_indices_options::FaissIndicesOptions, error::*, macros::*};
use faiss_next_sys as ffi;

pub trait FaissGpuClonerOptionsTrait {
    fn inner(&self) -> *mut ffi::FaissGpuClonerOptions;

    fn indices_options(&self) -> FaissIndicesOptions {
        unsafe { ffi::faiss_GpuClonerOptions_indicesOptions(self.inner()) }
    }

    fn set_indices_options(&mut self, options: FaissIndicesOptions) {
        unsafe { ffi::faiss_GpuClonerOptions_set_indicesOptions(self.inner(), options) }
    }
    fn use_float16_coarse_quantizer(&self) -> bool {
        unsafe { ffi::faiss_GpuClonerOptions_useFloat16CoarseQuantizer(self.inner()) > 0 }
    }
    fn set_use_float16_coarse_quantizer(&mut self, value: bool) {
        unsafe {
            ffi::faiss_GpuClonerOptions_set_useFloat16CoarseQuantizer(self.inner(), value as i32)
        }
    }
    fn use_float16(&self) -> bool {
        unsafe { ffi::faiss_GpuClonerOptions_useFloat16(self.inner()) > 0 }
    }
    fn set_use_float16(&mut self, value: bool) {
        unsafe { ffi::faiss_GpuClonerOptions_set_useFloat16(self.inner(), value as i32) }
    }
    fn use_precomputed(&self) -> bool {
        unsafe { ffi::faiss_GpuClonerOptions_usePrecomputed(self.inner()) > 0 }
    }
    fn set_use_precomputed(&mut self, value: bool) {
        unsafe { ffi::faiss_GpuClonerOptions_set_usePrecomputed(self.inner(), value as i32) }
    }
    fn reserve_vecs(&self) -> i64 {
        unsafe { ffi::faiss_GpuClonerOptions_reserveVecs(self.inner()) }
    }
    fn set_reserve_vecs(&mut self, value: i64) {
        unsafe { ffi::faiss_GpuClonerOptions_set_reserveVecs(self.inner(), value) }
    }
    fn store_transposed(&self) -> bool {
        unsafe { ffi::faiss_GpuClonerOptions_storeTransposed(self.inner()) > 0 }
    }
    fn set_store_transposed(&mut self, value: bool) {
        unsafe { ffi::faiss_GpuClonerOptions_set_storeTransposed(self.inner(), value as i32) }
    }
    fn verbose(&self) -> bool {
        unsafe { ffi::faiss_GpuClonerOptions_verbose(self.inner()) > 0 }
    }
    fn set_verbose(&mut self, value: bool) {
        unsafe { ffi::faiss_GpuClonerOptions_set_verbose(self.inner(), value as i32) }
    }
}

#[derive(Debug)]
pub struct FaissGpuClonerOptionsImpl {
    inner: *mut ffi::FaissGpuClonerOptions,
}
impl_faiss_drop!(FaissGpuClonerOptionsImpl, faiss_GpuClonerOptions_free);

impl FaissGpuClonerOptionsTrait for FaissGpuClonerOptionsImpl {
    fn inner(&self) -> *mut ffi::FaissGpuClonerOptions {
        self.inner
    }
}

impl FaissGpuClonerOptionsImpl {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_GpuClonerOptions_new(&mut inner) })?;
        Ok(FaissGpuClonerOptionsImpl { inner })
    }
}

impl Default for FaissGpuClonerOptionsImpl {
    fn default() -> Self {
        Self::new().expect("failed to create faiss gpu cloner options")
    }
}

#[derive(Debug)]
pub struct FaissGpuMultipleClonerOptions {
    pub inner: *mut ffi::FaissGpuClonerOptions,
}
impl_faiss_drop!(
    FaissGpuMultipleClonerOptions,
    faiss_GpuMultipleClonerOptions_free
);
impl FaissGpuMultipleClonerOptions {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_GpuMultipleClonerOptions_new(&mut inner) })?;
        Ok(FaissGpuMultipleClonerOptions { inner })
    }

    pub fn shard(&self) -> bool {
        unsafe { ffi::faiss_GpuMultipleClonerOptions_shard(self.inner) > 0 }
    }
    pub fn set_shard(&mut self, value: bool) {
        unsafe { ffi::faiss_GpuMultipleClonerOptions_set_shard(self.inner, value as i32) }
    }
    pub fn shared_type(&self) -> i32 {
        unsafe { ffi::faiss_GpuMultipleClonerOptions_shard_type(self.inner) }
    }
    pub fn set_shared_type(&mut self, val: i32) {
        unsafe { ffi::faiss_GpuMultipleClonerOptions_set_shard_type(self.inner, val) }
    }
}
impl Default for FaissGpuMultipleClonerOptions {
    fn default() -> Self {
        Self::new().expect("failed to create faiss gpu multiple cloner options")
    }
}
