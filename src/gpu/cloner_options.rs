use faiss_next_sys as sys;
use std::ptr::null_mut;

use crate::error::Result;
use crate::gpu::indices_options::IndicesOptions;
use crate::macros::rc;

pub trait GpuClonerOptionsTrait {
    fn ptr(&self) -> *mut sys::FaissGpuClonerOptions;

    fn get_indices_options(&self) -> IndicesOptions {
        unsafe { sys::faiss_GpuClonerOptions_indicesOptions(self.ptr()) }
    }

    fn set_indices_options(&mut self, options: IndicesOptions) {
        unsafe { sys::faiss_GpuClonerOptions_set_indicesOptions(self.ptr(), options) }
    }

    fn use_float16_coarse_quantizer(&self) -> bool {
        unsafe { sys::faiss_GpuClonerOptions_useFloat16CoarseQuantizer(self.ptr()) != 0 }
    }

    fn set_use_float16_coarse_quantizer(&mut self, value: bool) {
        unsafe {
            sys::faiss_GpuClonerOptions_set_useFloat16CoarseQuantizer(self.ptr(), value as i32)
        }
    }

    fn use_float16(&self) -> bool {
        unsafe { sys::faiss_GpuClonerOptions_useFloat16(self.ptr()) != 0 }
    }

    fn set_use_float16(&mut self, value: bool) {
        unsafe { sys::faiss_GpuClonerOptions_set_useFloat16(self.ptr(), value as i32) }
    }

    fn use_precomputed(&self) -> bool {
        unsafe { sys::faiss_GpuClonerOptions_usePrecomputed(self.ptr()) != 0 }
    }

    fn reverse_vecs(&self) -> i64 {
        unsafe { sys::faiss_GpuClonerOptions_reserveVecs(self.ptr()) }
    }

    fn set_reverse_vecs(&self, value: i64) {
        unsafe { sys::faiss_GpuClonerOptions_set_reserveVecs(self.ptr(), value) }
    }

    fn store_transposed(&self) -> bool {
        unsafe { sys::faiss_GpuClonerOptions_storeTransposed(self.ptr()) != 0 }
    }
    fn set_store_transposed(&mut self, value: bool) {
        unsafe { sys::faiss_GpuClonerOptions_set_storeTransposed(self.ptr(), value as i32) }
    }

    fn verbose(&self) -> bool {
        unsafe { sys::faiss_GpuClonerOptions_verbose(self.ptr()) != 0 }
    }

    fn set_verbose(&mut self, value: bool) {
        unsafe { sys::faiss_GpuClonerOptions_set_verbose(self.ptr(), value as i32) }
    }
}

pub struct GpuClonerOptions {
    inner: *mut sys::FaissGpuClonerOptions,
}

impl Drop for GpuClonerOptions {
    fn drop(&mut self) {
        unsafe { sys::faiss_GpuClonerOptions_free(self.inner) }
    }
}

impl GpuClonerOptionsTrait for GpuClonerOptions {
    fn ptr(&self) -> *mut sys::FaissGpuClonerOptions {
        self.inner
    }
}

impl GpuClonerOptions {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_GpuClonerOptions_new(&mut inner) })?;
        Ok(Self { inner })
    }
}

pub struct GpuMultipleClonerOptions {
    inner: *mut sys::FaissGpuMultipleClonerOptions,
}

impl Drop for GpuMultipleClonerOptions {
    fn drop(&mut self) {
        unsafe { sys::faiss_GpuMultipleClonerOptions_free(self.inner) }
    }
}

impl GpuClonerOptionsTrait for GpuMultipleClonerOptions {
    fn ptr(&self) -> *mut sys::FaissGpuClonerOptions {
        self.inner as *mut _
    }
}

impl GpuMultipleClonerOptions {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_GpuMultipleClonerOptions_new(&mut inner) })?;
        Ok(Self { inner })
    }
    pub fn shard(&self) -> bool {
        unsafe { sys::faiss_GpuMultipleClonerOptions_shard(self.inner) != 0 }
    }

    pub fn set_shard(&mut self, value: bool) {
        unsafe { sys::faiss_GpuMultipleClonerOptions_set_shard(self.inner, value as i32) }
    }

    pub fn shard_type(&self) -> i32 {
        unsafe { sys::faiss_GpuMultipleClonerOptions_shard_type(self.inner) }
    }

    pub fn set_shard_type(&self, value: i32) {
        unsafe { sys::faiss_GpuMultipleClonerOptions_set_shard_type(self.inner, value) }
    }
}
