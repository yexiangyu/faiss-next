use faiss_next_sys as sys;
use std::ptr::null_mut;
use tracing::trace;

use crate::error::Result;
use crate::gpu::indices_options::IndicesOptions;
use crate::macros::rc;

pub trait GpuClonerOptionsTrait {
    fn ptr(&self) -> *mut sys::FaissGpuClonerOptions;

    fn indices_options(&self) -> IndicesOptions {
        unsafe { sys::faiss_GpuClonerOptions_indicesOptions(self.ptr()) }.into()
    }

    fn set_indices_options(&mut self, options: IndicesOptions) {
        unsafe { sys::faiss_GpuClonerOptions_set_indicesOptions(self.ptr(), options.into()) }
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

impl std::fmt::Debug for GpuClonerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuClonerOptions")
            .field("inner", &self.ptr())
            .field("indices_options", &self.indices_options())
            .field(
                "use_float16_coarse_quantizer",
                &self.use_float16_coarse_quantizer(),
            )
            .field("use_float16", &self.use_float16())
            .field("use_precomputed", &self.use_precomputed())
            .field("reverse_vecs", &self.reverse_vecs())
            .field("store_transposed", &self.store_transposed())
            .field("verbose", &self.verbose())
            .finish()
    }
}

impl Drop for GpuClonerOptions {
    fn drop(&mut self) {
        trace!(?self, "drop");
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
        let r = Self { inner };
        trace!(?r, "new");
        Ok(r)
    }
}

pub struct GpuMultipleClonerOptions {
    inner: *mut sys::FaissGpuMultipleClonerOptions,
}

impl std::fmt::Debug for GpuMultipleClonerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMultipleClonerOptions")
            .field("inner", &self.inner)
            .field("indices_options", &self.indices_options())
            .field(
                "use_float16_coarse_quantizer",
                &self.use_float16_coarse_quantizer(),
            )
            .field("use_float16", &self.use_float16())
            .field("use_precomputed", &self.use_precomputed())
            .field("reverse_vecs", &self.reverse_vecs())
            .field("store_transposed", &self.store_transposed())
            .field("verbose", &self.verbose())
            .field("shard", &self.shard())
            .field("shard_type", &self.shard_type())
            .finish()
    }
}

impl Drop for GpuMultipleClonerOptions {
    fn drop(&mut self) {
        trace!(?self, "drop");
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
        let r = Self { inner };
        trace!(?r, "new");
        Ok(r)
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
