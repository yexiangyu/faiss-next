use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};

pub trait GpuClonerOptionsTrait {
    fn inner(&self) -> *mut bindings::GpuClonerOptions;
}

pub struct GpuClonerOptions {
    pub(crate) inner: *mut bindings::GpuClonerOptions,
}

impl Drop for GpuClonerOptions {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_GpuClonerOptions_free(self.inner) }
        }
    }
}

impl GpuClonerOptions {
    pub fn new() -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe { bindings::faiss_GpuClonerOptions_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn set_indices_options(&mut self, options: i32) {
        unsafe { bindings::faiss_GpuClonerOptions_set_indicesOptions(self.inner, options) }
    }

    pub fn set_use_float16_coarse_quantizer(&mut self, value: bool) {
        unsafe {
            bindings::faiss_GpuClonerOptions_set_useFloat16CoarseQuantizer(self.inner, value as i32)
        }
    }

    pub fn set_use_float16(&mut self, value: bool) {
        unsafe { bindings::faiss_GpuClonerOptions_set_useFloat16(self.inner, value as i32) }
    }

    pub fn set_use_precomputed(&mut self, value: bool) {
        unsafe { bindings::faiss_GpuClonerOptions_set_usePrecomputed(self.inner, value as i32) }
    }

    pub fn set_reserve_vectors(&mut self, value: usize) {
        unsafe { bindings::faiss_GpuClonerOptions_set_reserveVecs(self.inner, value) }
    }

    pub fn set_store_transposed(&mut self, value: bool) {
        unsafe { bindings::faiss_GpuClonerOptions_set_storeTransposed(self.inner, value as i32) }
    }
}

impl GpuClonerOptionsTrait for GpuClonerOptions {
    fn inner(&self) -> *mut bindings::GpuClonerOptions {
        self.inner
    }
}
