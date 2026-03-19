use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};

pub trait GpuResourcesProvider {
    fn inner(&self) -> *mut bindings::GpuResources;
}

pub struct GpuResources {
    pub(crate) inner: *mut bindings::GpuResources,
}

impl Drop for GpuResources {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_GpuResources_free(self.inner) }
        }
    }
}

impl GpuResourcesProvider for GpuResources {
    fn inner(&self) -> *mut bindings::GpuResources {
        self.inner
    }
}
