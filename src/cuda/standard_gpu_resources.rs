use std::ptr;

use crate::bindings;
use crate::cuda::gpu_resources::GpuResourcesProvider;
use crate::error::{check_return_code, Result};

pub struct StandardGpuResources {
    pub(crate) inner: *mut bindings::StandardGpuResources,
}

impl Drop for StandardGpuResources {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_StandardGpuResources_free(self.inner) }
        }
    }
}

impl StandardGpuResources {
    pub fn new() -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe { bindings::faiss_StandardGpuResources_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_StandardGpuResources_setTempMemory(self.inner, size)
        })
    }

    pub fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_StandardGpuResources_setPinnedMemory(self.inner, size)
        })
    }
}

impl GpuResourcesProvider for StandardGpuResources {
    fn inner(&self) -> *mut bindings::GpuResources {
        self.inner as *mut _
    }
}
