use faiss_next_sys as sys;
use tracing::trace;

use crate::{error::Result, macros::rc};

use super::resources::GpuResourcesProviderTrait;
pub struct StandardGpuResources {
    inner: *mut sys::FaissStandardGpuResources,
}

impl Drop for StandardGpuResources {
    fn drop(&mut self) {
        trace!("drop StandardGpuResources inner={:?}", self.inner);
        unsafe { sys::faiss_StandardGpuResources_free(self.inner) }
    }
}

impl StandardGpuResources {
    pub fn new() -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        crate::macros::rc!({ sys::faiss_StandardGpuResources_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        if size == 0 {
            rc!({ sys::faiss_StandardGpuResources_noTempMemory(self.inner) })?;
        } else {
            rc!({ sys::faiss_StandardGpuResources_setTempMemory(self.inner, size) })?;
        }
        Ok(())
    }

    pub fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        rc!({ sys::faiss_StandardGpuResources_setPinnedMemory(self.inner, size) })?;
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_default_stream(&mut self, device: i32, stream: sys::cudaStream_t) -> Result<()> {
        rc!({ sys::faiss_StandardGpuResources_setDefaultStream(self.inner, device, stream) })?;
        Ok(())
    }

    pub fn set_default_null_stream_all_devices(&mut self) -> Result<()> {
        rc!({ sys::faiss_StandardGpuResources_setDefaultNullStreamAllDevices(self.inner) })
    }
}

impl GpuResourcesProviderTrait for StandardGpuResources {
    fn ptr(&self) -> *mut sys::FaissGpuResourcesProvider {
        self.inner as *mut _
    }
}
