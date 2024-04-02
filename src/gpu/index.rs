use faiss_next_sys as sys;

use crate::error::{Error, Result};
use crate::gpu::cloner_options::GpuClonerOptionsTrait;
use crate::gpu::resources::GpuResourcesProviderTrait;
use crate::index::{impl_index, IndexTrait};
use crate::macros::rc;
use std::ptr::null_mut;

pub struct GpuIndexConfig {
    inner: *mut sys::FaissGpuIndexConfig,
}

impl GpuIndexConfig {
    pub fn device(&self) -> i32 {
        unsafe { sys::faiss_GpuIndexConfig_device(self.inner) }
    }
}

pub struct IndexGpuImpl {
    inner: *mut sys::FaissIndex,
}

impl_index!(IndexGpuImpl);

impl IndexGpuImpl {
    pub fn new<P, O>(
        providers: impl AsRef<[P]>,
        devices: impl AsRef<[i32]>,
        index: &impl IndexTrait,
        options: Option<O>,
    ) -> Result<Self>
    where
        P: GpuResourcesProviderTrait,
        O: GpuClonerOptionsTrait,
    {
        let mut inner = null_mut();
        let providers = providers
            .as_ref()
            .iter()
            .map(|p| p.ptr())
            .collect::<Vec<_>>();
        let devices = devices.as_ref();
        match devices.len() {
            0 => return Err(Error::InvalidGpuDevices),
            1 => {
                let device = devices[0];
                match options {
                    Some(options) => {
                        rc!({
                            sys::faiss_index_cpu_to_gpu_with_options(
                                providers[0],
                                device,
                                index.ptr(),
                                options.ptr(),
                                &mut inner,
                            )
                        })?;
                    }
                    None => rc!({
                        sys::faiss_index_cpu_to_gpu(providers[0], device, index.ptr(), &mut inner)
                    })?,
                }
            }
            _ => match options {
                Some(options) => {
                    rc!({
                        sys::faiss_index_cpu_to_gpu_multiple_with_options(
                            providers.as_ptr(),
                            providers.len(),
                            devices.as_ptr(),
                            devices.len(),
                            index.ptr(),
                            options.ptr(),
                            &mut inner,
                        )
                    })?;
                }
                None => rc!({
                    sys::faiss_index_cpu_to_gpu_multiple(
                        providers.as_ptr(),
                        devices.as_ptr(),
                        devices.len(),
                        index.ptr(),
                        &mut inner,
                    )
                })?,
            },
        };
        Ok(Self { inner })
    }
}
