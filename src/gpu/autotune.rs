use faiss_next_sys as sys;
use std::ptr::null_mut;

use crate::error::{Error, Result};
use crate::gpu::resources::GpuResourcesProviderTrait;
use crate::index::{impl_index, IndexTrait};
use crate::macros::rc;

use super::cloner_options::GpuClonerOptionsTrait;

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

#[cfg(test)]
#[test]
fn test_index_gpu_ok() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    let _ = tracing_subscriber::fmt::try_init();
    use tracing::*;

    use crate::gpu::{cloner_options::GpuClonerOptions, standard_resources::StandardGpuResources};
    use crate::index::SearchParameters;
    use crate::index_factory::index_factory;
    use crate::metric::MetricType;
    use ndarray::{s, Array2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    let d = 128;
    let n = 1024 * 16;
    let k = 1;
    let ids = (0..n).map(|i| i as i64).collect::<Vec<_>>();
    let base = Array2::random([n, d], Uniform::new(-1.0f32, 1.0f32));
    let query = base.slice(s![42, ..]);
    let query = query
        .as_slice_memory_order()
        .ok_or(Error::NotStandardLayout)?;
    let base = base
        .as_slice_memory_order()
        .ok_or(Error::NotStandardLayout)?;
    let mut index = index_factory(128, "IDMap,Flat", MetricType::METRIC_L2)?;
    index.add(base, Some(ids))?;
    trace!(?index);
    let mut distances = vec![0.0f32; k as usize];
    let mut labels = vec![0i64; k as usize];

    for _ in 0..2 {
        let tm = std::time::Instant::now();
        index.search(
            query,
            1,
            &mut distances,
            &mut labels,
            Option::<SearchParameters>::None,
        )?;
        info!(
            "distances={:?}, labels={:?}, delta={:?}",
            distances,
            labels,
            tm.elapsed()
        );
    }

    let options = GpuClonerOptions::new()?;
    let gpu_index = IndexGpuImpl::new(
        [StandardGpuResources::new()?, StandardGpuResources::new()?],
        [0],
        &index,
        Some(options),
    )?;

    trace!(?gpu_index);
    for n in 0..2 {
        let tm = std::time::Instant::now();
        gpu_index.search(
            query,
            1,
            &mut distances,
            &mut labels,
            Option::<SearchParameters>::None,
        )?;
        if n % 128 * 16 == 0 {
            info!(
                "distances={:?}, labels={:?}, delta={:?}",
                distances,
                labels,
                tm.elapsed()
            );
        }
    }
    Ok(())
}
