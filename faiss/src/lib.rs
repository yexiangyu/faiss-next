pub(crate) mod sys;
use std::ffi::CString;
use std::ptr::{addr_of_mut, null_mut};
use std::time::Instant;
use tracing::trace;

pub use sys::FaissMetricType;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    NulErr(#[from] std::ffi::NulError),
    #[error("faiss error,code={},message={}", .code, .message)]
    Faiss { code: i32, message: String },
}
pub type Result<T> = std::result::Result<T, Error>;

macro_rules! faiss_rc {
    ($blk: block) => {
        unsafe {
            let rc = { $blk };
            match rc {
                0 => Ok(()),
                _ => {
                    let message = sys::faiss_get_last_error();
                    let message = std::ffi::CStr::from_ptr(message);
                    let message = message
                        .to_str()
                        .expect("failed to convert to utf-8")
                        .to_string();
                    Err(Error::Faiss {
                        code: rc as i32,
                        message,
                    })
                }
            }
        }
    };
}

#[derive(Debug)]
pub struct SearchResult {
    pub labels: Vec<i64>,
    pub distances: Vec<f32>,
}

pub trait Index {
    fn inner(&self) -> *mut sys::FaissIndex;

    fn add<T: AsRef<[f32]> + ?Sized>(&mut self, x: &T) -> Result<()> {
        let x = x.as_ref();
        faiss_rc! {{sys::faiss_Index_add(self.inner(), (x.len() / self.d())as sys::idx_t, x.as_ptr())}}?;
        trace!("add: index={:?}, x.len={}", self.inner(), x.as_ref().len());
        Ok(())
    }

    fn search<T: AsRef<[f32]> + ?Sized>(&self, x: &T, k: usize) -> Result<SearchResult> {
        let x = x.as_ref();
        let n = (x.len() / self.d()) as sys::idx_t;
        let mut labels = vec![0 as sys::idx_t; n as usize * k];
        let mut distances = vec![0.0f32; n as usize * k];
        let tm = Instant::now();
        faiss_rc!({
            {
                sys::faiss_Index_search(
                    self.inner(),
                    n,
                    x.as_ptr(),
                    k as sys::idx_t,
                    distances.as_mut_ptr(),
                    labels.as_mut_ptr(),
                )
            }
        })?;
        trace!(
            "search index d={}, n={}, k={}, tm={:?}",
            self.d(),
            n,
            k,
            tm.elapsed()
        );
        Ok(SearchResult { labels, distances })
    }

    fn d(&self) -> usize {
        unsafe { sys::faiss_Index_d(self.inner()) as usize }
    }

    fn train<T: AsRef<[f32]> + ?Sized>(&mut self, x: &T) -> Result<()> {
        let x = x.as_ref();
        let n = (x.len() / self.d()) as sys::idx_t;
        faiss_rc!({ sys::faiss_Index_train(self.inner(), n, x.as_ptr()) })?;
        Ok(())
    }

    fn remove_ids(&mut self, sel: IDSelector) -> Result<usize> {
        let mut n_removed = 0usize;
        faiss_rc!({
            sys::faiss_Index_remove_ids(self.inner(), sel.inner, addr_of_mut!(n_removed))
        })?;
        Ok(n_removed)
    }

    fn save<P: AsRef<str>>(&self, pth: P) -> Result<()> {
        let pth = pth.as_ref();
        let pth = CString::new(pth)?;
        faiss_rc!({ sys::faiss_write_index_fname(self.inner() as *const _, pth.as_ptr()) })?;
        todo!()
    }

    fn load<P: AsRef<str>>(pth: P) -> Result<CpuIndex> {
        let pth = pth.as_ref();
        let pth = CString::new(pth)?;
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_read_index_fname(pth.as_ptr(), 0, addr_of_mut!(inner)) })?;
        Ok(CpuIndex { inner })
    }
}

pub struct IDSelector {
    pub inner: *mut sys::FaissIDSelector,
}

impl IDSelector {
    pub fn batch(ids: &[i64]) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_IDSelectorBatch_new(addr_of_mut!(inner), ids.len(), ids.as_ptr())
        })?;
        Ok(Self {
            inner: inner as *mut _,
        })
    }
}

impl Drop for IDSelector {
    fn drop(&mut self) {
        unsafe { sys::faiss_IDSelector_free(self.inner) }
    }
}

#[derive(Debug)]
pub struct CpuIndex {
    pub inner: *mut sys::FaissIndex,
}

impl Drop for CpuIndex {
    fn drop(&mut self) {
        unsafe {
            sys::faiss_Index_free(self.inner);
        }
        trace!("drop: index={:?}", self.inner);
    }
}

impl Index for CpuIndex {
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }
}

impl CpuIndex {
    #[cfg(feature = "gpu")]
    pub fn to_multi_gpu(&self, devices: &[i32], split: bool) -> Result<gpu::GpuIndex> {
        let mut p_out = 0 as *mut _;
        let providers = (0..devices.len())
            .map(|_| -> Result<_> { gpu::GpuResourcesProvider::new() })
            .collect::<Result<Vec<_>>>()?;
        let mut options = 0 as *mut sys::FaissGpuClonerOptions;
        faiss_rc!({ sys::faiss_GpuClonerOptions_new(addr_of_mut!(options)) })?;
        if split {
            unsafe { sys::faiss_GpuMultipleClonerOptions_set_shard(options, 1) };
        }
        let providers_ = providers.iter().map(|p| p.inner).collect::<Vec<_>>();
        faiss_rc!({
            sys::faiss_index_cpu_to_gpu_multiple_with_options(
                providers_.as_ptr(),
                providers.len(),
                devices.as_ptr(),
                devices.len(),
                self.inner,
                options,
                addr_of_mut!(p_out),
            )
        })?;
        Ok(gpu::GpuIndex {
            splitted: split,
            devices: devices.to_vec(),
            inner: p_out,
            providers: providers,
        })
    }

    #[cfg(feature = "gpu")]
    pub fn into_multi_gpu(self, devices: &[i32], split: bool) -> Result<gpu::GpuIndex> {
        self.to_multi_gpu(devices, split)
    }

    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self, device: i32) -> Result<gpu::GpuIndex> {
        let mut p_out = 0 as *mut _;
        let provider = gpu::GpuResourcesProvider::new()?;
        trace!(?provider, "create gpu provider");
        faiss_rc!({
            sys::faiss_index_cpu_to_gpu(provider.inner, device, self.inner, addr_of_mut!(p_out))
        })?;
        trace!(
            "into_gpu: from {:?} to index={:?}, device={}",
            self.inner,
            p_out,
            device
        );
        Ok(gpu::GpuIndex {
            splitted: false,
            inner: p_out,
            devices: vec![device],
            providers: vec![provider],
        })
    }

    #[cfg(feature = "gpu")]
    pub fn into_gpu(self, device: i32) -> Result<gpu::GpuIndex> {
        self.to_gpu(device)
    }
}

impl Clone for CpuIndex {
    fn clone(&self) -> Self {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_clone_index(self.inner, addr_of_mut!(inner)) })
            .unwrap_or_else(|_| panic!("failed to clone index with inner={:?}", self.inner));
        Self { inner }
    }
}

pub fn index_factory(d: i32, description: &str, metric: FaissMetricType) -> Result<CpuIndex> {
    let mut p_index = null_mut();
    let description_ = CString::new(description)?;
    faiss_rc! {{sys::faiss_index_factory(addr_of_mut!(p_index), d, description_.as_ptr(), metric)}}?;
    trace!(
        "index_factory: create index={:?}, description={}, metric={:?}",
        p_index,
        description,
        metric
    );
    Ok(CpuIndex { inner: p_index })
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::{addr_of_mut, null_mut, sys, Error, Index};
    use tracing::trace;
    pub struct GpuIndex {
        pub splitted: bool,
        pub devices: Vec<i32>,
        pub providers: Vec<GpuResourcesProvider>,
        pub inner: *mut sys::FaissGpuIndex,
    }

    #[derive(Debug)]
    pub struct GpuResourcesProvider {
        pub inner: *mut sys::FaissGpuResourcesProvider,
    }

    impl GpuResourcesProvider {
        pub fn new() -> super::Result<Self> {
            let mut inner = null_mut();
            faiss_rc!({ sys::faiss_StandardGpuResources_new(addr_of_mut!(inner)) })?;
            trace!(?inner, "create gpu provider");
            Ok(Self { inner })
        }
    }

    impl Drop for GpuResourcesProvider {
        fn drop(&mut self) {
            unsafe { sys::faiss_GpuResourcesProvider_free(self.inner) }
            trace!("drop: gpu provider={:?}", self.inner);
        }
    }

    impl Index for GpuIndex {
        fn inner(&self) -> *mut sys::FaissIndex {
            self.inner as *mut _
        }
    }

    impl GpuIndex {
        pub fn to_cpu(&self) -> super::Result<super::CpuIndex> {
            let mut inner = null_mut();
            faiss_rc!({ sys::faiss_index_gpu_to_cpu(self.inner, addr_of_mut!(inner)) })?;
            Ok(super::CpuIndex { inner })
        }

        pub fn into_cpu(self) -> super::Result<super::CpuIndex> {
            self.to_cpu()
        }
    }

    impl Drop for GpuIndex {
        fn drop(&mut self) {
            unsafe {
                sys::faiss_Index_free(self.inner);
                trace!("drop: index={:?}", self.inner);
            }
        }
    }

    impl Clone for GpuIndex {
        fn clone(&self) -> Self {
            let cpu = self
                .to_cpu()
                .expect("failed to create cpu index from gpu index");
            let gpu = match self.splitted {
                true => cpu.into_gpu(self.devices[0]),
                false => cpu.into_multi_gpu(&self.devices, self.splitted),
            };
            gpu.expect("failed to create gpu index from cpu index")
        }
    }
}

#[cfg(test)]
#[test]
fn test_faiss_index_ok() -> Result<()> {
    use ndarray::{s, Array2};
    use ndarray_rand::*;
    use tracing::debug;
    std::env::set_var("RUST_LOG", "trace");
    let _ = tracing_subscriber::fmt::try_init();
    let mut index = index_factory(128, "Flat", FaissMetricType::METRIC_L2)?;
    let feats = Array2::random(
        (1024 * 1024, 128),
        rand::distributions::Uniform::new(0., 1.),
    );
    let query = feats.slice(s![42..43, ..]);
    index.add(feats.as_slice_memory_order().unwrap())?;
    let ret = index.search(query.as_slice_memory_order().unwrap(), 1)?;
    debug!(?ret);
    let index = index.clone();
    let ret = index.search(query.as_slice_memory_order().unwrap(), 1)?;
    debug!(?ret, "cloned index");
    #[cfg(feature = "gpu")]
    {
        let index = index.into_gpu(1)?;
        let ret = index.search(query.as_slice_memory_order().unwrap(), 1)?;
        debug!(?ret);
        let index = index.clone();
        let ret = index.search(query.as_slice_memory_order().unwrap(), 1)?;
        debug!(?ret, "cloned index");
    }
    Ok(())
}
