#![doc = include_str!("../README.md")]

use faiss_next_sys as sys;
use std::ffi::CString;
use std::ptr::{addr_of_mut, null_mut};

/// ## Metric Type
/// please refer to <https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances>
pub use sys::FaissMetricType;

/// Error
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid index description {0}")]
    InvalidDescription(String),
    #[error("invalid index dimension")]
    InvalidDimension,
    #[error("gpu {0} not available")]
    GpuNotAvailable(i32),
    #[error("index already on gpu")]
    IndexOnGpu,
    #[error("index already on cpu")]
    IndexOnCpu,
    #[error("{0}")]
    NulErr(#[from] std::ffi::NulError),
    #[error("faiss error,code={},message={}", .code, .message)]
    Faiss { code: i32, message: String },
}

impl Error {
    pub fn from_rc(rc: i32) -> Self {
        match rc {
            0 => unimplemented!(),
            _ => {
                let message = unsafe { std::ffi::CStr::from_ptr(sys::faiss_get_last_error()) };
                let message = message
                    .to_str()
                    .unwrap_or("unknown error, failed to decode error message from bytes")
                    .to_string();
                Error::Faiss { code: rc, message }
            }
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! faiss_rc {
    ($blk: block) => {
        match unsafe { $blk } {
            0 => Ok(()),
            rc @ _ => Err($crate::Error::from_rc(rc)),
        }
    };
}

/// search result
#[derive(Debug)]
pub struct SearchResult {
    /// - labels of search result
    pub labels: Vec<i64>,
    /// - distances of search result
    pub distances: Vec<f32>,
}

/// Trait that can return inner pointer of index
pub trait IndexInner {
    /// return inner pointer
    fn inner(&self) -> *mut sys::FaissIndex;
}

/// Index trait, all index should implement this trait
pub trait Index: IndexInner {
    /// add vectors to index, x.len() should be a multiple of d
    fn add(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        faiss_rc!({
            sys::faiss_Index_add(self.inner(), (x.len() / self.d()) as sys::idx_t, x.as_ptr())
        })?;
        Ok(())
    }

    /// add vectors to index, x.len() should be a multiple of d, with ids
    fn add_with_ids(&mut self, x: impl AsRef<[f32]>, ids: impl AsRef<[i64]>) -> Result<()> {
        let x = x.as_ref();
        let ids = ids.as_ref();
        faiss_rc!({
            sys::faiss_Index_add_with_ids(
                self.inner(),
                (x.len() / self.d()) as sys::idx_t,
                x.as_ptr(),
                ids.as_ptr(),
            )
        })?;
        Ok(())
    }

    /// search vector against index, `x.len()` should be a multiple of `d`, `k` means top k
    fn search<T: AsRef<[f32]> + ?Sized>(&self, x: &T, k: usize) -> Result<SearchResult> {
        let x = x.as_ref();
        let n = (x.len() / self.d()) as sys::idx_t;
        let mut labels = vec![0 as sys::idx_t; n as usize * k];
        let mut distances = vec![0.0f32; n as usize * k];
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
        Ok(SearchResult { labels, distances })
    }

    /// return dimension of index
    fn d(&self) -> usize {
        unsafe { sys::faiss_Index_d(self.inner()) as usize }
    }

    // return whether index is trained
    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_Index_is_trained(self.inner()) != 0 }
    }

    /// train index when some index impl is used
    fn train<T: AsRef<[f32]> + ?Sized>(&mut self, x: &T) -> Result<()> {
        let x = x.as_ref();
        let n = (x.len() / self.d()) as sys::idx_t;
        faiss_rc!({ sys::faiss_Index_train(self.inner(), n, x.as_ptr()) })?;
        Ok(())
    }

    /// remove feature with [IDSelector](https://faiss.ai/cpp_api/struct/structfaiss_1_1IDSelector.html)
    fn remove_ids(&mut self, sel: IDSelector) -> Result<usize> {
        let mut n_removed = 0usize;
        faiss_rc!({
            sys::faiss_Index_remove_ids(self.inner(), sel.inner, addr_of_mut!(n_removed))
        })?;
        Ok(n_removed)
    }

    /// save index to disk
    fn save<P: AsRef<str>>(&self, pth: P) -> Result<()> {
        let pth = pth.as_ref();
        let pth = CString::new(pth)?;
        faiss_rc!({ sys::faiss_write_index_fname(self.inner() as *const _, pth.as_ptr()) })?;
        Ok(())
    }

    /// load index from disk
    fn load<P: AsRef<str>>(pth: P) -> Result<FaissIndex> {
        let pth = pth.as_ref();
        let pth = CString::new(pth)?;
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_read_index_fname(pth.as_ptr(), 0, addr_of_mut!(inner)) })?;
        Ok(FaissIndex {
            inner,
            #[cfg(feature = "gpu")]
            gpu_resources: None,
        })
    }
}

/// Select ID to delete feature in index [IDSelector](https://faiss.ai/cpp_api/struct/structfaiss_1_1IDSelector.html)
pub struct IDSelector {
    pub inner: *mut sys::FaissIDSelector,
}

impl IDSelector {
    /// create a selector from batch ids
    /// TODO: more id selector
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

pub struct FaissIndex {
    inner: *mut sys::FaissIndex,
    #[cfg(feature = "gpu")]
    #[allow(unused)]
    gpu_resources: Option<Vec<gpu::FaissGpuResourcesProvider>>,
}

impl IndexInner for FaissIndex {
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }
}

impl Index for FaissIndex {}

impl FaissIndex {
    pub fn builder() -> FaissIndexBuilder {
        Default::default()
    }
}

impl Drop for FaissIndex {
    fn drop(&mut self) {
        unsafe {
            sys::faiss_Index_free(self.inner);
        }
    }
}

#[derive(smart_default::SmartDefault)]
pub struct FaissIndexBuilder {
    description: String,
    dimension: u32,
    #[default(FaissMetricType::METRIC_L2)]
    metric: FaissMetricType,
    #[cfg(feature = "gpu")]
    devices: Option<Vec<i32>>,
    #[cfg(feature = "gpu")]
    sharding: bool,
}

impl FaissIndexBuilder {
    pub fn with_description(mut self, description: impl ToString) -> Self {
        self.description = description.to_string();
        self
    }

    pub fn with_dimension(mut self, dimension: u32) -> Self {
        self.dimension = dimension;
        self
    }
    pub fn with_metric(mut self, metric: FaissMetricType) -> Self {
        self.metric = metric;
        self
    }
    #[cfg(feature = "gpu")]
    pub fn with_gpus(mut self, devices: &[i32]) -> Self {
        self.devices = Some(devices.to_vec());
        self
    }

    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, device: i32) -> Self {
        self.devices = Some(vec![device]);
        self
    }

    #[cfg(feature = "gpu")]
    pub fn with_sharding(mut self, sharding: bool) -> Self {
        self.sharding = sharding;
        self
    }

    pub fn build(self) -> Result<FaissIndex> {
        let mut inner = null_mut();
        let description = CString::new(self.description)?;
        faiss_rc!({
            sys::faiss_index_factory(
                addr_of_mut!(inner),
                self.dimension as i32,
                description.as_ptr(),
                self.metric,
            )
        })?;

        #[allow(unused_mut)]
        let mut index = FaissIndex {
            inner,
            #[cfg(feature = "gpu")]
            gpu_resources: None,
        };

        #[cfg(feature = "gpu")]
        {
            use gpu::GpuIndex;

            if let Some(devices) = self.devices {
                if !devices.is_empty() {
                    if devices.len() == 1 {
                        index = index.into_gpu(devices[0])?;
                    } else {
                        index = index.into_gpus(&devices, self.sharding)?;
                    }
                }
            }
        }

        Ok(index)
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use std::ptr::{addr_of_mut, null_mut};

    use super::{sys, FaissIndex, Result};

    pub struct FaissGpuResourcesProvider {
        inner: *mut sys::FaissGpuResourcesProvider,
    }

    impl FaissGpuResourcesProvider {
        pub fn new() -> Result<Self> {
            let mut inner = std::ptr::null_mut();
            faiss_rc!({ sys::faiss_StandardGpuResources_new(addr_of_mut!(inner)) })?;
            Ok(Self { inner })
        }
    }

    impl Drop for FaissGpuResourcesProvider {
        fn drop(&mut self) {
            unsafe { sys::faiss_GpuResourcesProvider_free(self.inner) }
        }
    }

    fn gpu_num() -> Result<i32> {
        let mut r = 0;
        faiss_rc!({ sys::faiss_get_num_gpus(addr_of_mut!(r)) })?;
        Ok(r)
    }

    pub trait GpuIndex {
        fn to_gpu(&self, device: i32) -> Result<FaissIndex>;
        fn into_gpu(self, device: i32) -> Result<FaissIndex>;

        fn to_gpus(&self, devices: &[i32], sharding: bool) -> Result<FaissIndex>;
        fn into_gpus(self, devices: &[i32], sharding: bool) -> Result<FaissIndex>;

        fn to_cpu(&self) -> Result<FaissIndex>;
        fn into_cpu(self) -> Result<FaissIndex>;
    }

    impl GpuIndex for super::FaissIndex {
        fn to_gpu(&self, device: i32) -> Result<FaissIndex> {
            let gpu_num = gpu_num()?;

            if device >= gpu_num || device < 0 {
                return Err(super::Error::GpuNotAvailable(device));
            }

            let Self {
                inner: inner_rhs,
                gpu_resources,
            } = self;

            if gpu_resources.is_some() {
                return Err(super::Error::IndexOnGpu);
            }

            let gpu_resources = FaissGpuResourcesProvider::new()?;

            let mut inner = null_mut();

            faiss_rc!({
                sys::faiss_index_cpu_to_gpu(
                    gpu_resources.inner,
                    device,
                    *inner_rhs,
                    addr_of_mut!(inner),
                )
            })?;

            Ok(Self {
                inner,
                gpu_resources: Some(vec![gpu_resources]),
            })
        }

        fn into_gpu(self, device: i32) -> Result<FaissIndex> {
            self.to_gpu(device)
        }

        fn to_gpus(&self, devices: &[i32], sharding: bool) -> Result<FaissIndex> {
            let Self {
                inner: inner_rhs,
                gpu_resources,
            } = self;

            if gpu_resources.is_some() {
                return Err(super::Error::IndexOnGpu);
            }

            let gpu_num = gpu_num()?;

            devices
                .iter()
                .map(|device| match *device >= 0 && *device < gpu_num {
                    true => Ok(()),
                    false => Err(super::Error::GpuNotAvailable(*device)),
                })
                .collect::<Result<Vec<_>>>()?;

            let providers = (0..devices.len())
                .map(|_| -> Result<_> { FaissGpuResourcesProvider::new() })
                .collect::<Result<Vec<_>>>()?;

            let mut options = 0 as *mut sys::FaissGpuClonerOptions;

            faiss_rc!({ sys::faiss_GpuClonerOptions_new(addr_of_mut!(options)) })?;

            if sharding {
                unsafe { sys::faiss_GpuMultipleClonerOptions_set_shard(options, 1) };
            }

            let providers_ = providers.iter().map(|p| p.inner).collect::<Vec<_>>();

            let mut inner = null_mut();

            faiss_rc!({
                sys::faiss_index_cpu_to_gpu_multiple_with_options(
                    providers_.as_ptr(),
                    providers.len(),
                    devices.as_ptr(),
                    devices.len(),
                    *inner_rhs,
                    options,
                    addr_of_mut!(inner),
                )
            })?;

            unsafe { sys::faiss_GpuClonerOptions_free(options) }

            Ok(Self {
                inner,
                gpu_resources: Some(providers),
            })
        }

        fn into_gpus(self, devices: &[i32], sharding: bool) -> Result<FaissIndex> {
            self.to_gpus(devices, sharding)
        }

        fn to_cpu(&self) -> Result<FaissIndex> {
            let Self {
                inner: inner_rhs,
                gpu_resources,
            } = self;

            if gpu_resources.is_none() {
                return Err(super::Error::IndexOnCpu);
            }

            let mut inner = null_mut();

            faiss_rc!({ sys::faiss_index_gpu_to_cpu(*inner_rhs, addr_of_mut!(inner)) })?;

            Ok(Self {
                inner,
                gpu_resources: None,
            })
        }

        fn into_cpu(self) -> Result<FaissIndex> {
            self.to_cpu()
        }
    }
}

pub mod prelude {
    #[cfg(feature = "gpu")]
    pub use super::gpu::GpuIndex;
    pub use super::{FaissIndex, FaissMetricType, Index, IndexInner};
}
