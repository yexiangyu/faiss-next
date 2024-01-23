//! # Faiss
//!
//! `faiss` is a light weight rust wrapper for [facebookresearch/faiss](https://github.com/facebookresearch/faiss) c api. Quick example:
//!
//! ```rust
//! use faiss_next::{index_factory, FaissMetricType, Index};
//! use ndarray::{s, Array2};
//! use ndarray_rand::*;
//!
//! //create index
//! let mut index = index_factory(128, "Flat", FaissMetricType::METRIC_L2).expect("failed to create cpu index");
//!
//! //create some random feature
//! let feats = Array2::random((1024, 128), rand::distributions::Uniform::new(0., 1.));
//!
//! //get query from position 42
//! let query = feats.slice(s![42..43, ..]);
//!
//! //add features in index
//! index.add(feats.as_slice_memory_order().unwrap()).expect("failed to add feature");
//!
//! //do the search
//! let ret = index.search(query.as_slice_memory_order().unwrap(), 1).expect("failed to search");
//! assert_eq!(ret.labels[0], 42i64);
//!
//! //move index from cpu to gpu, only available when gpu feature is enabled
//! #[cfg(feature = "gpu")]
//! {
//! let index = index.into_gpu(0).expect("failed to move index to gpu");
//! let ret = index.search(query.as_slice_memory_order().unwrap(), 1).expect("failed to search");
//! assert_eq!(ret.labels[0], 42i64);
//! }
//! ```
use faiss_next_sys as sys;
use std::ffi::CString;
use std::ptr::{addr_of_mut, null_mut};
use std::time::Instant;
use tracing::trace;

/// ## Metric Type
/// please refer to <https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances>
pub use sys::FaissMetricType;

/// Error
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

/// search result
#[derive(Debug)]
pub struct SearchResult {
    /// - labels of search result
    pub labels: Vec<i64>,
    /// - distances of search result
    pub distances: Vec<f32>,
}

/// Index trait, all index should implement this trait
pub trait Index {
    /// return inner pointer
    fn inner(&self) -> *mut sys::FaissIndex;

    /// add vectors to index, x.len() should be a multiple of d
    fn add<T: AsRef<[f32]> + ?Sized>(&mut self, x: &T) -> Result<()> {
        let x = x.as_ref();
        faiss_rc! {{sys::faiss_Index_add(self.inner(), (x.len() / self.d())as sys::idx_t, x.as_ptr())}}?;
        trace!("add: index={:?}, x.len={}", self.inner(), x.as_ref().len());
        Ok(())
    }

    /// search vector against index, `x.len()` should be a multiple of `d`, `k` means top k
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

    /// return dimension of index
    fn d(&self) -> usize {
        unsafe { sys::faiss_Index_d(self.inner()) as usize }
    }

    /// train index when some index impl is used, `is_trained` is todo
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
        todo!()
    }

    /// load index from disk
    fn load<P: AsRef<str>>(pth: P) -> Result<CpuIndex> {
        let pth = pth.as_ref();
        let pth = CString::new(pth)?;
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_read_index_fname(pth.as_ptr(), 0, addr_of_mut!(inner)) })?;
        Ok(CpuIndex { inner })
    }
}

/// Select ID to delete feature in index [IDSelector](https://faiss.ai/cpp_api/struct/structfaiss_1_1IDSelector.html)
pub struct IDSelector {
    pub inner: *mut sys::FaissIDSelector,
}

impl IDSelector {
    /// create a selector from batch ids
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

/// Index use cpu
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
    /// create multi gpu index, `devices` is a list of gpu device id, `split` means split index on multiple gpu or not
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

    /// create multi gpu index, `devices` is a list of gpu device id, `split` means split index on multiple gpu or not, cpu index will be dropped
    #[cfg(feature = "gpu")]
    pub fn into_multi_gpu(self, devices: &[i32], split: bool) -> Result<gpu::GpuIndex> {
        self.to_multi_gpu(devices, split)
    }

    /// create gpu index, `device` is gpu device id
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

    /// create gpu index, `device` is gpu device id, cpu index will be dropped
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

/// helper function to create cpu index, please refer to [doc](https://github.com/facebookresearch/faiss/wiki/The-index-factory) for details of `description` and `metric`
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

///  gpu related module
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::{addr_of_mut, null_mut, sys, Error, Index};
    use tracing::trace;

    /// gpu index
    pub struct GpuIndex {
        /// - index is splitted on multiple gpu or not
        pub splitted: bool,
        /// - gpu device used by index
        pub devices: Vec<i32>,
        /// - gpu resource provider of faiss
        pub providers: Vec<GpuResourcesProvider>,
        /// - raw pointer
        pub inner: *mut sys::FaissGpuIndex,
    }

    /// gpu resource provider
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
        /// to clone gpu index, we need to clone index to cpu index first, then move this gpu index to  gpu index, kinds of stupid.
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
    let feats = Array2::random((1024, 128), rand::distributions::Uniform::new(0., 1.));
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
