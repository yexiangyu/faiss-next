pub mod clone;
pub mod factory;
pub mod flat;
pub mod id_map;
pub mod io;
pub mod ivf;
pub mod ivf_flat;
pub mod lsh;
pub mod pre_transform;
pub mod replicas;
pub mod scalar_quantizer;
pub mod shards;

use faiss_next_sys as sys;
use tracing::*;

use crate::{
    error::Result,
    implement::{id_selector::IDSelectorTrait, range_search_result::RangeSearchResult},
    macros::rc,
    metric::MetricType,
};
use std::ptr::null_mut;

pub trait SearchParametersTrait {
    fn ptr(&self) -> *mut sys::FaissSearchParameters;
}

/// SearchParameters for `IndexTrait::search`
#[derive(Debug)]
pub struct SearchParameters {
    inner: *mut sys::FaissSearchParameters,
}

impl Drop for SearchParameters {
    fn drop(&mut self) {
        trace!(?self, "drop");
        unsafe { sys::faiss_SearchParameters_free(self.inner) }
    }
}

impl SearchParametersTrait for SearchParameters {
    fn ptr(&self) -> *mut sys::FaissSearchParameters {
        self.inner
    }
}

impl SearchParameters {
    /// Create a new `SearchParameters`
    pub fn new(sel: &impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_SearchParameters_new(&mut inner, sel.ptr()) })?;
        let r = Self { inner };
        trace!(?r, "new");
        Ok(r)
    }
}

unsafe impl Send for SearchParameters {}
unsafe impl Sync for SearchParameters {}

pub trait IndexTrait: std::fmt::Debug + Send + Sync {
    fn ptr(&self) -> *mut sys::FaissIndex;

    fn dry_run(&self) -> Result<()> {
        let data = vec![0.0f32; self.d() as usize];
        let mut d = vec![0.0f32];
        let mut l = vec![0i64];
        self.search(&data, 1, &mut d, &mut l, Option::<SearchParameters>::None)?;
        Ok(())
    }

    fn d(&self) -> i32 {
        unsafe { sys::faiss_Index_d(self.ptr()) }
    }

    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_Index_is_trained(self.ptr()) != 0 }
    }

    fn ntotal(&self) -> i64 {
        unsafe { sys::faiss_Index_ntotal(self.ptr()) }
    }

    fn metric_type(&self) -> MetricType {
        unsafe { sys::faiss_Index_metric_type(self.ptr()).into() }
    }
    fn verbose(&self) -> bool {
        unsafe { sys::faiss_Index_verbose(self.ptr()) != 0 }
    }

    fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_Index_set_verbose(self.ptr(), verbose as i32) }
    }

    fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        let n = x.as_ref().len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Index_train(self.ptr(), n, x.as_ptr()) })
    }

    fn add(&mut self, x: impl AsRef<[f32]>, ids: Option<impl AsRef<[i64]>>) -> Result<()> {
        let x = x.as_ref();
        let n = x.as_ref().len() as i64 / self.d() as i64;
        match ids {
            Some(ids) => {
                let ids = ids.as_ref();
                assert_eq!(n, ids.as_ref().len() as i64);
                rc!({ sys::faiss_Index_add_with_ids(self.ptr(), n, x.as_ptr(), ids.as_ptr()) })
            }
            None => rc!({ sys::faiss_Index_add(self.ptr(), n, x.as_ptr()) }),
        }
    }

    fn search(
        &self,
        x: impl AsRef<[f32]>,
        k: usize,
        mut distances: impl AsMut<[f32]>,
        mut lablels: impl AsMut<[i64]>,
        params: Option<impl SearchParametersTrait>,
    ) -> Result<()> {
        let x = x.as_ref();
        let n = x.as_ref().len() / self.d() as usize;
        let distances = distances.as_mut();
        let lablels = lablels.as_mut();
        assert_eq!(n, lablels.as_ref().len());
        assert_eq!(n * k, distances.as_ref().len());
        match params {
            Some(params) => rc!({
                sys::faiss_Index_search_with_params(
                    self.ptr(),
                    n as _,
                    x.as_ptr(),
                    k as _,
                    params.ptr(),
                    distances.as_mut_ptr(),
                    lablels.as_mut_ptr(),
                )
            }),
            None => rc!({
                sys::faiss_Index_search(
                    self.ptr(),
                    n as _,
                    x.as_ptr(),
                    k as _,
                    distances.as_mut_ptr(),
                    lablels.as_mut_ptr(),
                )
            }),
        }
    }

    fn range_search(
        &self,
        x: impl AsRef<[f32]>,
        radius: f32,
        result: &mut RangeSearchResult,
    ) -> Result<()> {
        let x = x.as_ref();
        let n = x.as_ref().len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Index_range_search(self.ptr(), n, x.as_ptr(), radius, result.ptr()) })
    }

    fn assign(&self, x: impl AsRef<[f32]>, mut labels: impl AsMut<[i64]>, k: i64) -> Result<()> {
        let x = x.as_ref();
        let n = x.as_ref().len() as i64 / self.d() as i64;
        let labels = labels.as_mut();
        assert_eq!(n, labels.as_ref().len() as i64);
        rc!({ sys::faiss_Index_assign(self.ptr(), n, x.as_ptr(), labels.as_mut_ptr(), k) })
    }

    fn reset(&mut self) -> Result<()> {
        rc!({ sys::faiss_Index_reset(self.ptr()) })
    }

    fn remove_ids(&mut self, sel: &impl IDSelectorTrait) -> Result<usize> {
        let sel = Box::from(sel);
        let mut removed = 0usize;
        rc!({ sys::faiss_Index_remove_ids(self.ptr(), sel.ptr(), &mut removed) })?;
        Ok(removed)
    }

    fn reconstruct(&self, key: i64, mut recons: impl AsMut<[f32]>) -> Result<()> {
        let recons = recons.as_mut();
        rc!({ sys::faiss_Index_reconstruct(self.ptr(), key, recons.as_mut_ptr()) })
    }

    fn reconstruct_n(&self, i0: i64, ni: i64, mut recons: impl AsMut<[f32]>) -> Result<()> {
        let recons = recons.as_mut();
        rc!({ sys::faiss_Index_reconstruct_n(self.ptr(), i0, ni, recons.as_mut_ptr()) })?;
        Ok(())
    }

    fn compute_residual(
        &self,
        x: impl AsRef<[f32]>,
        mut residual: impl AsMut<[f32]>,
        key: i64,
    ) -> Result<()> {
        let x = x.as_ref();
        let residual = residual.as_mut();
        rc!({
            sys::faiss_Index_compute_residual(self.ptr(), x.as_ptr(), residual.as_mut_ptr(), key)
        })
    }

    fn compute_residual_n(
        &self,
        x: impl AsRef<[f32]>,
        mut residual: impl AsMut<[f32]>,
        keys: impl AsRef<[i64]>,
    ) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d() as i64;
        let keys = keys.as_ref();
        let residual = residual.as_mut();
        rc!({
            sys::faiss_Index_compute_residual_n(
                self.ptr(),
                n,
                x.as_ptr(),
                residual.as_mut_ptr(),
                keys.as_ptr(),
            )
        })
    }

    fn sa_code_size(&self) -> Result<usize> {
        let mut size = 0usize;
        rc!({ sys::faiss_Index_sa_code_size(self.ptr(), &mut size) })?;
        Ok(size)
    }

    fn sa_encode(&self, x: impl AsRef<[f32]>, mut codes: impl AsMut<[u8]>) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d() as i64;
        let codes = codes.as_mut();
        rc!({ sys::faiss_Index_sa_encode(self.ptr(), n, x.as_ptr(), codes.as_mut_ptr()) })
    }

    fn sa_decode(&self, codes: impl AsRef<[u8]>, mut x: impl AsMut<[f32]>) -> Result<()> {
        let codes = codes.as_ref();
        let n = codes.len() as i64 / self.sa_code_size()? as i64;
        let x = x.as_mut();
        rc!({ sys::faiss_Index_sa_decode(self.ptr(), n, codes.as_ptr(), x.as_mut_ptr()) })
    }
}

macro_rules! impl_index {
    ($cls: ty) => {
        unsafe impl Send for $cls {}
        unsafe impl Sync for $cls {}

        impl crate::index::IndexTrait for $cls {
            fn ptr(&self) -> *mut faiss_next_sys::FaissIndex {
                self.inner
            }
        }

        impl Drop for $cls {
            fn drop(&mut self) {
                tracing::trace!(?self, "drop");
                if !self.inner.is_null() {
                    unsafe { faiss_next_sys::faiss_Index_free(self.inner as *mut _) };
                }
            }
        }

        impl std::fmt::Debug for $cls {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($cls))
                    .field("inner", &self.inner)
                    .field("d", &self.d())
                    .field("is_trained", &self.is_trained())
                    .field("ntotal", &self.ntotal())
                    .field("metric_type", &self.metric_type())
                    .field("verbose", &self.verbose())
                    .finish()
            }
        }
    };
}

pub(crate) use impl_index;
