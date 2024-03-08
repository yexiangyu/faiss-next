use super::metric::FaissMetricType;
use super::parameters::FaissSearchParametersTrait;
use super::range::FaissRangeSearchResult;
use crate::error::Result;
use crate::rc;
use faiss_next_sys as sys;
use tracing::trace;

pub trait FaissIndexTrait {
    fn inner(&self) -> *mut sys::FaissIndex;

    fn into_inner(self) -> *mut sys::FaissIndex;

    fn d(&self) -> i32 {
        unsafe { sys::faiss_Index_d(self.inner()) }
    }

    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_Index_is_trained(self.inner()) != 0 }
    }

    fn ntotal(&self) -> i64 {
        unsafe { sys::faiss_Index_ntotal(self.inner()) }
    }

    fn metric_type(&self) -> FaissMetricType {
        unsafe { sys::faiss_Index_metric_type(self.inner()) }
    }

    fn verbose(&self) -> bool {
        unsafe { sys::faiss_Index_verbose(self.inner()) != 0 }
    }

    fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_Index_set_verbose(self.inner(), verbose as i32) }
    }

    fn train(&mut self, x: &[f32]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Index_train(self.inner(), n, x.as_ptr()) })?;
        trace!("train vector={},d={}", x.len(), self.d());
        Ok(())
    }

    fn add(&mut self, x: &[f32]) -> Result<()> {
        assert_eq!(x.len() % self.d() as usize, 0);
        let n = x.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Index_add(self.inner(), n, x.as_ptr()) })?;
        trace!(
            "add vector={},d={}, ntotal={}",
            x.len(),
            self.d(),
            self.ntotal()
        );
        Ok(())
    }

    fn add_with_ids(&mut self, x: &[f32], ids: &[i64]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Index_add_with_ids(self.inner(), n, x.as_ptr(), ids.as_ptr()) })?;
        Ok(())
    }

    fn search(&self, x: &[f32], k: i64) -> Result<(Vec<i64>, Vec<f32>)> {
        let n = x.len() as i64 / self.d() as i64;
        let mut distances = vec![0.0; n as usize * k as usize];
        let mut labels = vec![0; n as usize * k as usize];
        rc!({
            sys::faiss_Index_search(
                self.inner(),
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })?;
        Ok((labels, distances))
    }

    fn search_with_params(
        &self,
        x: &[f32],
        k: i64,
        params: &impl FaissSearchParametersTrait,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let n = x.len() as i64 / self.d() as i64;
        let mut distances = vec![0.0; n as usize * k as usize];
        let mut labels = vec![0; n as usize * k as usize];
        rc!({
            sys::faiss_Index_search_with_params(
                self.inner(),
                n,
                x.as_ptr(),
                k,
                params.inner(),
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })?;
        Ok((labels, distances))
    }

    fn range_search(&self, x: &[f32], radius: f32) -> Result<FaissRangeSearchResult> {
        let n = x.len() as i64 / self.d() as i64;
        assert!(n > 0);
        let result = FaissRangeSearchResult::new_with(n, true)?;
        rc!({ sys::faiss_Index_range_search(self.inner(), n, x.as_ptr(), radius, result.inner) })?;
        Ok(result)
    }
}

macro_rules! impl_index_drop {
    ($kls: ident, $free: ident) => {
        impl Drop for $kls {
            fn drop(&mut self) {
                tracing::trace!("drop faiss index inner={:?}", self.inner);
                if !self.inner.is_null() {
                    unsafe { sys::$free(self.inner) }
                }
            }
        }
    };
}

macro_rules! impl_index_trait {
    ($kls: ident) => {
        impl FaissIndexTrait for $kls {
            fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
                self.inner
            }

            fn into_inner(self) -> *mut faiss_next_sys::FaissIndex {
                let mut s = self;
                let inner = s.inner;
                s.inner = std::ptr::null_mut();
                inner
            }
        }
    };
}

pub(crate) use impl_index_drop;
pub(crate) use impl_index_trait;
