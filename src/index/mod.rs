use faiss_next_sys as sys;
pub mod factory;
pub mod flat;

use crate::error::Result;
use crate::macros::faiss_rc;
pub use sys::FaissMetricType;

pub trait IndexPtr {
    fn ptr(&self) -> *const sys::FaissIndex;

    fn mut_ptr(&mut self) -> *mut sys::FaissIndex;

    fn into_ptr(self) -> *mut sys::FaissIndex;
}

pub trait Index: IndexPtr {
    fn d(&self) -> i64 {
        unsafe { sys::faiss_Index_d(self.ptr()) as i64 }
    }

    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_Index_is_trained(self.ptr()) != 0 }
    }

    fn ntotal(&self) -> i64 {
        unsafe { sys::faiss_Index_ntotal(self.ptr()) }
    }

    fn metric_type(&self) -> FaissMetricType {
        unsafe { sys::faiss_Index_metric_type(self.ptr()) }
    }

    fn verbose(&self) -> bool {
        unsafe { sys::faiss_Index_verbose(self.ptr()) != 0 }
    }

    fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_Index_set_verbose(self.mut_ptr(), verbose as i32) }
    }

    fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        faiss_rc!({ sys::faiss_Index_train(self.mut_ptr(), n, x.as_ptr()) })?;
        Ok(())
    }

    fn add(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        faiss_rc!({ sys::faiss_Index_add(self.mut_ptr(), n, x.as_ptr()) })?;
        Ok(())
    }

    fn add_with_ids(&mut self, x: impl AsRef<[f32]>, ids: impl AsRef<[i64]>) {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        let ids = ids.as_ref();
        unsafe {
            sys::faiss_Index_add_with_ids(self.mut_ptr(), n, x.as_ptr(), ids.as_ptr());
        }
    }

    fn search(&self, x: impl AsRef<[f32]>, k: i64) -> Result<(Vec<i64>, Vec<f32>)> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        let mut distances = vec![0.0; n as usize * k as usize];
        let mut labels = vec![0; n as usize * k as usize];
        faiss_rc!({
            sys::faiss_Index_search(
                self.ptr(),
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
        x: impl AsRef<[f32]>,
        k: i64,
        params: FaissSearchParameters,
    ) -> (Vec<i64>, Vec<f32>) {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        let mut distances = vec![0.0; n as usize * k as usize];
        let mut labels = vec![0; n as usize * k as usize];
        unsafe {
            sys::faiss_Index_search_with_params(
                self.ptr(),
                n,
                x.as_ptr(),
                k,
                params.inner,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
        }
        (labels, distances)
    }
}

pub struct FaissSearchParameters {
    pub inner: *mut sys::FaissSearchParameters,
}

impl Drop for FaissSearchParameters {
    fn drop(&mut self) {
        unsafe { sys::faiss_SearchParameters_free(self.inner) }
    }
}
