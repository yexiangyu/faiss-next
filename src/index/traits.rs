use crate::error::Result;
use crate::index::parameters::FaissSearchParameters;
use crate::macros::rc;
use faiss_next_sys as sys;

pub trait FaissIndexMutTrait: FaissIndexMutPtr + FaissIndexConstTrait {
    fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_Index_set_verbose(self.to_mut(), verbose as i32) }
    }

    fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        rc!({ sys::faiss_Index_train(self.to_mut(), n, x.as_ptr()) })?;
        Ok(())
    }

    fn add(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        rc!({ sys::faiss_Index_add(self.to_mut(), n, x.as_ptr()) })?;
        Ok(())
    }

    fn add_with_ids(&mut self, x: impl AsRef<[f32]>, ids: impl AsRef<[i64]>) {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        let ids = ids.as_ref();
        unsafe {
            sys::faiss_Index_add_with_ids(self.to_mut(), n, x.as_ptr(), ids.as_ptr());
        }
    }

    fn search(&self, x: impl AsRef<[f32]>, k: i64) -> Result<(Vec<i64>, Vec<f32>)> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d();
        let mut distances = vec![0.0; (n * k) as usize];
        let mut labels = vec![0; (n * k) as usize];
        rc!({
            sys::faiss_Index_search(
                self.to_ptr(),
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
        let mut distances = vec![0.0; (n * k) as usize];
        let mut labels = vec![0; (n * k) as usize];
        unsafe {
            sys::faiss_Index_search_with_params(
                self.to_ptr(),
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

pub trait FaissIndexConstTrait: FaissIndexBorrowedPtr {
    fn d(&self) -> i64 {
        unsafe { sys::faiss_Index_d(self.to_ptr()) as i64 }
    }

    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_Index_is_trained(self.to_ptr()) != 0 }
    }

    fn ntotal(&self) -> i64 {
        unsafe { sys::faiss_Index_ntotal(self.to_ptr()) }
    }

    fn metric_type(&self) -> sys::FaissMetricType {
        unsafe { sys::faiss_Index_metric_type(self.to_ptr()) }
    }

    fn verbose(&self) -> bool {
        unsafe { sys::faiss_Index_verbose(self.to_ptr()) != 0 }
    }
}

pub trait FaissIndexOwnedTrait:
    FaissIndexMutTrait + FaissIndexConstTrait + FaissIndexOwnedPtr
{
}

pub trait FaissIndexOwnedPtr {
    fn into_ptr(self) -> *mut sys::FaissIndex;
}

pub trait FaissIndexBorrowedPtr {
    fn to_ptr(&self) -> *const sys::FaissIndex;
}

pub trait FaissIndexMutPtr {
    fn to_mut(&mut self) -> *mut sys::FaissIndex;
}
