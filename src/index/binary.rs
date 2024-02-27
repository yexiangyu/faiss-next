use super::FaissRangeSearchResult;
use crate::{error::Result, id_selector::IDSelector, macros::faiss_rc};
use faiss_next_sys as sys;
use sys::FaissMetricType;

pub struct FaissIndexBinary {
    inner: *mut sys::FaissIndexBinary,
}

impl Drop for FaissIndexBinary {
    fn drop(&mut self) {
        unsafe { sys::faiss_IndexBinary_free(self.inner) }
    }
}

impl FaissIndexBinary {
    pub fn d(&self) -> i32 {
        unsafe { sys::faiss_IndexBinary_d(self.inner) }
    }

    pub fn is_trained(&self) -> bool {
        unsafe { sys::faiss_IndexBinary_is_trained(self.inner) != 0 }
    }

    pub fn ntotal(&self) -> i64 {
        unsafe { sys::faiss_IndexBinary_ntotal(self.inner) }
    }

    pub fn metric_type(&self) -> FaissMetricType {
        unsafe { sys::faiss_IndexBinary_metric_type(self.inner) }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_IndexBinary_set_verbose(self.inner, verbose as i32) }
    }

    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_IndexBinary_verbose(self.inner) != 0 }
    }

    pub fn train(&mut self, x: &[u8]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        faiss_rc!(unsafe { sys::faiss_IndexBinary_train(self.inner, n, x.as_ptr()) })?;
        Ok(())
    }

    pub fn add(&mut self, x: &[u8]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        faiss_rc!(unsafe { sys::faiss_IndexBinary_add(self.inner, n, x.as_ptr()) })?;
        Ok(())
    }

    pub fn add_with_ids(&mut self, x: &[u8], xids: &[i64]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        faiss_rc!(unsafe {
            sys::faiss_IndexBinary_add_with_ids(self.inner, n, x.as_ptr(), xids.as_ptr())
        })?;
        Ok(())
    }
    pub fn search(&self, x: &[u8], k: i64) -> Result<(Vec<i64>, Vec<i32>)> {
        let n = x.len() as i64 / self.d() as i64;
        let mut distances = vec![0i32; n as usize * k as usize];
        let mut labels = vec![0i64; n as usize * k as usize];
        faiss_rc!(unsafe {
            sys::faiss_IndexBinary_search(
                self.inner,
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })?;
        Ok((labels, distances))
    }

    // TODO: more
    pub fn range_search(&mut self, x: &[u8], radius: i32) -> Result<FaissRangeSearchResult> {
        let n = x.len() as i64 / self.d() as i64;
        let result = FaissRangeSearchResult::new(n)?;
        faiss_rc!(unsafe {
            sys::faiss_IndexBinary_range_search(self.inner, n, x.as_ptr(), radius, result.inner)
        })?;
        Ok(result)
    }

    pub fn assign(&mut self, x: &[u8], labels: &mut [i64], k: i64) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        faiss_rc!(unsafe {
            sys::faiss_IndexBinary_assign(self.inner, n, x.as_ptr(), labels.as_mut_ptr(), k)
        })?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        faiss_rc!(unsafe { sys::faiss_IndexBinary_reset(self.inner) })?;
        Ok(())
    }

    pub fn remove_ids(&mut self, sel: impl IDSelector) -> Result<usize> {
        let mut n = 0usize;
        faiss_rc!({ sys::faiss_IndexBinary_remove_ids(self.inner, sel.ptr(), &mut n) })?;
        Ok(n)
    }

    pub fn reconstruct(&mut self, key: i64, recons: &mut [u8]) -> Result<()> {
        faiss_rc!(unsafe {
            sys::faiss_IndexBinary_reconstruct(self.inner, key, recons.as_mut_ptr())
        })?;
        Ok(())
    }

    pub fn reconstruct_n(&mut self, key: i64, recons: &mut [u8]) -> Result<()> {
        faiss_rc!(unsafe {
            sys::faiss_IndexBinary_reconstruct_n(
                self.inner,
                key,
                key + recons.len() as i64 / self.d() as i64,
                recons.as_mut_ptr(),
            )
        })?;
        Ok(())
    }
}
