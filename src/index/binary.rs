use super::id_selector::FaissIDSelectorTrait;
use super::metric::FaissMetricType;
use super::range::FaissRangeSearchResult;
use crate::error::Result;
use crate::rc;
use faiss_next_sys as sys;

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
        unsafe { sys::faiss_IndexBinary_metric_type(self.inner).into() }
    }

    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_IndexBinary_verbose(self.inner) != 0 }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_IndexBinary_set_verbose(self.inner, verbose as i32) }
    }

    pub fn train(&mut self, data: &[u8]) -> Result<()> {
        let n = data.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_IndexBinary_train(self.inner, n, data.as_ptr()) })?;
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<()> {
        let n = data.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_IndexBinary_add(self.inner, n, data.as_ptr()) })?;
        Ok(())
    }

    pub fn add_with_ids(&mut self, data: &[u8], ids: &[i64]) -> Result<()> {
        let n = data.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_IndexBinary_add_with_ids(self.inner, n, data.as_ptr(), ids.as_ptr()) })?;
        Ok(())
    }

    pub fn search(&self, data: &[u8], k: i64) -> Result<(Vec<i64>, Vec<i32>)> {
        let n = data.len() as i64 / self.d() as i64;
        let mut distances = vec![0; n as usize * k as usize];
        let mut labels = vec![0; n as usize * k as usize];
        rc!({
            sys::faiss_IndexBinary_search(
                self.inner,
                n,
                data.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })?;
        Ok((labels, distances))
    }

    pub fn range_search(&self, data: &[u8], radius: i32) -> Result<FaissRangeSearchResult> {
        let n = data.len() as i64 / self.d() as i64;
        let result = FaissRangeSearchResult::new(n)?;
        rc!({
            sys::faiss_IndexBinary_range_search(self.inner, n, data.as_ptr(), radius, result.inner)
        })?;
        Ok(result)
    }

    pub fn assign(&self, data: &[u8], k: i64) -> Result<Vec<i64>> {
        let n = data.len() as i64 / self.d() as i64;
        let mut labels = vec![0; n as usize * k as usize];
        rc!({
            sys::faiss_IndexBinary_assign(self.inner, n, data.as_ptr(), labels.as_mut_ptr(), k)
        })?;
        Ok(labels)
    }

    pub fn reset(&mut self) -> Result<()> {
        rc!({ sys::faiss_IndexBinary_reset(self.inner) })?;
        Ok(())
    }

    pub fn remove_ids(&mut self, selector: impl FaissIDSelectorTrait) -> Result<usize> {
        let mut removed = 0usize;
        rc!({ sys::faiss_IndexBinary_remove_ids(self.inner, selector.inner(), &mut removed) })?;
        Ok(removed)
    }

    pub fn reconstruct(&self, key: i64) -> Result<Vec<u8>> {
        let mut data = vec![0; self.d() as usize];
        rc!({ sys::faiss_IndexBinary_reconstruct(self.inner, key, data.as_mut_ptr()) })?;
        Ok(data)
    }

    pub fn reconstruct_n(&self, i0: i64, ni: i64) -> Result<Vec<Vec<u8>>> {
        let mut data = vec![0; self.d() as usize * ni as usize];
        rc!({ sys::faiss_IndexBinary_reconstruct_n(self.inner, i0, ni, data.as_mut_ptr(),) })?;
        Ok(data.chunks(self.d() as usize).map(|i| i.to_vec()).collect())
    }
}
