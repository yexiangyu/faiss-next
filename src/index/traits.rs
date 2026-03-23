use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexBinary, FaissRangeSearchResult};

use crate::error::{check_return_code, Error, Result};
use crate::idx::Idx;
use crate::metric::MetricType;
use crate::result::{BinarySearchResult, RangeSearchResult, SearchResult};

pub trait Index {
    fn inner_ptr(&self) -> *mut FaissIndex;

    fn is_trained(&self) -> bool {
        unsafe { faiss_next_sys::faiss_Index_is_trained(self.inner_ptr()) != 0 }
    }

    fn ntotal(&self) -> u64 {
        unsafe { faiss_next_sys::faiss_Index_ntotal(self.inner_ptr()) as u64 }
    }

    fn d(&self) -> u32 {
        unsafe { faiss_next_sys::faiss_Index_d(self.inner_ptr()) as u32 }
    }

    fn metric_type(&self) -> MetricType {
        let mt = unsafe { faiss_next_sys::faiss_Index_metric_type(self.inner_ptr()) };
        MetricType::from_native(mt)
    }

    fn train(&mut self, x: &[f32]) -> Result<()> {
        let n = x.len() / self.d() as usize;
        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_train(self.inner_ptr(), n as i64, x.as_ptr())
        })
    }

    fn add(&mut self, x: &[f32]) -> Result<()> {
        let n = x.len() / self.d() as usize;
        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_add(self.inner_ptr(), n as i64, x.as_ptr())
        })
    }

    fn add_with_ids(&mut self, x: &[f32], ids: &[Idx]) -> Result<()> {
        let n = x.len() / self.d() as usize;
        if ids.len() < n {
            return Err(Error::InvalidDimension {
                expected: n,
                actual: ids.len(),
            });
        }
        let ids_raw: Vec<i64> = ids.iter().map(|&id| id.as_repr()).collect();
        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_add_with_ids(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                ids_raw.as_ptr(),
            )
        })
    }

    fn search(&mut self, q: &[f32], k: usize) -> Result<SearchResult> {
        let d = self.d() as usize;
        let nq = q.len() / d;
        let mut distances = vec![0.0f32; nq * k];
        let mut labels = vec![Idx::NONE; nq * k];

        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_search(
                self.inner_ptr(),
                nq as i64,
                q.as_ptr(),
                k as i64,
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut i64,
            )
        })?;

        Ok(SearchResult::new(distances, labels))
    }

    fn search_with_params<P: crate::search_params::SearchParams>(
        &mut self,
        q: &[f32],
        k: usize,
        params: &P,
    ) -> Result<SearchResult> {
        let d = self.d() as usize;
        let nq = q.len() / d;
        let mut distances = vec![0.0f32; nq * k];
        let mut labels = vec![Idx::NONE; nq * k];

        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_search_with_params(
                self.inner_ptr(),
                nq as i64,
                q.as_ptr(),
                k as i64,
                params.as_ptr(),
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut i64,
            )
        })?;

        Ok(SearchResult::new(distances, labels))
    }

    fn range_search(&mut self, q: &[f32], radius: f32) -> Result<RangeSearchResult> {
        let d = self.d() as usize;
        let nq = q.len() / d;

        unsafe {
            let mut result: *mut FaissRangeSearchResult = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_RangeSearchResult_new(
                &mut result,
                nq as i64,
            ))?;

            check_return_code(faiss_next_sys::faiss_Index_range_search(
                self.inner_ptr(),
                nq as i64,
                q.as_ptr(),
                radius,
                result,
            ))?;

            let mut lims = ptr::null_mut();
            let mut distances = ptr::null_mut();
            let mut labels = ptr::null_mut();

            faiss_next_sys::faiss_RangeSearchResult_lims(result, &mut lims);
            faiss_next_sys::faiss_RangeSearchResult_labels(result, &mut labels, &mut distances);

            let nq_actual = nq + 1;
            let lims_slice = std::slice::from_raw_parts(lims, nq_actual).to_vec();
            let total = *lims_slice.last().unwrap_or(&0);

            let labels_slice = std::slice::from_raw_parts(labels as *const i64, total)
                .iter()
                .map(|&l| Idx(l))
                .collect();
            let distances_slice = std::slice::from_raw_parts(distances, total).to_vec();

            faiss_next_sys::faiss_RangeSearchResult_free(result);

            Ok(RangeSearchResult::new(
                labels_slice,
                distances_slice,
                lims_slice,
            ))
        }
    }

    fn reset(&mut self) -> Result<()> {
        check_return_code(unsafe { faiss_next_sys::faiss_Index_reset(self.inner_ptr()) })
    }

    fn reconstruct(&self, id: Idx) -> Result<Vec<f32>> {
        let d = self.d() as usize;
        let mut recons = vec![0.0f32; d];
        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_reconstruct(
                self.inner_ptr(),
                id.as_repr(),
                recons.as_mut_ptr(),
            )
        })?;
        Ok(recons)
    }

    fn verbose(&self) -> bool {
        unsafe { faiss_next_sys::faiss_Index_verbose(self.inner_ptr()) != 0 }
    }

    fn set_verbose(&mut self, verbose: bool) {
        unsafe { faiss_next_sys::faiss_Index_set_verbose(self.inner_ptr(), verbose as i32) }
    }

    fn compute_residual(&self, x: &[f32], key: Idx) -> Result<Vec<f32>> {
        let d = self.d() as usize;
        let mut residual = vec![0.0f32; d];
        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_compute_residual(
                self.inner_ptr(),
                x.as_ptr(),
                residual.as_mut_ptr(),
                key.as_repr(),
            )
        })?;
        Ok(residual)
    }

    fn compute_residual_n(&self, x: &[f32], keys: &[Idx]) -> Result<Vec<f32>> {
        let d = self.d() as usize;
        let n = x.len() / d;
        let mut residuals = vec![0.0f32; x.len()];
        let keys_raw: Vec<i64> = keys.iter().map(|&id| id.as_repr()).collect();
        check_return_code(unsafe {
            faiss_next_sys::faiss_Index_compute_residual_n(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                residuals.as_mut_ptr(),
                keys_raw.as_ptr(),
            )
        })?;
        Ok(residuals)
    }
}

pub trait IvfIndex: Index {
    fn nlist(&self) -> usize;
    fn nprobe(&self) -> usize;
    fn set_nprobe(&mut self, nprobe: usize);
}

pub trait BinaryIndex {
    fn inner_ptr(&self) -> *mut FaissIndexBinary;

    fn is_trained(&self) -> bool {
        unsafe { faiss_next_sys::faiss_IndexBinary_is_trained(self.inner_ptr()) != 0 }
    }

    fn ntotal(&self) -> u64 {
        unsafe { faiss_next_sys::faiss_IndexBinary_ntotal(self.inner_ptr()) as u64 }
    }

    fn d(&self) -> u32 {
        unsafe { faiss_next_sys::faiss_IndexBinary_d(self.inner_ptr()) as u32 }
    }

    fn metric_type(&self) -> MetricType {
        let mt = unsafe { faiss_next_sys::faiss_IndexBinary_metric_type(self.inner_ptr()) };
        MetricType::from_native(mt)
    }

    fn train(&mut self, x: &[u8]) -> Result<()> {
        let d_bytes = self.d() as usize / 8;
        let n = x.len() / d_bytes;
        check_return_code(unsafe {
            faiss_next_sys::faiss_IndexBinary_train(self.inner_ptr(), n as i64, x.as_ptr())
        })
    }

    fn add(&mut self, x: &[u8]) -> Result<()> {
        let d_bytes = self.d() as usize / 8;
        let n = x.len() / d_bytes;
        check_return_code(unsafe {
            faiss_next_sys::faiss_IndexBinary_add(self.inner_ptr(), n as i64, x.as_ptr())
        })
    }

    fn add_with_ids(&mut self, x: &[u8], ids: &[Idx]) -> Result<()> {
        let d_bytes = self.d() as usize / 8;
        let n = x.len() / d_bytes;
        if ids.len() < n {
            return Err(Error::InvalidDimension {
                expected: n,
                actual: ids.len(),
            });
        }
        let ids_raw: Vec<i64> = ids.iter().map(|&id| id.as_repr()).collect();
        check_return_code(unsafe {
            faiss_next_sys::faiss_IndexBinary_add_with_ids(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                ids_raw.as_ptr(),
            )
        })
    }

    fn search(&mut self, q: &[u8], k: usize) -> Result<BinarySearchResult> {
        let d_bytes = self.d() as usize / 8;
        let nq = q.len() / d_bytes;
        let mut distances = vec![0i32; nq * k];
        let mut labels = vec![Idx::NONE; nq * k];

        check_return_code(unsafe {
            faiss_next_sys::faiss_IndexBinary_search(
                self.inner_ptr(),
                nq as i64,
                q.as_ptr(),
                k as i64,
                distances.as_mut_ptr(),
                labels.as_mut_ptr() as *mut i64,
            )
        })?;

        Ok(BinarySearchResult::new(distances, labels))
    }

    fn reset(&mut self) -> Result<()> {
        check_return_code(unsafe { faiss_next_sys::faiss_IndexBinary_reset(self.inner_ptr()) })
    }
}
