use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};
use crate::macros::*;
use crate::traits::FaissIndex;

pub struct IndexLSH {
    pub(crate) inner: *mut bindings::FaissIndex,
}

impl_faiss_drop!(IndexLSH, faiss_IndexLSH_free);
impl_index_common!(IndexLSH);

impl FaissIndex for IndexLSH {
    fn inner(&self) -> *mut bindings::FaissIndex {
        self.inner
    }

    fn train(&mut self, n: i64, x: &[f32]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_train(self.inner, n, x.as_ptr()) })
    }

    fn add(&mut self, n: i64, x: &[f32]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_add(self.inner, n, x.as_ptr()) })
    }

    fn add_with_ids(&mut self, n: i64, x: &[f32], ids: &[i64]) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_add_with_ids(self.inner, n, x.as_ptr(), ids.as_ptr())
        })
    }

    fn search(
        &self,
        n: i64,
        x: &[f32],
        k: i64,
        distances: &mut [f32],
        labels: &mut [i64],
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_search(
                self.inner,
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })
    }

    fn range_search(
        &self,
        n: i64,
        x: &[f32],
        radius: f32,
        result: *mut bindings::FaissRangeSearchResult,
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_range_search(self.inner, n, x.as_ptr(), radius, result)
        })
    }

    fn reset(&mut self) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_reset(self.inner) })
    }

    fn reconstruct(&self, key: i64, recons: &mut [f32]) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_reconstruct(self.inner, key, recons.as_mut_ptr())
        })
    }
}

impl IndexLSH {
    pub fn new(d: i32, nbits: i32) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_IndexLSH_new(&mut inner, d as i64, nbits as i32)
        })?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    #[test]
    fn test_index_lsh_basic() {
        let d = 16;
        let n = 100;
        let nbits = 32;
        let k = 5;

        let mut index = IndexLSH::new(d as i32, nbits).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);
        assert_eq!(index.d(), d as i32);

        let query = data.row(42).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert!(labels[0] >= 0);
    }

    #[test]
    #[ignore = "IndexLSH does not support custom IDs. Use IndexIDMap wrapper instead."]
    fn test_index_lsh_add_with_ids() {
        let d = 8;
        let n = 50;
        let nbits = 16;

        let mut index = IndexLSH::new(d as i32, nbits).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();
        let ids: Vec<i64> = (500..500 + n as i64).collect();

        index.add_with_ids(n as i64, &data_slice, &ids).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(25).to_owned();
        let mut distances = vec![0.0f32; 1];
        let mut labels = vec![-1i64; 1];

        index
            .search(1, query.as_slice().unwrap(), 1, &mut distances, &mut labels)
            .unwrap();

        assert!(labels[0] >= 500);
    }

    #[test]
    fn test_index_lsh_reset() {
        let d = 8;
        let n = 30;
        let nbits = 16;

        let mut index = IndexLSH::new(d as i32, nbits).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn test_index_lsh_multiple_queries() {
        let d = 16;
        let n = 200;
        let nbits = 64;
        let nq = 10;
        let k = 5;

        let mut index = IndexLSH::new(d as i32, nbits).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let queries = data.slice(ndarray::s![0..nq, ..]).to_owned();
        let queries_slice: Vec<f32> = queries.iter().copied().collect();

        let mut distances = vec![0.0f32; nq * k];
        let mut labels = vec![-1i64; nq * k];

        index
            .search(
                nq as i64,
                &queries_slice,
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        for i in 0..nq {
            assert!(labels[i * k] >= 0);
        }
    }
}
