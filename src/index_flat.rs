use std::ptr;

use crate::bindings::{self, FaissMetricType};
use crate::error::{check_return_code, Result};
use crate::macros::*;
use crate::traits::FaissIndex;

pub struct IndexFlat {
    pub(crate) inner: *mut bindings::FaissIndex,
}

impl_faiss_drop!(IndexFlat, faiss_IndexFlat_free);
impl_index_common!(IndexFlat);

impl FaissIndex for IndexFlat {
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

impl IndexFlat {
    pub fn new(d: i32, metric: FaissMetricType) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_IndexFlat_new_with(&mut inner, d as i64, metric)
        })?;
        Ok(Self { inner })
    }

    pub fn xb(&self) -> &[f32] {
        let mut ptr = ptr::null_mut();
        let mut size = 0usize;
        unsafe { bindings::faiss_IndexFlat_xb(self.inner, &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
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
    fn test_index_flat_l2() {
        let d = 64;
        let n = 1000;
        let k = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);
        assert_eq!(index.d(), d as i32);

        let query = data.row(42).to_owned();
        let mut distances = vec![0.0f32; k as usize];
        let mut labels = vec![-1i64; k as usize];

        index
            .search(1, query.as_slice().unwrap(), k, &mut distances, &mut labels)
            .unwrap();

        assert_eq!(labels[0], 42);
        assert!(distances[0] < 1e-5);
    }

    #[test]
    fn test_index_flat_inner_product() {
        let d = 32;
        let n = 500;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_INNER_PRODUCT).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(100).to_owned();
        let mut distances = vec![0.0f32; 5];
        let mut labels = vec![-1i64; 5];

        index
            .search(1, query.as_slice().unwrap(), 5, &mut distances, &mut labels)
            .unwrap();

        assert_eq!(labels[0], 100);
    }

    #[test]
    #[ignore = "IndexFlat does not support custom IDs. Use IndexIDMap wrapper instead."]
    fn test_index_flat_add_with_ids() {
        let d = 16;
        let n = 100;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        index.add_with_ids(n as i64, &data_slice, &ids).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(50).to_owned();
        let mut distances = vec![0.0f32; 1];
        let mut labels = vec![-1i64; 1];

        index
            .search(1, query.as_slice().unwrap(), 1, &mut distances, &mut labels)
            .unwrap();

        assert_eq!(labels[0], 1050);
    }

    #[test]
    fn test_index_flat_reset() {
        let d = 8;
        let n = 50;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn test_index_flat_reconstruct() {
        let d = 16;
        let n = 20;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let mut reconstructed = vec![0.0f32; d];
        index.reconstruct(5, &mut reconstructed).unwrap();

        let original = data.row(5);
        for i in 0..d {
            assert!((reconstructed[i] - original[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_index_flat_xb() {
        let d = 8;
        let n = 10;

        let mut index = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let xb = index.xb();
        assert_eq!(xb.len(), n * d);
    }
}
