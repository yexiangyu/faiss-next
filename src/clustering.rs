use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};

pub struct Clustering {
    pub(crate) inner: *mut bindings::FaissClustering,
}

impl Drop for Clustering {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_Clustering_free(self.inner) }
        }
    }
}

impl Clustering {
    pub fn new(d: i32, k: i32) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe { bindings::faiss_Clustering_new(&mut inner, d, k) })?;
        Ok(Self { inner })
    }

    pub fn new_with_params(
        d: i32,
        k: i32,
        params: &bindings::FaissClusteringParameters,
    ) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_Clustering_new_with_params(&mut inner, d, k, params)
        })?;
        Ok(Self { inner })
    }

    pub fn train(&mut self, n: i64, x: &[f32], index: &mut crate::index::Index) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Clustering_train(self.inner, n, x.as_ptr(), index.inner)
        })
    }

    pub fn niter(&self) -> i32 {
        unsafe { bindings::faiss_Clustering_niter(self.inner) }
    }

    pub fn k(&self) -> usize {
        unsafe { bindings::faiss_Clustering_k(self.inner) }
    }

    pub fn d(&self) -> usize {
        unsafe { bindings::faiss_Clustering_d(self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::FaissMetricType;
    use crate::index_factory::index_factory;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    #[test]
    fn test_clustering_basic() {
        let d = 16;
        let n = 1000;
        let k = 10;

        let clustering = Clustering::new(d as i32, k as i32).unwrap();
        assert_eq!(clustering.k(), k);
        assert_eq!(clustering.d(), d);
    }

    #[test]
    fn test_clustering_niter() {
        let d = 8;
        let k = 5;

        let clustering = Clustering::new(d as i32, k as i32).unwrap();
        let niter = clustering.niter();
        assert!(niter > 0);
    }

    #[test]
    fn test_clustering_train() {
        let d = 16;
        let n = 500;
        let k = 5;

        let mut clustering = Clustering::new(d as i32, k as i32).unwrap();
        let mut index = index_factory(d as i32, "Flat", FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        clustering.train(n as i64, &data_slice, &mut index).unwrap();
    }

    #[test]
    fn test_clustering_with_params() {
        let d = 8;
        let k = 3;

        let mut params = bindings::FaissClusteringParameters::default();
        unsafe { bindings::faiss_ClusteringParameters_init(&mut params) };
        params.niter = 5;
        params.verbose = 0;

        let clustering = Clustering::new_with_params(d as i32, k as i32, &params).unwrap();
        assert_eq!(clustering.d(), d);
    }
}
