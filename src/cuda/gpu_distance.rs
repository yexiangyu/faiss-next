use std::{ffi::c_void, ptr::null_mut, slice::from_raw_parts};

use faiss_next_sys::{extension::*, FaissMetricType};
use tracing::*;

use crate::error::{faiss_rc, Result};

use super::prelude::FaissGpuResourcesProviderTrait;

#[derive(Debug)]
pub struct GpuDistanceParams {
    inner: *mut c_void,
}

impl GpuDistanceParams {
    pub fn new(metric: FaissMetricType, dims: usize, device: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(cuda::gpu_distance_params_new(
            &mut inner,
            metric as i32,
            dims as i32,
            device as i32,
        ))?;
        let r = Ok(Self { inner });
        trace!(?r, "created");
        r
    }

    pub fn dims(&self) -> usize {
        let mut dims = 0;
        cuda::gpu_distance_params_get_dims(self.inner, &mut dims);
        dims as usize
    }

    pub fn k(&self) -> i32 {
        let mut k = 0;
        cuda::gpu_distance_params_get_k(self.inner, &mut k);
        k
    }

    pub fn set_k(&mut self, k: i32) {
        cuda::gpu_distance_params_set_k(self.inner, k);
    }

    pub fn num_vectors(&self) -> usize {
        let mut num_vectors = 0;
        cuda::gpu_distance_params_get_num_vectors(self.inner, &mut num_vectors);
        num_vectors as usize
    }

    pub fn num_queries(&self) -> usize {
        let mut num_queries = 0;
        cuda::gpu_distance_params_get_num_queries(self.inner, &mut num_queries);
        num_queries as usize
    }

    pub fn set_vectors(&mut self, vectors: &[f32]) {
        cuda::gpu_distance_params_set_vectors(self.inner, vectors.as_ptr());
        cuda::gpu_distance_params_set_num_vectors(
            self.inner,
            vectors.len() as i32 / self.dims() as i32,
        );
    }

    pub fn set_queries(&mut self, queries: &[f32]) {
        cuda::gpu_distance_params_set_queries(self.inner, queries.as_ptr());
        cuda::gpu_distance_params_set_num_queries(
            self.inner,
            queries.len() as i32 / self.dims() as i32,
        );
    }

    pub fn get_results(&self) -> &[f32] {
        let mut results = null_mut();
        cuda::gpu_distance_params_get_results(self.inner, &mut results);
        let num_results = match self.k() {
            -1 => self.num_vectors() * self.num_queries(),
            0 => panic!("k cannot be 0"),
            _ => self.k() as usize * self.num_queries(),
        };
        unsafe { from_raw_parts(results, num_results) }
    }

    pub fn get_indices(&self) -> &[i64] {
        let mut indices = null_mut();
        cuda::gpu_distance_params_get_indices(self.inner, &mut indices);
        let num_results = match self.k() {
            -1 => self.num_vectors() * self.num_queries(),
            0 => panic!("k cannot be 0"),
            _ => self.num_queries() * self.k() as usize,
        };
        unsafe { from_raw_parts(indices, num_results) }
    }

    pub fn compute(&mut self, resources: &impl FaissGpuResourcesProviderTrait) -> Result<()> {
        cuda::gpu_distance_params_compute(self.inner, resources.inner() as *mut _);
        Ok(())
    }
}

impl Drop for GpuDistanceParams {
    fn drop(&mut self) {
        trace!(?self, "dropping");
        cuda::gpu_distance_params_free(self.inner);
    }
}

#[cfg(test)]
#[test]
fn test_gpu_distance_params_ok() -> Result<()> {
    use super::prelude::FaissStandardGpuResources;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    let vectors = Array2::random((1024 * 1024 * 10, 128), Uniform::new(-1.0, 1.0));
    let queries = vectors.slice(ndarray::s![1024, ..]).to_owned();

    let _ = dotenv::dotenv();
    let _ = tracing_subscriber::fmt::try_init();

    let mut params = GpuDistanceParams::new(FaissMetricType::METRIC_L2, 128, 0)?;

    let resources = FaissStandardGpuResources::new()?;

    params.set_vectors(&vectors.as_slice_memory_order().unwrap());
    params.set_queries(&queries.as_slice_memory_order().unwrap());
    params.set_k(1);
    params.compute(&resources)?;

    let results = params.get_results();
    let indices = params.get_indices();

    info!(?results, ?indices);

    Ok(())
}
