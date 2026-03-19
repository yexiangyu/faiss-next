use std::ffi::c_void;
use std::ptr;

use crate::bindings::FaissMetricType;
use crate::cuda::gpu_resources::GpuResourcesProvider;
use crate::error::{check_return_code, Result};

pub struct GpuDistanceParams {
    inner: *mut c_void,
}

impl Drop for GpuDistanceParams {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { crate::extension::cuda::gpu_distance_params_free(self.inner) }
        }
    }
}

impl GpuDistanceParams {
    pub fn new(metric: FaissMetricType, dims: i32, device: i32) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            crate::extension::cuda::gpu_distance_params_new(&mut inner, metric as i32, dims, device)
        })?;
        Ok(Self { inner })
    }

    pub fn set_vectors(&mut self, vectors: &[f32]) {
        unsafe {
            crate::extension::cuda::gpu_distance_params_set_vectors(self.inner, vectors.as_ptr())
        }
    }

    pub fn set_queries(&mut self, queries: &[f32]) {
        unsafe {
            crate::extension::cuda::gpu_distance_params_set_queries(self.inner, queries.as_ptr())
        }
    }

    pub fn set_k(&mut self, k: i32) {
        unsafe { crate::extension::cuda::gpu_distance_params_set_k(self.inner, k) }
    }

    pub fn set_num_vectors(&mut self, num_vectors: i32) {
        unsafe {
            crate::extension::cuda::gpu_distance_params_set_num_vectors(self.inner, num_vectors)
        }
    }

    pub fn set_num_queries(&mut self, num_queries: i32) {
        unsafe {
            crate::extension::cuda::gpu_distance_params_set_num_queries(self.inner, num_queries)
        }
    }

    pub fn set_results(&mut self, results: &mut [f32]) {
        unsafe {
            crate::extension::cuda::gpu_distance_params_set_results(
                self.inner,
                results.as_mut_ptr(),
            )
        }
    }

    pub fn set_indices(&mut self, indices: &mut [i64]) {
        unsafe {
            crate::extension::cuda::gpu_distance_params_set_indices(
                self.inner,
                indices.as_mut_ptr(),
            )
        }
    }

    pub fn compute(&self, resources: &impl GpuResourcesProvider) -> Result<()> {
        check_return_code(unsafe {
            crate::extension::cuda::gpu_distance_params_compute(
                self.inner,
                resources.inner() as *mut _,
            )
        })
    }

    pub fn get_dims(&self) -> i32 {
        let mut dims = 0;
        unsafe { crate::extension::cuda::gpu_distance_params_get_dims(self.inner, &mut dims) }
        dims
    }

    pub fn get_k(&self) -> i32 {
        let mut k = 0;
        unsafe { crate::extension::cuda::gpu_distance_params_get_k(self.inner, &mut k) }
        k
    }

    pub fn get_num_vectors(&self) -> i32 {
        let mut num = 0;
        unsafe { crate::extension::cuda::gpu_distance_params_get_num_vectors(self.inner, &mut num) }
        num
    }

    pub fn get_num_queries(&self) -> i32 {
        let mut num = 0;
        unsafe { crate::extension::cuda::gpu_distance_params_get_num_queries(self.inner, &mut num) }
        num
    }
}
