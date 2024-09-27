use std::os::raw::c_char;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("faiss-next-sys/src/extension/index_binary_factory.hpp");
        #[cfg(feature = "cuda")]
        include!("faiss-next-sys/src/extension/gpu_distance.hpp");

        type FaissIndexBinary;
        type Char;
        type GpuDistanceParams;
        type GpuResources;

        unsafe fn faiss_index_binary_factory(
            d: i32,
            description: *const Char,
            index: *mut *mut FaissIndexBinary,
        ) -> i32;

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_new(
            params: *mut *mut GpuDistanceParams,
            metric: i32,
            dims: i32,
            device: i32,
        ) -> i32;

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_get_dims(params: *mut GpuDistanceParams, dims: *mut i32);

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_free(params: *mut GpuDistanceParams);

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_vectors(
            params: *mut GpuDistanceParams,
            vectors: *const f32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_get_k(params: *mut GpuDistanceParams, k: *mut i32);

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_k(params: *mut GpuDistanceParams, k: i32);

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_get_num_vectors(
            params: *mut GpuDistanceParams,
            num_vectors: *mut i32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_get_num_queries(
            params: *mut GpuDistanceParams,
            num_queries: *mut i32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_num_vectors(
            params: *mut GpuDistanceParams,
            num_vectors: i32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_queries(
            params: *mut GpuDistanceParams,
            queries: *const f32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_num_queries(
            params: *mut GpuDistanceParams,
            num_queries: i32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_get_results(
            params: *mut GpuDistanceParams,
            results: *mut *mut f32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_results(
            params: *mut GpuDistanceParams,
            results: *mut f32,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_get_indices(
            params: *mut GpuDistanceParams,
            indices: *mut *mut i64,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_set_indices(
            params: *mut GpuDistanceParams,
            indices: *mut i64,
        );

        #[cfg(feature = "cuda")]
        unsafe fn gpu_distance_params_compute(
            params: *mut GpuDistanceParams,
            resources: *mut GpuResources,
        ) -> i32;
    }
}

pub fn faiss_index_binary_factory(
    d: i32,
    description: *const c_char,
    index: *mut *mut super::FaissIndexBinary,
) -> i32 {
    unsafe { ffi::faiss_index_binary_factory(d, description as *const _, index as *mut *mut _) }
}

#[cfg(feature = "cuda")]
pub mod cuda {

    use super::ffi;
    use std::ffi::c_void;

    pub fn gpu_distance_params_new(
        params: *mut *mut c_void,
        metric: i32,
        dims: i32,
        device: i32,
    ) -> i32 {
        unsafe { ffi::gpu_distance_params_new(params as *mut *mut _, metric, dims, device) }
    }

    pub fn gpu_distance_params_get_k(params: *mut c_void, k: *mut i32) {
        unsafe { ffi::gpu_distance_params_get_k(params as *mut _, k) }
    }

    pub fn gpu_distance_params_set_k(params: *mut c_void, k: i32) {
        unsafe { ffi::gpu_distance_params_set_k(params as *mut _, k) }
    }

    pub fn gpu_distance_params_get_num_vectors(params: *mut c_void, num_vectors: *mut i32) {
        unsafe { ffi::gpu_distance_params_get_num_vectors(params as *mut _, num_vectors) }
    }

    pub fn gpu_distance_params_get_num_queries(params: *mut c_void, num_queries: *mut i32) {
        unsafe { ffi::gpu_distance_params_get_num_queries(params as *mut _, num_queries) }
    }

    pub fn gpu_distance_params_free(params: *mut c_void) {
        unsafe { ffi::gpu_distance_params_free(params as *mut _) }
    }

    pub fn gpu_distance_params_get_dims(params: *mut c_void, dims: *mut i32) {
        unsafe { ffi::gpu_distance_params_get_dims(params as *mut _, dims) }
    }

    pub fn gpu_distance_params_set_vectors(params: *mut c_void, vectors: *const f32) {
        unsafe { ffi::gpu_distance_params_set_vectors(params as *mut _, vectors) }
    }

    pub fn gpu_distance_params_set_num_vectors(params: *mut c_void, num_vectors: i32) {
        unsafe { ffi::gpu_distance_params_set_num_vectors(params as *mut _, num_vectors) }
    }

    pub fn gpu_distance_params_set_queries(params: *mut c_void, queries: *const f32) {
        unsafe { ffi::gpu_distance_params_set_queries(params as *mut _, queries) }
    }

    pub fn gpu_distance_params_set_num_queries(params: *mut c_void, num_queries: i32) {
        unsafe { ffi::gpu_distance_params_set_num_queries(params as *mut _, num_queries) }
    }

    pub fn gpu_distance_params_get_results(params: *mut c_void, results: *mut *mut f32) {
        unsafe { ffi::gpu_distance_params_get_results(params as *mut _, results) }
    }

    pub fn gpu_distance_params_get_indices(params: *mut c_void, indices: *mut *mut i64) {
        unsafe { ffi::gpu_distance_params_get_indices(params as *mut _, indices) }
    }

    pub fn gpu_distance_params_compute(params: *mut c_void, resources: *mut c_void) -> i32 {
        unsafe { ffi::gpu_distance_params_compute(params as *mut _, resources as *mut _) }
    }

    pub fn gpu_distance_params_set_results(params: *mut c_void, results: *mut f32) {
        unsafe { ffi::gpu_distance_params_set_results(params as *mut _, results) }
    }

    pub fn gpu_distance_params_set_indices(params: *mut c_void, indices: *mut i64) {
        unsafe { ffi::gpu_distance_params_set_indices(params as *mut _, indices) }
    }
}

// #[cfg(feature = "cuda")]
// pub mod cuda {
//     use super::ffi;

//     pub struct GpuDistanceParams {
//         inner: *mut ffi::GpuDistanceParams,
//     }

//     impl GpuDistanceParams {
//         pub fn new(metric: i32, dims: i32) -> Self {
//             let mut params = std::ptr::null_mut();
//             unsafe {
//                 ffi::gpu_distance_params_new(&mut params, metric, dims);
//             }
//             Self { inner: params }
//         }
//     }

//     impl Drop for GpuDistanceParams {
//         fn drop(&mut self) {
//             unsafe { ffi::gpu_distance_params_free(self.inner) }
//         }
//     }

//     #[cfg(test)]
//     #[test]
//     fn test_gpu_distance_params_ok() {
//         use crate::FaissMetricType;

//         let params = GpuDistanceParams::new(FaissMetricType::METRIC_L2 as i32, 128)?;
//     }
// }
