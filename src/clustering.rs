use std::ptr;

use faiss_next_sys::{self, FaissClustering, FaissClusteringParameters};

use crate::error::{check_return_code, Result};
use crate::index::Index;

pub struct Clustering {
    inner: *mut FaissClustering,
}

impl Clustering {
    pub fn new(d: u32, k: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_Clustering_new(
                &mut inner, d as i32, k as i32,
            ))?;
            Ok(Self { inner })
        }
    }

    pub fn new_with_params(d: u32, k: u32, params: &ClusteringParameters) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_Clustering_new_with_params(
                &mut inner,
                d as i32,
                k as i32,
                params.inner,
            ))?;
            Ok(Self { inner })
        }
    }

    pub fn train(&mut self, n: u64, x: &[f32], index: &mut impl Index) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_Clustering_train(
                self.inner,
                n as i64,
                x.as_ptr(),
                index.inner_ptr(),
            )
        })
    }

    pub fn niter(&self) -> i32 {
        unsafe { faiss_next_sys::faiss_Clustering_niter(self.inner) }
    }

    pub fn k(&self) -> usize {
        unsafe { faiss_next_sys::faiss_Clustering_k(self.inner) }
    }

    pub fn d(&self) -> usize {
        unsafe { faiss_next_sys::faiss_Clustering_d(self.inner) }
    }

    pub fn centroids(&self) -> Vec<f32> {
        unsafe {
            let mut ptr = ptr::null_mut();
            let mut size = 0usize;
            faiss_next_sys::faiss_Clustering_centroids(self.inner, &mut ptr, &mut size);
            if ptr.is_null() || size == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ptr, size).to_vec()
            }
        }
    }

    pub fn verbose(&self) -> bool {
        unsafe { faiss_next_sys::faiss_Clustering_verbose(self.inner) != 0 }
    }

    pub fn seed(&self) -> i32 {
        unsafe { faiss_next_sys::faiss_Clustering_seed(self.inner) }
    }
}

impl Drop for Clustering {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                faiss_next_sys::faiss_Clustering_free(self.inner);
            }
        }
    }
}

pub struct ClusteringParameters {
    inner: *mut FaissClusteringParameters,
}

impl ClusteringParameters {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut inner = Box::new(std::mem::zeroed::<FaissClusteringParameters>());
            faiss_next_sys::faiss_ClusteringParameters_init(inner.as_mut() as *mut _);
            Ok(Self {
                inner: Box::into_raw(inner),
            })
        }
    }

    pub fn niter(&mut self, niter: i32) -> &mut Self {
        unsafe { (*self.inner).niter = niter }
        self
    }

    pub fn verbose(&mut self, verbose: bool) -> &mut Self {
        unsafe { (*self.inner).verbose = verbose as i32 }
        self
    }

    pub fn spherical(&mut self, spherical: bool) -> &mut Self {
        unsafe { (*self.inner).spherical = spherical as i32 }
        self
    }

    pub fn min_points_per_centroid(&mut self, n: i32) -> &mut Self {
        unsafe { (*self.inner).min_points_per_centroid = n }
        self
    }

    pub fn max_points_per_centroid(&mut self, n: i32) -> &mut Self {
        unsafe { (*self.inner).max_points_per_centroid = n }
        self
    }

    pub fn seed(&mut self, seed: i32) -> &mut Self {
        unsafe { (*self.inner).seed = seed }
        self
    }

    pub fn nredo(&mut self, nredo: i32) -> &mut Self {
        unsafe { (*self.inner).nredo = nredo }
        self
    }
}

impl Default for ClusteringParameters {
    fn default() -> Self {
        Self::new().expect("failed to create ClusteringParameters")
    }
}

impl Drop for ClusteringParameters {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                let _ = Box::from_raw(self.inner);
            }
        }
    }
}
