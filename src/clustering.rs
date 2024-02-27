use std::{
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use crate::error::Result;
use crate::macros::faiss_rc;
use faiss_next_sys as sys;

pub struct FaissClusteringParameters {
    inner: sys::FaissClusteringParameters,
}

impl Default for FaissClusteringParameters {
    fn default() -> Self {
        let mut ret = Self {
            inner: Default::default(),
        };
        unsafe { sys::faiss_ClusteringParameters_init(&mut ret.inner) }
        ret
    }
}

impl Deref for FaissClusteringParameters {
    type Target = sys::FaissClusteringParameters;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for FaissClusteringParameters {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub struct FaissClustering {
    inner: *mut sys::FaissClustering,
}

impl Drop for FaissClustering {
    fn drop(&mut self) {
        unsafe { sys::faiss_Clustering_free(self.inner) }
    }
}

impl FaissClustering {
    pub fn new(d: usize, k: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_Clustering_new(&mut inner, d as i32, k as i32) })?;
        Ok(Self { inner })
    }

    pub fn new_with_params(d: usize, k: usize, params: FaissClusteringParameters) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_Clustering_new_with_params(&mut inner, d as i32, k as i32, &params.inner)
        })?;
        Ok(Self { inner })
    }

    pub fn niter(&self) -> i32 {
        unsafe { sys::faiss_Clustering_niter(self.inner) }
    }

    pub fn nredo(&self) -> i32 {
        unsafe { sys::faiss_Clustering_nredo(self.inner) }
    }

    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_Clustering_verbose(self.inner) != 0 }
    }

    pub fn spherical(&self) -> bool {
        unsafe { sys::faiss_Clustering_spherical(self.inner) != 0 }
    }

    pub fn int_centroids(&self) -> bool {
        unsafe { sys::faiss_Clustering_int_centroids(self.inner) != 0 }
    }

    pub fn update_index(&self) -> bool {
        unsafe { sys::faiss_Clustering_update_index(self.inner) != 0 }
    }

    pub fn frozen_centroids(&self) -> bool {
        unsafe { sys::faiss_Clustering_frozen_centroids(self.inner) != 0 }
    }
    pub fn min_points_per_centroid(&self) -> i32 {
        unsafe { sys::faiss_Clustering_min_points_per_centroid(self.inner) }
    }

    pub fn max_points_per_centroid(&self) -> i32 {
        unsafe { sys::faiss_Clustering_max_points_per_centroid(self.inner) }
    }

    pub fn seed(&self) -> i32 {
        unsafe { sys::faiss_Clustering_seed(self.inner) }
    }

    pub fn decode_block_size(&self) -> usize {
        unsafe { sys::faiss_Clustering_decode_block_size(self.inner) }
    }

    pub fn d(&self) -> usize {
        unsafe { sys::faiss_Clustering_d(self.inner) }
    }

    pub fn k(&self) -> usize {
        unsafe { sys::faiss_Clustering_k(self.inner) }
    }

    pub fn centroids(&self) -> Vec<&[f32]> {
        let mut ptr = null_mut();
        let mut len = 0usize;
        unsafe { sys::faiss_Clustering_centroids(self.inner, &mut ptr, &mut len) };
        // let n = len / self.d();
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };
        data.chunks(self.d()).collect()
    }
}

pub struct ClusteringIterationStats {
    inner: *mut sys::FaissClusteringIterationStats,
}

impl ClusteringIterationStats {
    pub fn obj(&self) -> f32 {
        unsafe { sys::faiss_ClusteringIterationStats_obj(self.inner) }
    }

    pub fn time(&self) -> f64 {
        unsafe { sys::faiss_ClusteringIterationStats_time(self.inner) }
    }

    pub fn time_search(&self) -> f64 {
        unsafe { sys::faiss_ClusteringIterationStats_time_search(self.inner) }
    }

    pub fn imbalance_factor(&self) -> f64 {
        unsafe { sys::faiss_ClusteringIterationStats_imbalance_factor(self.inner) }
    }

    pub fn nsplit(&self) -> i32 {
        unsafe { sys::faiss_ClusteringIterationStats_nsplit(self.inner) }
    }
}
