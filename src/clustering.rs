use std::{
    ops::{Deref, DerefMut},
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use faiss_next_sys as sys;
use tracing::trace;

use crate::macros::rc;
use crate::{error::Result, index::IndexTrait};

/// Search parameter for search
pub struct ClusteringParameters {
    inner: sys::FaissClusteringParameters,
}

impl std::fmt::Debug for ClusteringParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusteringParameters")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Default for ClusteringParameters {
    fn default() -> Self {
        let mut inner = sys::FaissClusteringParameters::default();
        unsafe { sys::faiss_ClusteringParameters_init(&mut inner) };
        let r = Self { inner };
        trace!(?r, "default");
        r
    }
}

impl Deref for ClusteringParameters {
    type Target = sys::FaissClusteringParameters;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for ClusteringParameters {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub struct Clustering {
    inner: *mut sys::FaissClustering,
}

impl Drop for Clustering {
    fn drop(&mut self) {
        trace!(?self, "drop");
        unsafe { sys::faiss_Clustering_free(self.inner) }
    }
}

impl std::fmt::Debug for Clustering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Clustering")
            .field("inner", &self.inner)
            .field("niter", &self.niter())
            .field("verbose", &self.verbose())
            .field("spherical", &self.spherical())
            .field("int_centroid", &self.int_centroid())
            .field("update_index", &self.update_index())
            .field("min_points_per_centroid", &self.min_points_per_centroid())
            .field("max_points_per_centroid", &self.max_points_per_centroid())
            .field("seed", &self.seed())
            .field("decode_block_size", &self.decode_block_size())
            .field("d", &self.d())
            .field("k", &self.k())
            .finish()
    }
}

impl Clustering {
    pub fn niter(&self) -> i32 {
        unsafe { sys::faiss_Clustering_niter(self.inner) }
    }

    pub fn nredo(&self) -> i32 {
        unsafe { sys::faiss_Clustering_nredo(self.inner) }
    }

    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_Clustering_spherical(self.inner) != 0 }
    }

    pub fn spherical(&self) -> bool {
        unsafe { sys::faiss_Clustering_spherical(self.inner) != 0 }
    }

    pub fn int_centroid(&self) -> bool {
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
        let mut size = 0usize;
        let mut data = null_mut();
        unsafe {
            sys::faiss_Clustering_centroids(self.inner, &mut data, &mut size);
        }
        let data = unsafe { from_raw_parts(data, size) };
        (0..self.k())
            .map(|i| &data[i * self.d()..(i + 1) * self.d()])
            .collect()
    }

    pub fn iteration_stats(&self) -> Vec<ClusteringIterationStats> {
        let mut size = 0usize;
        let mut data = null_mut();
        unsafe {
            sys::faiss_Clustering_iteration_stats(self.inner, &mut data, &mut size);
        }
        let data = unsafe { from_raw_parts_mut(data, size) };
        (0..self.niter())
            .map(|i| ClusteringIterationStats {
                inner: &mut data[i as usize],
            })
            .collect()
    }

    pub fn new(d: i32, k: i32, params: Option<&ClusteringParameters>) -> Result<Self> {
        let mut inner = null_mut();
        match params {
            Some(params) => {
                rc!({ sys::faiss_Clustering_new_with_params(&mut inner, d, k, &params.inner) })?;
            }
            None => {
                rc!({ sys::faiss_Clustering_new(&mut inner, d, k) })?;
            }
        }
        trace!(%d, %k, ?params, ?inner, "create Clustering");
        Ok(Self { inner })
    }

    pub fn train(&mut self, x: impl AsRef<[f32]>, index: &mut impl IndexTrait) -> Result<()> {
        let x = x.as_ref();
        let n = x.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Clustering_train(self.inner, n, x.as_ptr(), index.ptr()) })
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

impl std::fmt::Debug for ClusteringIterationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusteringIterationStats")
            .field("inner", &self.inner)
            .field("obj", &self.obj())
            .field("time", &self.time())
            .field("time_search", &self.time_search())
            .field("imbalance_factory", &self.imbalance_factor())
            .field("nsplit", &self.nsplit())
            .finish()
    }
}

pub fn kmeans_clustering(
    d: usize,
    k: usize,
    x: impl AsRef<[f32]>,
    mut centroids: impl AsMut<[f32]>,
) -> Result<f32> {
    let n = x.as_ref().len() / d;
    let mut q_error = 0f32;
    rc!({
        sys::faiss_kmeans_clustering(
            d,
            n,
            k,
            x.as_ref().as_ptr(),
            centroids.as_mut().as_mut_ptr(),
            &mut q_error,
        )
    })?;
    Ok(q_error)
}
