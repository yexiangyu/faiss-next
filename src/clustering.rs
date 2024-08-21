use crate::{error::*, index::FaissIndexTrait, macros::*};
use faiss_next_sys as ffi;
use itertools::Itertools;
use std::{
    ops::{Deref, DerefMut},
    ptr::null_mut,
    slice::from_raw_parts,
};

// pub use ffi::FaissClusteringParameters;

pub struct FaissClusteringParameters {
    pub inner: ffi::FaissClusteringParameters,
}

impl Deref for FaissClusteringParameters {
    type Target = ffi::FaissClusteringParameters;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for FaissClusteringParameters {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl FaissClusteringParameters {
    pub fn new() -> Self {
        let mut inner = ffi::FaissClusteringParameters::default();
        unsafe { ffi::faiss_ClusteringParameters_init(&mut inner) };
        Self { inner }
    }
}

pub struct FaissClustering {
    inner: *mut ffi::FaissClustering,
}
impl_faiss_getter!(FaissClustering, niter, faiss_Clustering_niter, i32);
impl_faiss_getter!(FaissClustering, nredo, faiss_Clustering_nredo, i32);
impl_faiss_getter!(FaissClustering, verbose, faiss_Clustering_verbose, i32);
impl_faiss_getter!(FaissClustering, spherical, faiss_Clustering_spherical, i32);
impl_faiss_getter!(
    FaissClustering,
    int_centroids,
    faiss_Clustering_int_centroids,
    i32
);
impl_faiss_getter!(
    FaissClustering,
    update_index,
    faiss_Clustering_update_index,
    i32
);
impl_faiss_getter!(
    FaissClustering,
    frozen_centroids,
    faiss_Clustering_frozen_centroids,
    i32
);
impl_faiss_getter!(
    FaissClustering,
    min_points_per_centroid,
    faiss_Clustering_min_points_per_centroid,
    i32
);
impl_faiss_getter!(
    FaissClustering,
    max_points_per_centroid,
    faiss_Clustering_max_points_per_centroid,
    i32
);

impl_faiss_getter!(FaissClustering, seed, faiss_Clustering_seed, i32);
impl_faiss_getter!(
    FaissClustering,
    decode_block_size,
    faiss_Clustering_decode_block_size,
    usize
);
impl_faiss_getter!(FaissClustering, d, faiss_Clustering_d, usize);
impl_faiss_getter!(FaissClustering, k, faiss_Clustering_k, usize);
impl FaissClustering {
    pub fn centroids(&self) -> &[f32] {
        let mut centroid = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_Clustering_centroids(self.inner, &mut centroid, &mut size) };
        unsafe { from_raw_parts(centroid, size) }
    }
    pub fn iteration_stats(&self) -> Vec<FaissClusteringIterationStats> {
        let mut stats = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_Clustering_iteration_stats(self.inner, &mut stats, &mut size) };
        (0..size)
            .map(|i| FaissClusteringIterationStats {
                inner: unsafe { stats.add(i) },
            })
            .collect_vec()
    }
    pub fn new_with_params(d: i32, k: i32, params: &FaissClusteringParameters) -> Result<Self> {
        let mut inner = null_mut();
        crate::error::faiss_rc(unsafe {
            ffi::faiss_Clustering_new_with_params(&mut inner, d, k, &params.inner)
        })?;
        Ok(Self { inner })
    }
    pub fn train(&mut self, x: impl AsRef<[f32]>, index: &mut impl FaissIndexTrait) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe {
            ffi::faiss_Clustering_train(self.inner, n, x.as_ref().as_ptr(), index.inner())
        })
    }
}
impl_faiss_new!(
    FaissClustering,
    new,
    FaissClustering,
    faiss_Clustering_new,
    d,
    i32,
    k,
    i32
);
impl_faiss_new!(
    FaissClustering,
    raw_new_with_params,
    FaissClustering,
    faiss_Clustering_new_with_params,
    d,
    i32,
    k,
    i32,
    params,
    *const ffi::FaissClusteringParameters
);
impl_faiss_drop!(FaissClustering, faiss_Clustering_free);

pub struct FaissClusteringIterationStats {
    pub inner: *mut ffi::FaissClusteringIterationStats,
}
impl_faiss_getter!(
    FaissClusteringIterationStats,
    obj,
    faiss_ClusteringIterationStats_obj,
    f32
);
impl_faiss_getter!(
    FaissClusteringIterationStats,
    time,
    faiss_ClusteringIterationStats_time,
    f64
);
impl_faiss_getter!(
    FaissClusteringIterationStats,
    time_search,
    faiss_ClusteringIterationStats_time_search,
    f64
);
impl_faiss_getter!(
    FaissClusteringIterationStats,
    imbalance_factor,
    faiss_ClusteringIterationStats_imbalance_factor,
    f64
);
impl_faiss_getter!(
    FaissClusteringIterationStats,
    nsplit,
    faiss_ClusteringIterationStats_nsplit,
    i32
);

pub fn faiss_kmeans_clustering(
    d: usize,
    x: impl AsRef<[f32]>,
    mut centroids: impl AsMut<[f32]>,
) -> Result<f32> {
    assert_eq!(x.as_ref().len() % d, 0);
    let n = x.as_ref().len() / d;
    let k = centroids.as_mut().len() / d;
    let mut q_error = 0.0f32;
    faiss_rc(unsafe {
        ffi::faiss_kmeans_clustering(
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
