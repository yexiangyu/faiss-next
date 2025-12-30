use crate::error::{Error, Result};
use faiss_next_sys as ffi;
use std::{
    ops::{Deref, DerefMut},
    ptr::null_mut,
    slice::from_raw_parts,
};

#[derive(Debug)]
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

impl Default for FaissClusteringParameters {
    fn default() -> Self {
        let mut inner = ffi::FaissClusteringParameters::default();
        unsafe { ffi::faiss_ClusteringParameters_init(&mut inner) };
        Self { inner }
    }
}

impl FaissClusteringParameters {
    /// Creates a new `FaissClusteringParameters` with default values.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct FaissClustering {
    pub inner: *mut ffi::FaissClustering,
}

// Getters for clustering parameters
ffi::impl_getter!(FaissClustering, niter, faiss_Clustering_niter, i32);
ffi::impl_getter!(FaissClustering, nredo, faiss_Clustering_nredo, i32);
ffi::impl_getter!(FaissClustering, verbose, faiss_Clustering_verbose, i32);
ffi::impl_getter!(FaissClustering, spherical, faiss_Clustering_spherical, i32);
ffi::impl_getter!(
    FaissClustering,
    int_centroids,
    faiss_Clustering_int_centroids,
    i32
);
ffi::impl_getter!(
    FaissClustering,
    update_index,
    faiss_Clustering_update_index,
    i32
);
ffi::impl_getter!(
    FaissClustering,
    frozen_centroids,
    faiss_Clustering_frozen_centroids,
    i32
);
ffi::impl_getter!(
    FaissClustering,
    min_points_per_centroid,
    faiss_Clustering_min_points_per_centroid,
    i32
);
ffi::impl_getter!(
    FaissClustering,
    max_points_per_centroid,
    faiss_Clustering_max_points_per_centroid,
    i32
);

ffi::impl_getter!(FaissClustering, seed, faiss_Clustering_seed, i32);
ffi::impl_getter!(
    FaissClustering,
    decode_block_size,
    faiss_Clustering_decode_block_size,
    usize
);
ffi::impl_getter!(FaissClustering, d, faiss_Clustering_d, usize);
ffi::impl_getter!(FaissClustering, k, faiss_Clustering_k, usize);

impl FaissClustering {
    /// Creates a new `FaissClustering` with the given dimension and number of centroids.
    ///
    /// # Arguments
    /// * `d` - Dimension of the data points
    /// * `k` - Number of centroids to create
    pub fn new(d: i32, k: i32) -> Result<Self> {
        let mut inner = null_mut();
        let code = unsafe { ffi::faiss_Clustering_new(&mut inner, d, k) };
        ffi::rc(code).map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Creates a new `FaissClustering` with custom parameters.
    ///
    /// # Arguments
    /// * `d` - Dimension of the data points
    /// * `k` - Number of centroids to create
    /// * `params` - Clustering parameters to use
    pub fn new_with_params(d: i32, k: i32, params: &FaissClusteringParameters) -> Result<Self> {
        let mut inner = null_mut();
        let code = unsafe { ffi::faiss_Clustering_new_with_params(&mut inner, d, k, &params.inner) };
        ffi::rc(code).map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Returns a slice of the centroids stored in this clustering object.
    /// Size is k * d floats.
    pub fn centroids(&self) -> &[f32] {
        let mut centroid = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_Clustering_centroids(self.inner, &mut centroid, &mut size) };
        unsafe { from_raw_parts(centroid, size) }
    }

    /// Returns the iteration statistics collected during clustering.
    pub fn iteration_stats(&self) -> Vec<FaissClusteringIterationStats> {
        let mut stats = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_Clustering_iteration_stats(self.inner, &mut stats, &mut size) };
        (0..size)
            .map(|i| FaissClusteringIterationStats {
                inner: unsafe { stats.add(i) },
            })
            .collect()
    }

    /// Performs clustering on the provided data points using the given index.
    ///
    /// # Arguments
    /// * `x` - Training vectors (size n * d)
    /// * `index` - Index to use for assigning points to centroids
    pub fn train(&mut self, x: impl AsRef<[f32]>, index: &mut impl crate::index::IndexTrait) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        let code = unsafe { 
            ffi::faiss_Clustering_train(self.inner, n, x.as_ref().as_ptr(), index.inner())
        };
        ffi::rc(code).map_err(Error::from)?;
        Ok(())
    }
}

ffi::impl_drop!(FaissClustering, faiss_Clustering_free);

/// Statistics collected during clustering iterations.
pub struct FaissClusteringIterationStats {
    pub inner: *mut ffi::FaissClusteringIterationStats,
}

ffi::impl_getter!(
    FaissClusteringIterationStats,
    obj,
    faiss_ClusteringIterationStats_obj,
    f32
);
ffi::impl_getter!(
    FaissClusteringIterationStats,
    time,
    faiss_ClusteringIterationStats_time,
    f64
);
ffi::impl_getter!(
    FaissClusteringIterationStats,
    time_search,
    faiss_ClusteringIterationStats_time_search,
    f64
);
ffi::impl_getter!(
    FaissClusteringIterationStats,
    imbalance_factor,
    faiss_ClusteringIterationStats_imbalance_factor,
    f64
);
ffi::impl_getter!(
    FaissClusteringIterationStats,
    nsplit,
    faiss_ClusteringIterationStats_nsplit,
    i32
);

/// Performs k-means clustering on the provided data points.
///
/// # Arguments
/// * `d` - Dimension of the data points
/// * `x` - Training vectors (size n * d)
/// * `centroids` - Output centroids (size k * d), will be overwritten
///
/// # Returns
/// The final quantization error
pub fn faiss_kmeans_clustering(
    d: usize,
    x: impl AsRef<[f32]>,
    mut centroids: impl AsMut<[f32]>,
) -> Result<f32> {
    assert_eq!(x.as_ref().len() % d, 0);
    let n = x.as_ref().len() / d;
    let k = centroids.as_mut().len() / d;
    let mut q_error = 0.0f32;
    let code = unsafe {
        ffi::faiss_kmeans_clustering(
            d,
            n,
            k,
            x.as_ref().as_ptr(),
            centroids.as_mut().as_mut_ptr(),
            &mut q_error,
        )
    };
    ffi::rc(code).map_err(Error::from)?;
    Ok(q_error)
}
