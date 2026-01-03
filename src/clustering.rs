use crate::error::FaissError;
use crate::ffi;
use crate::index::IndexTrait;
use anyhow::Result;

/// Statistics for a single clustering iteration
#[derive(Debug, Clone, Copy)]
pub struct ClusteringIterationStats {
    /// Objective function value
    pub obj: f32,
    /// Total time for the iteration
    pub time: f64,
    /// Time spent on search
    pub time_search: f64,
    /// Imbalance factor
    pub imbalance_factor: f64,
    /// Number of splits
    pub nsplit: i32,
}

impl ClusteringIterationStats {
    unsafe fn from_ptr(ptr: *const ffi::FaissClusteringIterationStats) -> Self {
        unsafe {
            Self {
                obj: ffi::faiss_ClusteringIterationStats_obj(ptr),
                time: ffi::faiss_ClusteringIterationStats_time(ptr),
                time_search: ffi::faiss_ClusteringIterationStats_time_search(ptr),
                imbalance_factor: ffi::faiss_ClusteringIterationStats_imbalance_factor(ptr),
                nsplit: ffi::faiss_ClusteringIterationStats_nsplit(ptr),
            }
        }
    }
}

/// Parameters for k-means clustering
#[derive(Debug, Clone)]
pub struct ClusteringParameters {
    /// Number of clustering iterations
    pub niter: i32,
    /// Redo clustering this many times and keep best
    pub nredo: i32,
    /// Verbose output
    pub verbose: bool,
    /// Do we want normalized centroids?
    pub spherical: bool,
    /// Round centroids coordinates to integer
    pub int_centroids: bool,
    /// Update index after each iteration?
    pub update_index: bool,
    /// Use the centroids provided as input and do not change them
    pub frozen_centroids: bool,
    /// Minimum points per centroid
    pub min_points_per_centroid: i32,
    /// Maximum points per centroid
    pub max_points_per_centroid: i32,
    /// Random seed
    pub seed: i32,
    /// Decode block size
    pub decode_block_size: usize,
}

impl Default for ClusteringParameters {
    fn default() -> Self {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::FaissClusteringParameters>::uninit();
            ffi::faiss_ClusteringParameters_init(params.as_mut_ptr());
            let params = params.assume_init();

            Self {
                niter: params.niter,
                nredo: params.nredo,
                verbose: params.verbose != 0,
                spherical: params.spherical != 0,
                int_centroids: params.int_centroids != 0,
                update_index: params.update_index != 0,
                frozen_centroids: params.frozen_centroids != 0,
                min_points_per_centroid: params.min_points_per_centroid,
                max_points_per_centroid: params.max_points_per_centroid,
                seed: params.seed,
                decode_block_size: params.decode_block_size,
            }
        }
    }
}

impl ClusteringParameters {
    fn to_ffi(&self) -> ffi::FaissClusteringParameters {
        ffi::FaissClusteringParameters {
            niter: self.niter,
            nredo: self.nredo,
            verbose: self.verbose as i32,
            spherical: self.spherical as i32,
            int_centroids: self.int_centroids as i32,
            update_index: self.update_index as i32,
            frozen_centroids: self.frozen_centroids as i32,
            min_points_per_centroid: self.min_points_per_centroid,
            max_points_per_centroid: self.max_points_per_centroid,
            seed: self.seed,
            decode_block_size: self.decode_block_size,
        }
    }
}

/// K-means clustering
pub struct Clustering {
    inner: *mut ffi::FaissClustering,
}

impl Clustering {
    /// Create a new k-means clustering instance
    ///
    /// # Arguments
    /// * `d` - Dimension of the vectors
    /// * `k` - Number of clusters
    pub fn new(d: i32, k: i32) -> Result<Self> {
        unsafe {
            let mut clustering = std::ptr::null_mut();
            let ret = ffi::faiss_Clustering_new(&mut clustering, d, k);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: clustering })
        }
    }

    /// Create a new k-means clustering instance with custom parameters
    ///
    /// # Arguments
    /// * `d` - Dimension of the vectors
    /// * `k` - Number of clusters
    /// * `params` - Clustering parameters
    pub fn new_with_params(d: i32, k: i32, params: &ClusteringParameters) -> Result<Self> {
        unsafe {
            let mut clustering = std::ptr::null_mut();
            let ffi_params = params.to_ffi();
            let ret = ffi::faiss_Clustering_new_with_params(&mut clustering, d, k, &ffi_params);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: clustering })
        }
    }

    /// Get the dimension of vectors
    pub fn d(&self) -> usize {
        unsafe { ffi::faiss_Clustering_d(self.inner) }
    }

    /// Get the number of clusters
    pub fn k(&self) -> usize {
        unsafe { ffi::faiss_Clustering_k(self.inner) }
    }

    /// Get the number of iterations
    pub fn niter(&self) -> i32 {
        unsafe { ffi::faiss_Clustering_niter(self.inner) }
    }

    /// Get the number of redo attempts
    pub fn nredo(&self) -> i32 {
        unsafe { ffi::faiss_Clustering_nredo(self.inner) }
    }

    /// Check if verbose mode is enabled
    pub fn verbose(&self) -> bool {
        unsafe { ffi::faiss_Clustering_verbose(self.inner) != 0 }
    }

    /// Check if spherical clustering is enabled
    pub fn spherical(&self) -> bool {
        unsafe { ffi::faiss_Clustering_spherical(self.inner) != 0 }
    }

    /// Check if integer centroids mode is enabled
    pub fn int_centroids(&self) -> bool {
        unsafe { ffi::faiss_Clustering_int_centroids(self.inner) != 0 }
    }

    /// Check if index update mode is enabled
    pub fn update_index(&self) -> bool {
        unsafe { ffi::faiss_Clustering_update_index(self.inner) != 0 }
    }

    /// Check if frozen centroids mode is enabled
    pub fn frozen_centroids(&self) -> bool {
        unsafe { ffi::faiss_Clustering_frozen_centroids(self.inner) != 0 }
    }

    /// Get the minimum points per centroid
    pub fn min_points_per_centroid(&self) -> i32 {
        unsafe { ffi::faiss_Clustering_min_points_per_centroid(self.inner) }
    }

    /// Get the maximum points per centroid
    pub fn max_points_per_centroid(&self) -> i32 {
        unsafe { ffi::faiss_Clustering_max_points_per_centroid(self.inner) }
    }

    /// Get the random seed
    pub fn seed(&self) -> i32 {
        unsafe { ffi::faiss_Clustering_seed(self.inner) }
    }

    /// Get the decode block size
    pub fn decode_block_size(&self) -> usize {
        unsafe { ffi::faiss_Clustering_decode_block_size(self.inner) }
    }

    /// Get iteration statistics after training
    ///
    /// # Returns
    /// A vector of statistics for each iteration
    pub fn iteration_stats(&self) -> Result<Vec<ClusteringIterationStats>> {
        unsafe {
            let mut stats_ptr: *mut ffi::FaissClusteringIterationStats = std::ptr::null_mut();
            let mut size: usize = 0;

            ffi::faiss_Clustering_iteration_stats(self.inner, &mut stats_ptr, &mut size);

            if stats_ptr.is_null() || size == 0 {
                return Ok(Vec::new());
            }

            // Convert each iteration stats
            let mut stats = Vec::with_capacity(size);
            for i in 0..size {
                // Use byte offset to handle potential zero-sized types
                let stat_ptr = (stats_ptr as *const u8)
                    .add(i * std::mem::size_of::<ffi::FaissClusteringIterationStats>())
                    as *mut ffi::FaissClusteringIterationStats;
                stats.push(ClusteringIterationStats::from_ptr(stat_ptr));
            }

            Ok(stats)
        }
    }

    /// Train the clustering on a set of vectors
    ///
    /// # Arguments
    /// * `n` - Number of training vectors
    /// * `x` - Training vectors (n * d values)
    /// * `index` - Index to use for assignments (must support add and search)
    pub fn train(&mut self, n: i64, x: &[f32], index: &mut dyn IndexTrait) -> Result<()> {
        let d = self.d();
        let expected_len = (n as usize) * d;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} values (n={} * d={}), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        unsafe {
            let ret = ffi::faiss_Clustering_train(self.inner, n, x.as_ptr(), index.inner_ptr());
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }
        Ok(())
    }

    /// Get the cluster centroids
    ///
    /// # Returns
    /// A vector of centroids (k * d values)
    pub fn centroids(&self) -> Result<Vec<f32>> {
        unsafe {
            let mut centroids_ptr: *mut f32 = std::ptr::null_mut();
            let mut size: usize = 0;

            ffi::faiss_Clustering_centroids(self.inner, &mut centroids_ptr, &mut size);

            if centroids_ptr.is_null() {
                anyhow::bail!("Failed to get centroids");
            }

            // Copy the centroids data
            let centroids = std::slice::from_raw_parts(centroids_ptr, size).to_vec();
            Ok(centroids)
        }
    }
}

impl Drop for Clustering {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_Clustering_free(self.inner);
            }
        }
    }
}

unsafe impl Send for Clustering {}
