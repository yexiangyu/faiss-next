use crate::error::{Error, Result};
use crate::index::{FaissIndex, IndexInner};
use crate::macros::faiss_rc;
use faiss_next_sys as sys;
use std::ptr::null_mut;
pub struct FaissClusteringIterationStats<'a> {
    inner: *mut sys::FaissClusteringIterationStats,
    #[allow(unused)]
    clustering: &'a FaissClustering,
}

impl<'a> FaissClusteringIterationStats<'a> {
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

pub struct FaissClustering {
    inner: *mut sys::FaissClustering,
}

impl Drop for FaissClustering {
    fn drop(&mut self) {
        unsafe { sys::faiss_Clustering_free(self.inner) }
    }
}

///```rust
/// use faiss_next::prelude::*;
/// use ndarray::{Array2, s};
/// use ndarray_rand::*;
///
/// let feats = Array2::random((1024, 128), rand::distributions::Uniform::new(0., 1.));
/// let mut builder = FaissIndex::builder().with_dimension(128).with_description("Flat");
/// #[cfg(feature = "gpu")]
/// {
///     builder = builder.with_gpu(0);
/// }
/// let mut index = builder.build().expect("failed to build index");
/// let mut clustering = FaissClustering::builder().with_d(128).with_k(10).build().expect("failed to build clustering");
/// clustering.train(feats.as_slice().unwrap(), &mut index).expect("failed to train");
///
/// assert_eq!(index.ntotal(), 10);
///
/// let result = index.search(&feats.slice(s![10..11, ..]).as_slice().unwrap(), 1).expect("failed to search");
/// assert_eq!(result.labels.len(), 1);
///
///```
impl FaissClustering {
    pub fn builder() -> FaissClusteringBuilder {
        Default::default()
    }

    pub fn d(&self) -> usize {
        unsafe { sys::faiss_Clustering_d(self.inner) }
    }

    pub fn k(&self) -> usize {
        unsafe { sys::faiss_Clustering_k(self.inner) }
    }

    pub fn centroids(&self) -> Result<Vec<&[f32]>> {
        let mut data = null_mut();
        let mut size = 0;
        unsafe { sys::faiss_Clustering_centroids(self.inner, &mut data, &mut size) }
        Ok(unsafe { std::slice::from_raw_parts(data, size) }
            .chunks(self.d())
            .collect())
    }

    pub fn iteration_stats(&self) -> Result<Vec<FaissClusteringIterationStats<'_>>> {
        let mut stats = null_mut();
        let mut n = 0usize;
        unsafe { sys::faiss_Clustering_iteration_stats(self.inner, &mut stats, &mut n) };
        let mut ret = vec![];
        for i in 0..n {
            ret.push(FaissClusteringIterationStats {
                #[allow(clippy::zst_offset)]
                inner: unsafe { stats.add(i) },
                clustering: self,
            });
        }
        Ok(ret)
    }

    pub fn train(&mut self, x: impl AsRef<[f32]>, index: &mut FaissIndex) -> Result<()> {
        let x = x.as_ref();
        let n = (x.len() / self.d()) as sys::idx_t;
        faiss_rc!({ sys::faiss_Clustering_train(self.inner, n, x.as_ptr(), index.inner()) })?;
        Ok(())
    }
}

pub struct FaissClusteringBuilder {
    params: sys::FaissClusteringParameters,
    d: usize,
    k: usize,
}

impl Default for FaissClusteringBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FaissClusteringBuilder {
    pub fn build(self) -> Result<FaissClustering> {
        if self.d == 0 {
            return Err(Error::InvalidDimension);
        }
        if self.k == 0 {
            return Err(Error::InvalidClusterNumber);
        }

        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_Clustering_new_with_params(
                &mut inner,
                self.d as i32,
                self.k as i32,
                &self.params,
            )
        })?;
        Ok(FaissClustering { inner })
    }

    pub fn new() -> Self {
        let mut params = sys::FaissClusteringParameters {
            niter: 0,
            nredo: 0,
            verbose: 0,
            spherical: 0,
            int_centroids: 0,
            update_index: 0,
            frozen_centroids: 0,
            min_points_per_centroid: 0,
            max_points_per_centroid: 0,
            seed: 0,
            decode_block_size: 0,
        };
        unsafe { sys::faiss_ClusteringParameters_init(&mut params) };
        Self { params, k: 0, d: 0 }
    }

    pub fn with_d(mut self, d: usize) -> Self {
        self.d = d;
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_niter(mut self, niter: i32) -> Self {
        self.params.niter = niter;
        self
    }

    pub fn with_nredo(mut self, nredo: i32) -> Self {
        self.params.nredo = nredo;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.params.verbose = match verbose {
            true => 1,
            false => 0,
        };
        self
    }

    pub fn with_sperical(mut self, spherical: bool) -> Self {
        self.params.spherical = match spherical {
            true => 1,
            false => 0,
        };
        self
    }

    pub fn with_int_centroids(mut self, int_centroids: bool) -> Self {
        self.params.int_centroids = match int_centroids {
            true => 1,
            false => 0,
        };
        self
    }

    pub fn with_update_index(mut self, update_index: bool) -> Self {
        self.params.update_index = match update_index {
            true => 1,
            false => 0,
        };
        self
    }

    pub fn with_frozen_centroids(mut self, frozen_centroids: bool) -> Self {
        self.params.frozen_centroids = match frozen_centroids {
            true => 1,
            false => 0,
        };
        self
    }

    pub fn with_min_points_per_centroid(mut self, min_points_per_centroid: i32) -> Self {
        self.params.min_points_per_centroid = min_points_per_centroid;
        self
    }

    pub fn with_max_points_per_centroid(mut self, max_points_per_centroid: i32) -> Self {
        self.params.max_points_per_centroid = max_points_per_centroid;
        self
    }

    pub fn with_seed(mut self, seed: i32) -> Self {
        self.params.seed = seed;
        self
    }

    pub fn with_decode_block_size(mut self, decode_block_size: usize) -> Self {
        self.params.decode_block_size = decode_block_size;
        self
    }
}

pub struct ClusteringResult {
    pub centroids: Vec<Vec<f32>>,
    pub q_error: Vec<f32>,
}

/// clustering using kmeans, ***cpu*** only
///```rust
/// use faiss_next::cluster::faiss_kmeans_clustering;
/// use ndarray::Array2;
/// use ndarray_rand::*;
///
/// let feats = Array2::random((1024, 128), rand::distributions::Uniform::new(0., 1.));
///
/// faiss_kmeans_clustering(128, 1024, 10, feats.as_slice().unwrap()).unwrap();
///```
pub fn faiss_kmeans_clustering(
    d: usize,
    n: usize,
    k: usize,
    x: impl AsRef<[f32]>,
) -> Result<ClusteringResult> {
    let x = x.as_ref();
    let mut centroids = vec![0.0f32; d * k];
    let mut q_error = vec![0.0f32; k];
    faiss_rc!({
        sys::faiss_kmeans_clustering(
            d,
            n,
            k,
            x.as_ptr(),
            centroids.as_mut_ptr(),
            q_error.as_mut_ptr(),
        )
    })?;
    let centroids = centroids.chunks(d).map(|v| v.to_owned()).collect();
    Ok(ClusteringResult { centroids, q_error })
}
