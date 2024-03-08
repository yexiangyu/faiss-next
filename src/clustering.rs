use crate::{error::Result, index::common::FaissIndexTrait, rc};
use faiss_next_sys as sys;
use std::ptr::{addr_of, addr_of_mut, null_mut};

pub struct FaissClusteringParameters {
    pub inner: sys::FaissClusteringParameters,
}

impl Default for FaissClusteringParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl FaissClusteringParameters {
    pub fn new() -> Self {
        let inner = sys::FaissClusteringParameters::default();
        let mut r = Self { inner };
        unsafe { sys::faiss_ClusteringParameters_init(addr_of_mut!(r.inner)) };
        r
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

    pub fn udpate_index(&self) -> bool {
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
        let mut data = null_mut();
        let mut size = 0usize;
        unsafe {
            sys::faiss_Clustering_centroids(self.inner, addr_of_mut!(data), addr_of_mut!(size))
        };
        let data = unsafe { std::slice::from_raw_parts(data, size) };
        data.chunks(self.d()).collect()
    }

    pub fn iteration_stats(&self) -> &[FaissClusteringIterationStats] {
        let mut data = null_mut();
        let mut size = 0usize;
        unsafe {
            sys::faiss_Clustering_iteration_stats(
                self.inner,
                addr_of_mut!(data),
                addr_of_mut!(size),
            )
        };
        let data = unsafe { std::slice::from_raw_parts(data, size) };
        unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const FaissClusteringIterationStats, size)
        }
    }

    pub fn new(d: i32, k: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_Clustering_new(addr_of_mut!(inner), d, k) })?;
        Ok(Self { inner })
    }

    pub fn new_with_params(d: i32, k: i32, params: &FaissClusteringParameters) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_Clustering_new_with_params(addr_of_mut!(inner), d, k, addr_of!(params.inner))
        })?;
        Ok(Self { inner })
    }

    pub fn train(&mut self, data: &[f32], index: &mut impl FaissIndexTrait) -> Result<()> {
        let n = data.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_Clustering_train(self.inner, n, data.as_ptr(), index.inner()) })?;
        Ok(())
    }
}

pub fn faiss_kmeans_clustering(data: &[f32], d: usize, k: usize) -> Result<(Vec<Vec<f32>>, f32)> {
    let n = data.len() / d;
    let mut centroids = vec![0.0f32; d * k];
    let mut q_error = 0.0f32;
    rc!({
        sys::faiss_kmeans_clustering(
            d,
            n,
            k,
            data.as_ptr(),
            centroids.as_mut_ptr(),
            addr_of_mut!(q_error),
        )
    })?;
    Ok((centroids.chunks(d).map(|c| c.to_vec()).collect(), q_error))
}

pub struct FaissClusteringIterationStats {
    inner: *mut sys::FaissClusteringIterationStats,
}

impl FaissClusteringIterationStats {
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
