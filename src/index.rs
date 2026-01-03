use crate::error::FaissError;
use crate::ffi;
use crate::id_selector::IDSelectorTrait;
use crate::range_search::RangeSearchResult;
use anyhow::Result;

/// Distance metric types used by FAISS indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetricType {
    /// Inner product (maximum inner product search)
    InnerProduct = 0,
    /// L2 (Euclidean) distance
    L2 = 1,
    /// L1 (Manhattan) distance
    L1 = 2,
    /// L-infinity distance
    Linf = 3,
    /// Lp distance (generalized p-norm)
    Lp = 4,
    /// Canberra distance
    Canberra = 20,
    /// Bray-Curtis dissimilarity
    BrayCurtis = 21,
    /// Jensen-Shannon divergence
    JensenShannon = 22,
}

impl MetricType {
    /// Convert from FAISS metric type constant
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(MetricType::InnerProduct),
            1 => Some(MetricType::L2),
            2 => Some(MetricType::L1),
            3 => Some(MetricType::Linf),
            4 => Some(MetricType::Lp),
            20 => Some(MetricType::Canberra),
            21 => Some(MetricType::BrayCurtis),
            22 => Some(MetricType::JensenShannon),
            _ => None,
        }
    }

    /// Convert to FAISS metric type constant
    pub fn to_code(self) -> u32 {
        self as u32
    }
}

impl From<MetricType> for u32 {
    fn from(metric: MetricType) -> Self {
        metric.to_code()
    }
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::InnerProduct => write!(f, "InnerProduct"),
            MetricType::L2 => write!(f, "L2"),
            MetricType::L1 => write!(f, "L1"),
            MetricType::Linf => write!(f, "Linf"),
            MetricType::Lp => write!(f, "Lp"),
            MetricType::Canberra => write!(f, "Canberra"),
            MetricType::BrayCurtis => write!(f, "BrayCurtis"),
            MetricType::JensenShannon => write!(f, "JensenShannon"),
        }
    }
}

/// An owned wrapper around a FaissIndex pointer
///
/// This struct takes ownership of a FaissIndex pointer and ensures
/// it is properly freed when dropped.
pub struct IndexOwned {
    inner: *mut ffi::FaissIndex,
}

impl IndexOwned {
    /// Create a new IndexOwned from a raw FaissIndex pointer
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer is valid and points to a properly initialized FaissIndex
    /// - The pointer is not used elsewhere after being passed to this function
    /// - The pointer was allocated by FAISS and can be freed with faiss_Index_free
    pub unsafe fn from_raw(ptr: *mut ffi::FaissIndex) -> Result<Self> {
        if ptr.is_null() {
            anyhow::bail!("Cannot create IndexOwned from null pointer");
        }
        Ok(Self { inner: ptr })
    }

    /// Get a reference to the inner raw pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}

impl Drop for IndexOwned {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_Index_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexOwned {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}

// Ensure IndexOwned is Send if the underlying index is thread-safe
// FAISS indices are generally not thread-safe for mutation, so we don't implement Send/Sync by default
// Users can wrap in Arc<Mutex<>> if needed

/// A trait representing common operations for FAISS indices
pub trait IndexTrait {
    /// Get the underlying raw pointer to the FAISS index
    fn inner_ptr(&self) -> *mut ffi::FaissIndex;

    /// Get the dimension of the vectors in the index
    fn d(&self) -> Result<i32> {
        unsafe {
            let d = ffi::faiss_Index_d(self.inner_ptr());
            Ok(d)
        }
    }

    /// Get the number of vectors in the index
    fn ntotal(&self) -> Result<i64> {
        unsafe {
            let n = ffi::faiss_Index_ntotal(self.inner_ptr());
            Ok(n)
        }
    }

    /// Check if the index is trained
    fn is_trained(&self) -> Result<bool> {
        unsafe {
            let trained = ffi::faiss_Index_is_trained(self.inner_ptr());
            Ok(trained != 0)
        }
    }

    /// Get the metric type used by this index
    fn metric_type(&self) -> Result<MetricType> {
        unsafe {
            let code = ffi::faiss_Index_metric_type(self.inner_ptr());
            MetricType::from_code(code)
                .ok_or_else(|| anyhow::anyhow!("Unknown metric type: {}", code))
        }
    }

    /// Train the index on a set of vectors
    ///
    /// # Arguments
    /// * `n` - Number of training vectors
    /// * `x` - Training vectors (n * d values)
    fn train(&mut self, n: i64, x: &[f32]) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * d as i64) as usize;

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
            let ret = ffi::faiss_Index_train(self.inner_ptr(), n, x.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to train index");
            }
        }
        Ok(())
    }

    /// Add vectors to the index
    ///
    /// # Arguments
    /// * `n` - Number of vectors to add
    /// * `x` - Vectors to add (n * d values)
    fn add(&mut self, n: i64, x: &[f32]) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * d as i64) as usize;

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
            let ret = ffi::faiss_Index_add(self.inner_ptr(), n, x.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to add vectors to index");
            }
        }
        Ok(())
    }

    /// Search the index for nearest neighbors
    ///
    /// # Arguments
    /// * `n` - Number of query vectors
    /// * `x` - Query vectors (n * d values)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// A tuple of (distances, labels) where each is a vector of length n*k
    fn search(&self, n: i64, x: &[f32], k: i64) -> Result<(Vec<f32>, Vec<i64>)> {
        let d = self.d()?;
        let expected_len = (n * d as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} values (n={} * d={}), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        let mut distances = vec![0.0f32; (n * k) as usize];
        let mut labels = vec![0i64; (n * k) as usize];

        unsafe {
            let ret = ffi::faiss_Index_search(
                self.inner_ptr(),
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            if ret != 0 {
                anyhow::bail!("Failed to search index");
            }
        }

        Ok((distances, labels))
    }

    /// Range search: find all vectors within a given radius
    ///
    /// # Arguments
    /// * `n` - Number of query vectors
    /// * `x` - Query vectors (n * d values)
    /// * `radius` - Search radius (L2 distance threshold)
    /// * `result` - RangeSearchResult to store the results
    fn range_search(&self, n: i64, x: &[f32], radius: f32, result: &mut RangeSearchResult) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * d as i64) as usize;

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
            let ret = ffi::faiss_Index_range_search(
                self.inner_ptr(),
                n,
                x.as_ptr(),
                radius,
                result.as_ptr(),
            );
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }

        Ok(())
    }

    /// Reset the index, removing all vectors
    fn reset(&mut self) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_Index_reset(self.inner_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to reset index");
            }
        }
        Ok(())
    }

    /// Remove vectors from the index using an ID selector
    ///
    /// # Arguments
    /// * `selector` - The ID selector that determines which IDs to remove
    ///
    /// # Returns
    /// The number of vectors actually removed
    fn remove_ids(&mut self, selector: &dyn IDSelectorTrait) -> Result<usize> {
        unsafe {
            let mut n_removed: usize = 0;
            let ret = ffi::faiss_Index_remove_ids(
                self.inner_ptr(),
                selector.as_ptr(),
                &mut n_removed,
            );

            if ret != 0 {
                anyhow::bail!("Failed to remove IDs from index");
            }

            Ok(n_removed)
        }
    }
}
