use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, IndexTrait, MetricType};
use anyhow::Result;

/// A trait for flat (exhaustive search) FAISS indices
///
/// Flat indices store vectors without compression and perform exact search
/// by computing distances to all stored vectors. This is the most accurate
/// but also the slowest search method for large datasets.
///
/// # Features
/// - Exact search (no approximation)
/// - No training required
/// - Direct access to stored vectors
///
/// # Performance
/// - Search time: O(n*d) where n is number of vectors, d is dimension
/// - Memory: O(n*d) float values
/// - Best for: small to medium datasets (<1M vectors)
pub trait IndexFlatTrait: IndexTrait {
    /// Get the inner pointer as a FaissIndexFlat
    fn inner_flat_ptr(&self) -> *mut ffi::FaissIndexFlat;

    /// Get direct access to the stored vectors
    ///
    /// # Returns
    /// A tuple of (pointer to data, size in float values)
    ///
    /// # Safety
    /// The returned pointer is only valid until the next add/remove operation.
    /// The caller must ensure the pointer is not used after the index is modified.
    ///
    /// # Note
    /// The size is in number of float values, not vectors. Divide by dimension
    /// to get the number of vectors.
    fn xb(&self) -> (*mut f32, usize) {
        unsafe {
            let mut data_ptr = std::ptr::null_mut();
            let mut size = 0usize;
            ffi::faiss_IndexFlat_xb(self.inner_flat_ptr(), &mut data_ptr, &mut size);
            (data_ptr, size)
        }
    }

    /// Get the stored vectors as a slice
    ///
    /// # Returns
    /// A slice containing all stored vectors (size = ntotal * d)
    ///
    /// # Safety
    /// The returned slice is only valid until the next add/remove operation.
    ///
    /// # Example
    /// ```ignore
    /// let vectors = index.xb_slice();
    /// let num_vectors = vectors.len() / index.d()? as usize;
    /// ```
    fn xb_slice(&self) -> &[f32] {
        unsafe {
            let (data_ptr, size) = self.xb();
            if data_ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(data_ptr, size)
            }
        }
    }
}

/// A flat (exhaustive search) FAISS index
///
/// Generic flat index supporting any metric type.
pub struct IndexFlat {
    inner: *mut ffi::FaissIndexFlat,
}

impl IndexFlat {
    /// Create a new flat index with default parameters
    ///
    /// # Returns
    /// An empty flat index that needs to be initialized with `new_with`
    ///
    /// # Note
    /// You probably want to use `new_with` or the specialized constructors
    /// `IndexFlatL2::new` or `IndexFlatIP::new` instead.
    pub fn new() -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexFlat_new(&mut index_ptr);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new flat index with specified dimension and metric
    ///
    /// # Arguments
    /// * `d` - Dimension of the vectors
    /// * `metric` - Distance metric to use (L2, InnerProduct, etc.)
    ///
    /// # Example
    /// ```ignore
    /// let index = IndexFlat::new_with(128, MetricType::L2)?;
    /// ```
    pub fn new_with(d: i32, metric: MetricType) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexFlat_new_with(&mut index_ptr, d as i64, metric.to_code());

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexFlat
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexFlat)` if the index is actually a flat index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let flat_ptr = ffi::faiss_IndexFlat_cast(index.as_ptr());
            if flat_ptr.is_null() {
                None
            } else {
                Some(Self { inner: flat_ptr })
            }
        }
    }

    /// Convert to IndexOwned
    ///
    /// This consumes the IndexFlat and returns it as a generic IndexOwned.
    /// The index pointer is transferred, not copied.
    pub fn into_index(self) -> Result<IndexOwned> {
        let ptr = self.inner;
        std::mem::forget(self); // Don't drop, transfer ownership
        unsafe { IndexOwned::from_raw(ptr as *mut ffi::FaissIndex) }
    }

    /// Get the inner pointer for use with FAISS functions
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexFlat {
        self.inner
    }
}

impl Drop for IndexFlat {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexFlat_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexFlat {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

impl IndexFlatTrait for IndexFlat {
    fn inner_flat_ptr(&self) -> *mut ffi::FaissIndexFlat {
        self.inner
    }
}

/// Flat index specialized for L2 (Euclidean) distance
///
/// This is a convenience wrapper around IndexFlat that is pre-configured
/// to use L2 distance metric.
pub struct IndexFlatL2 {
    inner: *mut ffi::FaissIndexFlatL2,
}

impl IndexFlatL2 {
    /// Create a new flat L2 index
    ///
    /// # Arguments
    /// * `d` - Dimension of the vectors
    ///
    /// # Example
    /// ```ignore
    /// let index = IndexFlatL2::new(128)?;
    /// ```
    pub fn new(d: i32) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexFlatL2_new_with(&mut index_ptr, d as i64);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexFlatL2
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexFlatL2)` if the index is actually a flat L2 index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let flat_ptr = ffi::faiss_IndexFlatL2_cast(index.as_ptr());
            if flat_ptr.is_null() {
                None
            } else {
                Some(Self { inner: flat_ptr })
            }
        }
    }

    /// Convert to IndexOwned
    pub fn into_index(self) -> Result<IndexOwned> {
        let ptr = self.inner;
        std::mem::forget(self);
        unsafe { IndexOwned::from_raw(ptr as *mut ffi::FaissIndex) }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexFlatL2 {
        self.inner
    }
}

impl Drop for IndexFlatL2 {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexFlatL2_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexFlatL2 {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

impl IndexFlatTrait for IndexFlatL2 {
    fn inner_flat_ptr(&self) -> *mut ffi::FaissIndexFlat {
        self.inner as *mut ffi::FaissIndexFlat
    }
}

/// Flat index specialized for Inner Product (maximum inner product search)
///
/// This is a convenience wrapper around IndexFlat that is pre-configured
/// to use inner product as the similarity metric.
///
/// # Note
/// Inner product search finds vectors with the highest dot product with
/// the query vector. For normalized vectors, this is equivalent to cosine
/// similarity.
pub struct IndexFlatIP {
    inner: *mut ffi::FaissIndexFlatIP,
}

impl IndexFlatIP {
    /// Create a new flat inner product index
    ///
    /// # Arguments
    /// * `d` - Dimension of the vectors
    ///
    /// # Example
    /// ```ignore
    /// let index = IndexFlatIP::new(128)?;
    /// ```
    pub fn new(d: i32) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexFlatIP_new_with(&mut index_ptr, d as i64);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexFlatIP
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexFlatIP)` if the index is actually a flat IP index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let flat_ptr = ffi::faiss_IndexFlatIP_cast(index.as_ptr());
            if flat_ptr.is_null() {
                None
            } else {
                Some(Self { inner: flat_ptr })
            }
        }
    }

    /// Convert to IndexOwned
    pub fn into_index(self) -> Result<IndexOwned> {
        let ptr = self.inner;
        std::mem::forget(self);
        unsafe { IndexOwned::from_raw(ptr as *mut ffi::FaissIndex) }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexFlatIP {
        self.inner
    }
}

impl Drop for IndexFlatIP {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexFlatIP_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexFlatIP {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

impl IndexFlatTrait for IndexFlatIP {
    fn inner_flat_ptr(&self) -> *mut ffi::FaissIndexFlat {
        self.inner as *mut ffi::FaissIndexFlat
    }
}

/// Compute distances to a subset of vectors
///
/// This is a standalone function that computes distances between query vectors
/// and a specific subset of vectors in a flat index.
///
/// # Arguments
/// * `index` - Any index implementing IndexTrait
/// * `queries` - Query vectors (n * d values)
/// * `labels` - Indices of vectors to compute distances to (n * k values)
///
/// # Returns
/// Distances between queries and the specified vectors (n * k values)
///
/// # Example
/// ```ignore
/// let queries = vec![1.0; 128]; // 1 query vector of dimension 128
/// let labels = vec![0, 1, 2, 3, 4]; // Compare to vectors 0-4
/// let distances = compute_distance_subset(&index, 1, &queries, 5, &labels)?;
/// ```
pub fn compute_distance_subset(
    index: &impl IndexTrait,
    n: i64,
    queries: &[f32],
    k: i64,
    labels: &[i64],
) -> Result<Vec<f32>> {
    let d = index.d()?;
    let expected_queries_len = (n * d as i64) as usize;
    let expected_labels_len = (n * k) as usize;

    if queries.len() != expected_queries_len {
        anyhow::bail!(
            "Expected {} query values (n={} * d={}), got {}",
            expected_queries_len,
            n,
            d,
            queries.len()
        );
    }

    if labels.len() != expected_labels_len {
        anyhow::bail!(
            "Expected {} labels (n={} * k={}), got {}",
            expected_labels_len,
            n,
            k,
            labels.len()
        );
    }

    let mut distances = vec![0.0f32; expected_labels_len];

    unsafe {
        let ret = ffi::faiss_IndexFlat_compute_distance_subset(
            index.inner_ptr(),
            n,
            queries.as_ptr(),
            k,
            distances.as_mut_ptr(),
            labels.as_ptr(),
        );

        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }
    }

    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_flat_l2() {
        let index = IndexFlatL2::new(4);
        assert!(index.is_ok(), "Failed to create IndexFlatL2");

        let mut index = index.unwrap();
        assert_eq!(index.d().unwrap(), 4);
        assert_eq!(index.ntotal().unwrap(), 0);
        assert!(index.is_trained().unwrap());

        // Add some vectors
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(index.add(2, &vectors).is_ok());
        assert_eq!(index.ntotal().unwrap(), 2);

        // Test IndexFlatTrait methods
        let (ptr, size) = index.xb();
        assert!(!ptr.is_null());
        assert_eq!(size, 8);
    }

    #[test]
    fn test_index_flat_ip() {
        let index = IndexFlatIP::new(4);
        assert!(index.is_ok(), "Failed to create IndexFlatIP");

        let mut index = index.unwrap();
        assert_eq!(index.d().unwrap(), 4);
        assert_eq!(index.metric_type().unwrap(), MetricType::InnerProduct);

        // Add some vectors
        let vectors = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(index.add(2, &vectors).is_ok());
        assert_eq!(index.ntotal().unwrap(), 2);

        // Test IndexFlatTrait methods
        let slice = index.xb_slice();
        assert_eq!(slice.len(), 8);
    }

    #[test]
    fn test_index_flat_with_metric() {
        let index = IndexFlat::new_with(4, MetricType::L2);
        assert!(index.is_ok(), "Failed to create IndexFlat");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 4);
        assert_eq!(index.metric_type().unwrap(), MetricType::L2);
    }

    #[test]
    fn test_index_flat_trait() {
        let mut index = IndexFlat::new_with(4, MetricType::L2).unwrap();

        // Add some vectors
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        index.add(2, &vectors).unwrap();

        // Test IndexFlatTrait methods
        let (ptr, size) = index.xb();
        assert!(!ptr.is_null());
        assert_eq!(size, 8); // 2 vectors * 4 dimensions

        let slice = index.xb_slice();
        assert_eq!(slice.len(), 8);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[7], 8.0);
    }

    #[test]
    fn test_search() {
        let mut index = IndexFlatL2::new(4).unwrap();

        // Add some vectors
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, 0.0, // Vector 1
            0.0, 0.0, 1.0, 0.0, // Vector 2
        ];
        index.add(3, &vectors).unwrap();

        // Search for nearest neighbor to [1, 0, 0, 0]
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let (distances, labels) = index.search(1, &query, 1).unwrap();

        assert_eq!(labels[0], 0); // Should find vector 0
        assert!(distances[0] < 0.001); // Should be very close (almost 0)
    }

    #[test]
    fn test_compute_distance_subset() {
        let mut flat = IndexFlat::new_with(4, MetricType::L2).unwrap();

        // Add some vectors
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, 0.0, // Vector 1
            0.0, 0.0, 1.0, 0.0, // Vector 2
            0.0, 0.0, 0.0, 1.0, // Vector 3
        ];
        flat.add(4, &vectors).unwrap();

        // Convert to IndexOwned for the function
        let index_owned = flat.into_index().unwrap();

        // Compute distances from query to vectors 0 and 2
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let labels = vec![0, 2];
        let distances = compute_distance_subset(&index_owned, 1, &query, 2, &labels).unwrap();

        assert_eq!(distances.len(), 2);
        assert!(distances[0] < 0.001); // Distance to vector 0 should be ~0
        assert!((distances[1] - 2.0).abs() < 0.001); // Distance to vector 2 should be ~2
    }
}
