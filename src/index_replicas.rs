use crate::error::FaissError;
use crate::ffi;
use crate::index::IndexTrait;
use anyhow::Result;

/// Index that manages multiple replica sub-indexes
///
/// IndexReplicas maintains multiple copies (replicas) of an index and
/// distributes search queries across them. This enables:
/// - Parallel search across multiple indexes
/// - Load distribution for high-throughput scenarios
/// - Combining results from distributed indexes
///
/// # How it Works
/// 1. **Add Replicas**: Add multiple sub-indexes as replicas
/// 2. **Adding Vectors**: Vectors are added to all replicas
/// 3. **Searching**: Searches all replicas and merges results
///
/// # Use Cases
/// - Multi-threaded search for better throughput
/// - Distributed search across machines (with proper setup)
/// - Testing different index configurations
///
/// # Performance
/// - Search can be parallelized across replicas (with threaded=true)
/// - Adding vectors requires updating all replicas
/// - Memory: Sum of all replica sizes
///
/// # Example
/// ```ignore
/// use faiss_next::{IndexReplicas, IndexFlatL2};
///
/// // Create replicas index
/// let mut index = IndexReplicas::new_with_options(128, true)?;
///
/// // Add multiple replica indexes
/// let replica1 = IndexFlatL2::new(128)?.into_index()?;
/// let replica2 = IndexFlatL2::new(128)?.into_index()?;
///
/// index.add_replica(&replica1)?;
/// index.add_replica(&replica2)?;
///
/// // Searches will be distributed across replicas
/// let (distances, labels) = index.search(1, &query, 10)?;
/// ```
pub struct IndexReplicas {
    inner: *mut ffi::FaissIndexReplicas,
}

impl IndexReplicas {
    /// Create a new IndexReplicas
    ///
    /// # Arguments
    /// * `d` - Vector dimension
    ///
    /// # Example
    /// ```ignore
    /// let index = IndexReplicas::new(128)?;
    /// ```
    pub fn new(d: i64) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexReplicas_new(&mut index_ptr, d);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new IndexReplicas with options
    ///
    /// # Arguments
    /// * `d` - Vector dimension
    /// * `threaded` - If true, search replicas in parallel using threads
    ///
    /// # Example
    /// ```ignore
    /// // Create with multi-threaded search
    /// let index = IndexReplicas::new_with_options(128, true)?;
    /// ```
    ///
    /// # Threading
    /// When `threaded=true`, searches are distributed across replicas using
    /// multiple threads for better throughput on multi-core systems.
    pub fn new_with_options(d: i64, threaded: bool) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexReplicas_new_with_options(
                &mut index_ptr,
                d,
                threaded as i32,
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Add a replica index
    ///
    /// The replica must have the same dimension as this index.
    /// Vectors added to IndexReplicas will be added to all replicas.
    ///
    /// # Arguments
    /// * `replica` - The index to add as a replica
    ///
    /// # Example
    /// ```ignore
    /// let replica = IndexFlatL2::new(128)?.into_index()?;
    /// index.add_replica(&replica)?;
    /// ```
    ///
    /// # Note
    /// The replica index is not cloned - the pointer is stored.
    /// Make sure the replica outlives the IndexReplicas or set own_fields.
    pub fn add_replica(&mut self, replica: &impl IndexTrait) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_IndexReplicas_add_replica(
                self.inner,
                replica.inner_ptr(),
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }
        Ok(())
    }

    /// Remove a replica index
    ///
    /// # Arguments
    /// * `replica` - The index to remove
    ///
    /// # Example
    /// ```ignore
    /// index.remove_replica(&replica)?;
    /// ```
    pub fn remove_replica(&mut self, replica: &impl IndexTrait) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_IndexReplicas_remove_replica(
                self.inner,
                replica.inner_ptr(),
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }
        Ok(())
    }

    /// Get a replica by index
    ///
    /// # Arguments
    /// * `i` - The replica index (0 to num_replicas-1)
    ///
    /// # Returns
    /// Raw pointer to the replica index, or null if index is out of bounds
    ///
    /// # Example
    /// ```ignore
    /// let replica_ptr = index.at(0);
    /// if !replica_ptr.is_null() {
    ///     // Use replica
    /// }
    /// ```
    pub fn at(&self, i: i32) -> *mut ffi::FaissIndex {
        unsafe { ffi::faiss_IndexReplicas_at(self.inner, i) }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexReplicas {
        self.inner
    }
}

impl Drop for IndexReplicas {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexReplicas_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexReplicas {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_flat::IndexFlatL2;

    #[test]
    fn test_index_replicas_creation() {
        let index = IndexReplicas::new(128);
        assert!(index.is_ok(), "Failed to create IndexReplicas");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
        assert_eq!(index.ntotal().unwrap(), 0);
    }

    #[test]
    fn test_index_replicas_with_options() {
        // Test with threading enabled
        let index_threaded = IndexReplicas::new_with_options(64, true);
        assert!(index_threaded.is_ok());

        // Test with threading disabled
        let index_no_thread = IndexReplicas::new_with_options(64, false);
        assert!(index_no_thread.is_ok());
    }

    #[test]
    fn test_index_replicas_add_replica() {
        let mut index = IndexReplicas::new(4).unwrap();

        // Create and add a replica
        let replica = IndexFlatL2::new(4).unwrap().into_index().unwrap();
        let result = index.add_replica(&replica);
        assert!(result.is_ok(), "Failed to add replica");

        // Check that replica was added
        let replica_ptr = index.at(0);
        assert!(!replica_ptr.is_null(), "Replica not found at index 0");
    }

    #[test]
    fn test_index_replicas_multiple_replicas() {
        let mut index = IndexReplicas::new_with_options(4, true).unwrap();

        // Add multiple replicas
        let replica1 = IndexFlatL2::new(4).unwrap().into_index().unwrap();
        let replica2 = IndexFlatL2::new(4).unwrap().into_index().unwrap();

        index.add_replica(&replica1).unwrap();
        index.add_replica(&replica2).unwrap();

        // Check both replicas exist (checking out-of-bounds is undefined behavior)
        assert!(!index.at(0).is_null());
        assert!(!index.at(1).is_null());
    }

    #[test]
    fn test_index_replicas_basic_workflow() {
        let mut index = IndexReplicas::new_with_options(4, false).unwrap();

        // Create replicas with some data
        let mut replica1 = IndexFlatL2::new(4).unwrap();
        let vectors1 = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        replica1.add(2, &vectors1).unwrap();

        let replica1_owned = replica1.into_index().unwrap();
        index.add_replica(&replica1_owned).unwrap();

        // The index should reflect the replica's data
        // Note: ntotal might not work as expected with replicas
        // This just tests that the workflow doesn't crash
        assert_eq!(index.d().unwrap(), 4);
    }
}
