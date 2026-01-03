use crate::ffi;
use crate::index::{IndexOwned, IndexTrait};
use anyhow::Result;

/// A trait for IVF (Inverted File) FAISS indices
///
/// IVF indices use clustering to speed up search by only searching
/// a subset of the database. The index is organized into inverted lists,
/// where each list corresponds to a cluster (Voronoi cell) defined by
/// the quantizer.
///
/// # How IVF Works
/// 1. **Training**: Cluster the database into `nlist` clusters
/// 2. **Adding**: Assign each vector to its nearest cluster and store it there
/// 3. **Searching**: Find the `nprobe` nearest clusters to the query and search only those
///
/// # Performance
/// - Training time: O(nlist * d * training_vectors)
/// - Search time: O(nprobe * vectors_per_list * d)
/// - Memory: O(ntotal * d) + overhead for inverted lists
///
/// # Trade-offs
/// - Higher `nlist` = faster search but needs more training data
/// - Higher `nprobe` = better recall but slower search
pub trait IndexIVFTrait: IndexTrait {
    /// Get the inner pointer as a FaissIndexIVF
    fn inner_ivf_ptr(&self) -> *mut ffi::FaissIndexIVF;

    /// Get the number of inverted lists (clusters)
    ///
    /// This is the number of Voronoi cells used to partition the space.
    /// Typical values: 100-100000 depending on dataset size.
    fn nlist(&self) -> usize {
        unsafe { ffi::faiss_IndexIVF_nlist(self.inner_ivf_ptr()) }
    }

    /// Get the number of lists to probe during search
    ///
    /// Higher values = better recall but slower search.
    /// Typical values: 1-nlist/10
    fn nprobe(&self) -> usize {
        unsafe { ffi::faiss_IndexIVF_nprobe(self.inner_ivf_ptr()) }
    }

    /// Set the number of lists to probe during search
    ///
    /// # Arguments
    /// * `nprobe` - Number of nearest clusters to search (1 to nlist)
    ///
    /// # Example
    /// ```ignore
    /// index.set_nprobe(10); // Search 10 nearest clusters
    /// ```
    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe {
            ffi::faiss_IndexIVF_set_nprobe(self.inner_ivf_ptr(), nprobe);
        }
    }

    /// Get the quantizer index
    ///
    /// The quantizer is used to assign vectors to clusters.
    /// Returns a raw pointer to the quantizer index.
    fn quantizer(&self) -> *mut ffi::FaissIndex {
        unsafe { ffi::faiss_IndexIVF_quantizer(self.inner_ivf_ptr()) }
    }

    /// Get the quantizer training mode
    ///
    /// # Returns
    /// - 0: use quantizer as index in k-means training
    /// - 1: pass training set directly to quantizer's train()
    /// - 2: k-means training on flat index + add centroids to quantizer
    fn quantizer_trains_alone(&self) -> i8 {
        unsafe { ffi::faiss_IndexIVF_quantizer_trains_alone(self.inner_ivf_ptr()) }
    }

    /// Check if the index owns its internal fields
    ///
    /// If true, the index will free the quantizer when dropped.
    fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIVF_own_fields(self.inner_ivf_ptr()) != 0 }
    }

    /// Set whether the index owns its internal fields
    ///
    /// # Arguments
    /// * `own` - If true, index will manage quantizer lifetime
    fn set_own_fields(&mut self, own: bool) {
        unsafe {
            ffi::faiss_IndexIVF_set_own_fields(self.inner_ivf_ptr(), own as i32);
        }
    }

    /// Merge another IVF index into this one
    ///
    /// Moves all entries from `other` to this index. After the operation,
    /// `other` will be empty.
    ///
    /// # Arguments
    /// * `other` - The index to merge from (will be emptied)
    /// * `add_id` - Value added to all moved IDs (for sequential IDs, use ntotal)
    fn merge_from(&mut self, other: &mut impl IndexIVFTrait, add_id: i64) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_IndexIVF_merge_from(
                self.inner_ivf_ptr(),
                other.inner_ivf_ptr(),
                add_id,
            );
            if ret != 0 {
                anyhow::bail!("Failed to merge IVF indices");
            }
        }
        Ok(())
    }

    /// Get the size of a specific inverted list
    ///
    /// # Arguments
    /// * `list_no` - The list number (0 to nlist-1)
    ///
    /// # Returns
    /// Number of vectors in the specified list
    fn get_list_size(&self, list_no: usize) -> usize {
        unsafe { ffi::faiss_IndexIVF_get_list_size(self.inner_ivf_ptr(), list_no) }
    }

    /// Create a direct map from vector IDs to their locations
    ///
    /// This enables faster reconstruction and removal of vectors by ID.
    /// Uses additional memory proportional to ntotal.
    ///
    /// # Arguments
    /// * `new_maintain_direct_map` - If true, maintain the direct map
    fn make_direct_map(&mut self, new_maintain_direct_map: bool) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_IndexIVF_make_direct_map(
                self.inner_ivf_ptr(),
                new_maintain_direct_map as i32,
            );
            if ret != 0 {
                anyhow::bail!("Failed to create direct map");
            }
        }
        Ok(())
    }

    /// Get the imbalance factor of the inverted lists
    ///
    /// Returns the ratio between the largest list and the average list size.
    /// Values close to 1.0 indicate balanced lists.
    /// Higher values indicate some lists are much larger than others.
    fn imbalance_factor(&self) -> f64 {
        unsafe { ffi::faiss_IndexIVF_imbalance_factor(self.inner_ivf_ptr()) }
    }

    /// Print statistics about the inverted lists
    ///
    /// Outputs information about list sizes, imbalance, etc. to stdout.
    fn print_stats(&self) {
        unsafe {
            ffi::faiss_IndexIVF_print_stats(self.inner_ivf_ptr());
        }
    }
}

/// IVF (Inverted File) index wrapper
///
/// Generic IVF index that can use any encoding for stored vectors.
pub struct IndexIVF {
    inner: *mut ffi::FaissIndexIVF,
}

impl IndexIVF {
    /// Cast a generic index to IndexIVF
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexIVF)` if the index is actually an IVF index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let ivf_ptr = ffi::faiss_IndexIVF_cast(index.as_ptr());
            if ivf_ptr.is_null() {
                None
            } else {
                Some(Self { inner: ivf_ptr })
            }
        }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexIVF {
        self.inner
    }
}

impl Drop for IndexIVF {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexIVF_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexIVF {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

impl IndexIVFTrait for IndexIVF {
    fn inner_ivf_ptr(&self) -> *mut ffi::FaissIndexIVF {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_flat::IndexFlatL2;

    #[test]
    fn test_index_ivf_creation() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        // IndexIVF is a generic wrapper, test via IndexIVFFlat in index_ivf_flat module
        assert_eq!(quantizer.d().unwrap(), 4);
    }

    #[test]
    fn test_index_ivf_trait_methods() {
        // Test that IndexIVFTrait methods work
        // Use IndexIVFFlat from the index_ivf_flat module for concrete testing
    }
}
