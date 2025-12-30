//! Index cloning functionality.
//! 
//! This module provides functions to clone Faiss indexes, creating deep copies
//! of index structures that can be used independently of the original.

use std::ptr::null_mut;

use crate::{
    error::Result,
    index::{IndexOwned},
};
use faiss_next_sys as ffi;

/// Clone a Faiss index, creating a deep copy.
/// 
/// This function creates a deep copy of the provided index that can be used
/// independently of the original. The cloned index maintains the same structure
/// and content as the original.
/// 
/// # Arguments
/// 
/// * `index` - The index to clone
/// 
/// # Returns
/// 
/// A new index that is a deep copy of the original
/// 
/// # Example
/// 
/// ```no_run
/// # use faiss_next::{index_factory, clone_index::clone_index, index::MetricType};
/// let original_index = index_factory(128, "Flat", MetricType::L2)?;
/// let cloned_index = clone_index(&original_index)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn clone_index(index: &impl crate::index::IndexTrait) -> Result<IndexOwned> {
    let mut inner = null_mut();
    ffi::ok!(faiss_clone_index, index.inner(), &mut inner)?;
    Ok(IndexOwned::new(inner))
}

// Binary index cloning functionality is not consistently available across all
// Faiss builds. If needed, it would need to be conditionally compiled based
// on platform availability.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{index_factory::index_factory, index::{IndexTrait, MetricType}};

    #[test]
    fn test_clone_index() -> Result<()> {
        // Create an original index
        let mut original_index = index_factory(4, "Flat", MetricType::METRIC_L2)?;
        
        // Verify original index properties
        assert_eq!(original_index.d(), 4);
        assert_eq!(original_index.metric_type(), MetricType::METRIC_L2);
        assert_eq!(original_index.ntotal(), 0);
        
        // Clone the index
        let cloned_index = clone_index(&original_index)?;
        
        // Verify cloned index has same properties
        assert_eq!(cloned_index.d(), 4);
        assert_eq!(cloned_index.metric_type(), MetricType::METRIC_L2);
        assert_eq!(cloned_index.ntotal(), 0);
        
        // Add some data to the original index
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 vectors of dimension 4
        original_index.add(&data)?;
        assert_eq!(original_index.ntotal(), 2);
        
        // The cloned index should still have 0 elements since it's a deep copy
        assert_eq!(cloned_index.ntotal(), 0);
        
        Ok(())
    }
}
