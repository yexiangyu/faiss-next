use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, MetricType};
use crate::index_binary::IndexBinaryOwned;
use anyhow::Result;
use std::ffi::CString;

/// Create a FAISS index from a factory description string
///
/// The index_factory function interprets a string to produce a composite FAISS index.
/// The string is a comma-separated list of components that typically includes:
/// - A preprocessing component (optional)
/// - An inverted file structure (optional)
/// - An encoding component
///
/// # Arguments
/// * `d` - Dimension of the vectors
/// * `description` - Factory description string
/// * `metric` - Distance metric to use
///
/// # Description String Format
///
/// The description string can contain the following components:
///
/// ## Preprocessing / Vector Transforms
/// - `PCA64`, `PCAR64`, `PCAW64`, `PCAWR64` - PCA dimensionality reduction
///   - W = with whitening, R = with random rotation
/// - `OPQ16`, `OPQ16_64` - Optimized Product Quantization preprocessing
///
/// ## Inverted File Structures
/// - `IVF4096` - Inverted file with 4096 clusters (flat quantizer)
/// - `IVF65536_HNSW32` - IVF with HNSW graph quantizer
/// - `IMI2x8` - Inverted multi-index with 2x8 bits (65536 lists)
/// - `HNSW32` - Hierarchical Navigable Small World graph
///
/// ## Encoding Components
/// - `Flat` - No encoding, store vectors as-is
/// - `PQ8` - Product Quantization with 8 bytes
/// - `SQ8` - Scalar Quantization with 8 bits
/// - `PQFastScan` - Fast scan variant of PQ
///
/// # Examples
///
/// ```ignore
/// // Simple flat index with PCA preprocessing
/// let index = index_factory(128, "PCA80,Flat", MetricType::L2)?;
///
/// // IVF index with PQ encoding
/// let index = index_factory(128, "IVF4096,PQ32", MetricType::L2)?;
///
/// // Complex index with OPQ, IMI, and PQ refinement
/// let index = index_factory(128, "OPQ16_64,IMI2x8,PQ8+16", MetricType::L2)?;
///
/// // HNSW graph index
/// let index = index_factory(128, "HNSW32", MetricType::L2)?;
/// ```
///
/// # Notes
///
/// - The resulting index may need to be trained before use (check `is_trained()`)
/// - For IVF indices, you need roughly 30-100 training vectors per cluster
/// - Some combinations may not be valid - FAISS will return an error
///
pub fn index_factory(d: i32, description: &str, metric: MetricType) -> Result<IndexOwned> {
    let c_description = CString::new(description)?;

    unsafe {
        let mut index_ptr = std::ptr::null_mut();
        let ret = ffi::faiss_index_factory(
            &mut index_ptr,
            d,
            c_description.as_ptr(),
            metric.to_code(),
        );

        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }

        IndexOwned::from_raw(index_ptr)
    }
}

/// Create a binary FAISS index from a factory description string
///
/// Similar to `index_factory`, but creates indices for binary vectors using Hamming distance.
/// Binary indices are useful for hashing-based search methods and work with binary (0/1) vectors.
///
/// # Arguments
/// * `d` - Dimension of the binary vectors (in bits)
/// * `description` - Factory description string
///
/// # Description String Format
///
/// Binary index descriptions follow similar patterns to regular indices:
///
/// ## Binary Index Types
/// - `BFlat` - Flat binary index (exhaustive search)
/// - `BIVF<n>` - Binary IVF with n clusters
/// - `BHNSW<M>` - Binary HNSW graph with M connections
///
/// # Examples
///
/// ```ignore
/// // Flat binary index for 256-bit vectors
/// let index = index_binary_factory(256, "BFlat")?;
///
/// // Binary IVF index with 1024 clusters
/// let index = index_binary_factory(256, "BIVF1024")?;
/// ```
///
/// # Notes
///
/// - Binary indices use Hamming distance by default
/// - The dimension `d` is in bits, not bytes
/// - Input vectors should be packed into bytes (d/8 bytes per vector)
/// - Some binary indices may need training before use
///
pub fn index_binary_factory(d: i32, description: &str) -> Result<IndexBinaryOwned> {
    let c_description = CString::new(description)?;

    unsafe {
        let mut index_ptr = std::ptr::null_mut();
        let ret = ffi::faiss_index_binary_factory(
            &mut index_ptr,
            d,
            c_description.as_ptr(),
        );

        if let Some(err) = FaissError::from_code(ret) {
            return Err(err.into());
        }

        IndexBinaryOwned::from_raw(index_ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexTrait;
    use crate::index_binary::IndexBinaryTrait;

    #[test]
    fn test_flat_index() {
        let index = index_factory(128, "Flat", MetricType::L2);
        assert!(index.is_ok(), "Failed to create Flat index");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
        assert!(index.is_trained().unwrap());
    }

    #[test]
    fn test_pca_flat_index() {
        let index = index_factory(128, "PCA80,Flat", MetricType::L2);
        assert!(index.is_ok(), "Failed to create PCA80,Flat index");

        let index = index.unwrap();
        // Note: The index dimension remains 128 (input dimension),
        // but internally PCA transform reduces to 80 dimensions
        assert_eq!(index.d().unwrap(), 128);
    }

    #[test]
    fn test_ivf_index() {
        let index = index_factory(128, "IVF256,Flat", MetricType::L2);
        assert!(index.is_ok(), "Failed to create IVF256,Flat index");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
        assert!(!index.is_trained().unwrap()); // IVF needs training
    }

    #[test]
    fn test_hnsw_index() {
        let index = index_factory(128, "HNSW32", MetricType::L2);
        assert!(index.is_ok(), "Failed to create HNSW32 index");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
        assert!(index.is_trained().unwrap()); // HNSW doesn't need training
    }

    #[test]
    fn test_inner_product_metric() {
        let index = index_factory(128, "Flat", MetricType::InnerProduct);
        assert!(index.is_ok(), "Failed to create Flat index with InnerProduct");

        let index = index.unwrap();
        assert_eq!(index.metric_type().unwrap(), MetricType::InnerProduct);
    }

    #[test]
    fn test_invalid_description() {
        let index = index_factory(128, "InvalidDescription123", MetricType::L2);
        assert!(index.is_err(), "Should fail with invalid description");
    }

    #[test]
    fn test_binary_flat_index() {
        let index = index_binary_factory(256, "BFlat");
        assert!(index.is_ok(), "Failed to create BFlat index");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 256);
        assert!(index.is_trained().unwrap());
    }

    #[test]
    fn test_binary_ivf_index() {
        let index = index_binary_factory(256, "BIVF256");
        assert!(index.is_ok(), "Failed to create BIVF256 index");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 256);
        // Binary IVF needs training
        assert!(!index.is_trained().unwrap());
    }
}
