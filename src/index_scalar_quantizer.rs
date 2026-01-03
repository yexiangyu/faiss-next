use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, IndexTrait, MetricType};
use anyhow::Result;

/// Scalar quantizer type
///
/// Determines how many bits are used per vector component and
/// whether the quantization range is shared across dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantizerType {
    /// 8 bits per component (per-dimension ranges)
    QT8bit = ffi::FaissQuantizerType_QT_8bit,
    /// 4 bits per component (per-dimension ranges)
    QT4bit = ffi::FaissQuantizerType_QT_4bit,
    /// 8 bits per component (shared range for all dimensions)
    QT8bitUniform = ffi::FaissQuantizerType_QT_8bit_uniform,
    /// 4 bits per component (shared range)
    QT4bitUniform = ffi::FaissQuantizerType_QT_4bit_uniform,
    /// 16-bit floating point
    QTFp16 = ffi::FaissQuantizerType_QT_fp16,
    /// Direct 8-bit indexing of uint8s
    QT8bitDirect = ffi::FaissQuantizerType_QT_8bit_direct,
    /// 6 bits per component
    QT6bit = ffi::FaissQuantizerType_QT_6bit,
    /// Brain floating point 16
    QTBf16 = ffi::FaissQuantizerType_QT_bf16,
    /// Direct 8-bit indexing of signed int8s [-128, 127]
    QT8bitDirectSigned = ffi::FaissQuantizerType_QT_8bit_direct_signed,
}

impl QuantizerType {
    /// Convert to C enum value
    pub fn to_code(self) -> u32 {
        self as u32
    }

    /// Convert from C enum value
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            ffi::FaissQuantizerType_QT_8bit => Some(Self::QT8bit),
            ffi::FaissQuantizerType_QT_4bit => Some(Self::QT4bit),
            ffi::FaissQuantizerType_QT_8bit_uniform => Some(Self::QT8bitUniform),
            ffi::FaissQuantizerType_QT_4bit_uniform => Some(Self::QT4bitUniform),
            ffi::FaissQuantizerType_QT_fp16 => Some(Self::QTFp16),
            ffi::FaissQuantizerType_QT_8bit_direct => Some(Self::QT8bitDirect),
            ffi::FaissQuantizerType_QT_6bit => Some(Self::QT6bit),
            ffi::FaissQuantizerType_QT_bf16 => Some(Self::QTBf16),
            ffi::FaissQuantizerType_QT_8bit_direct_signed => Some(Self::QT8bitDirectSigned),
            _ => None,
        }
    }
}

/// Scalar Quantizer index
///
/// IndexScalarQuantizer compresses vectors by quantizing each dimension
/// to a small number of bits (typically 4, 6, or 8). This reduces memory
/// usage significantly while providing approximate search.
///
/// # Quantization
/// Each vector component is quantized independently using:
/// - **Per-dimension**: Each dimension has its own min/max range
/// - **Uniform**: All dimensions share the same range
///
/// # Characteristics
/// - Compact storage: 4-8 bits per dimension vs 32 bits (float)
/// - Approximate distances
/// - Fast search (optimized bit operations)
/// - Training required to learn quantization ranges
///
/// # Example
/// ```ignore
/// use faiss_next::{IndexScalarQuantizer, QuantizerType, MetricType};
///
/// // Create index with 8-bit quantization
/// let mut index = IndexScalarQuantizer::new_with(
///     128,
///     QuantizerType::QT8bit,
///     MetricType::L2
/// )?;
///
/// // Train to learn quantization ranges
/// index.train(10000, &training_data)?;
///
/// // Add and search
/// index.add(1000, &vectors)?;
/// let (distances, labels) = index.search(1, &query, 10)?;
/// ```
pub struct IndexScalarQuantizer {
    inner: *mut ffi::FaissIndexScalarQuantizer,
}

impl IndexScalarQuantizer {
    /// Create a new IndexScalarQuantizer (requires manual setup)
    pub fn new() -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexScalarQuantizer_new(&mut index_ptr);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new IndexScalarQuantizer with parameters
    ///
    /// # Arguments
    /// * `d` - Vector dimension
    /// * `qt` - Quantizer type (number of bits per component)
    /// * `metric` - Distance metric
    ///
    /// # Example
    /// ```ignore
    /// let index = IndexScalarQuantizer::new_with(
    ///     128,
    ///     QuantizerType::QT8bit,
    ///     MetricType::L2
    /// )?;
    /// ```
    pub fn new_with(d: i64, qt: QuantizerType, metric: MetricType) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexScalarQuantizer_new_with(
                &mut index_ptr,
                d,
                qt.to_code(),
                metric.to_code(),
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexScalarQuantizer
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let ptr = ffi::faiss_IndexScalarQuantizer_cast(index.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(Self { inner: ptr })
            }
        }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexScalarQuantizer {
        self.inner
    }
}

impl Drop for IndexScalarQuantizer {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexScalarQuantizer_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexScalarQuantizer {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

/// IVF index with scalar quantization
///
/// IndexIVFScalarQuantizer combines IVF clustering with scalar quantization
/// for efficient approximate search with reduced memory usage.
///
/// # How it Works
/// 1. **Clustering**: Partition space into nlist Voronoi cells
/// 2. **Quantization**: Compress vectors using scalar quantization
/// 3. **Search**: Search nprobe nearest clusters with quantized vectors
///
/// # Memory Usage
/// - Per vector: nlist overhead + (d * bits_per_component / 8) bytes
/// - Example: d=128, QT8bit = 128 bytes vs 512 bytes (float32)
///
/// # Example
/// ```ignore
/// use faiss_next::{IndexIVFScalarQuantizer, IndexFlatL2, QuantizerType, MetricType};
///
/// let quantizer = IndexFlatL2::new(128)?;
/// let mut index = IndexIVFScalarQuantizer::new_with(
///     &quantizer,
///     128,
///     100,  // nlist
///     QuantizerType::QT8bit
/// )?;
///
/// index.train(10000, &training_data)?;
/// index.add(100000, &vectors)?;
/// index.set_nprobe(10);
/// let (distances, labels) = index.search(1, &query, 10)?;
/// ```
pub struct IndexIVFScalarQuantizer {
    inner: *mut ffi::FaissIndexIVFScalarQuantizer,
}

impl IndexIVFScalarQuantizer {
    /// Create a new IndexIVFScalarQuantizer with all options
    ///
    /// # Arguments
    /// * `quantizer` - The quantizer index
    /// * `d` - Vector dimension
    /// * `nlist` - Number of inverted lists
    /// * `qt` - Quantizer type
    /// * `metric` - Distance metric
    /// * `encode_residual` - If true, quantize residuals instead of vectors
    pub fn new_with_metric(
        quantizer: &impl IndexTrait,
        d: usize,
        nlist: usize,
        qt: QuantizerType,
        metric: MetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexIVFScalarQuantizer_new_with_metric(
                &mut index_ptr,
                quantizer.inner_ptr(),
                d,
                nlist,
                qt.to_code(),
                metric.to_code(),
                encode_residual as i32,
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexIVFScalarQuantizer
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let ptr = ffi::faiss_IndexIVFScalarQuantizer_cast(index.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(Self { inner: ptr })
            }
        }
    }

    /// Get the number of inverted lists
    pub fn nlist(&self) -> usize {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_nlist(self.inner) }
    }

    /// Get the number of lists to probe during search
    pub fn nprobe(&self) -> usize {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_nprobe(self.inner) }
    }

    /// Set the number of lists to probe during search
    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe {
            ffi::faiss_IndexIVFScalarQuantizer_set_nprobe(self.inner, nprobe);
        }
    }

    /// Get the quantizer index
    pub fn quantizer(&self) -> *mut ffi::FaissIndex {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_quantizer(self.inner) }
    }

    /// Check if the index owns its internal fields
    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_own_fields(self.inner) != 0 }
    }

    /// Set whether the index owns its internal fields
    pub fn set_own_fields(&mut self, own: bool) {
        unsafe {
            ffi::faiss_IndexIVFScalarQuantizer_set_own_fields(self.inner, own as i32);
        }
    }

    /// Add vectors with pre-computed assignments
    ///
    /// # Arguments
    /// * `n` - Number of vectors
    /// * `x` - Vector data
    /// * `xids` - Vector IDs (optional)
    /// * `precomputed_idx` - Pre-computed cluster assignments (optional)
    pub fn add_core(
        &mut self,
        n: i64,
        x: &[f32],
        xids: Option<&[i64]>,
        precomputed_idx: Option<&[i64]>,
    ) -> Result<()> {
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
            let xids_ptr = xids.map_or(std::ptr::null(), |ids| ids.as_ptr());
            let idx_ptr = precomputed_idx.map_or(std::ptr::null(), |idx| idx.as_ptr());

            let ret = ffi::faiss_IndexIVFScalarQuantizer_add_core(
                self.inner,
                n,
                x.as_ptr(),
                xids_ptr,
                idx_ptr,
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }

        Ok(())
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexIVFScalarQuantizer {
        self.inner
    }
}

impl Drop for IndexIVFScalarQuantizer {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexIVFScalarQuantizer_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexIVFScalarQuantizer {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_flat::IndexFlatL2;

    #[test]
    fn test_quantizer_type() {
        assert_eq!(QuantizerType::QT8bit.to_code(), ffi::FaissQuantizerType_QT_8bit);
        assert_eq!(
            QuantizerType::from_code(ffi::FaissQuantizerType_QT_8bit),
            Some(QuantizerType::QT8bit)
        );
    }

    #[test]
    fn test_index_scalar_quantizer_creation() {
        let index = IndexScalarQuantizer::new_with(128, QuantizerType::QT8bit, MetricType::L2);
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
    }

    #[test]
    fn test_index_ivf_scalar_quantizer_creation() {
        let quantizer = IndexFlatL2::new(128).unwrap();
        let index = IndexIVFScalarQuantizer::new_with_metric(
            &quantizer,
            128,
            10,
            QuantizerType::QT8bit,
            MetricType::L2,
            false,
        );
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
        assert_eq!(index.nlist(), 10);
    }

    #[test]
    fn test_index_ivf_scalar_quantizer_with_metric() {
        let quantizer = IndexFlatL2::new(64).unwrap();
        let index = IndexIVFScalarQuantizer::new_with_metric(
            &quantizer,
            64,
            5,
            QuantizerType::QT4bit,
            MetricType::L2,
            false,
        );
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.nlist(), 5);
    }

    #[test]
    fn test_index_ivf_scalar_quantizer_nprobe() {
        let quantizer = IndexFlatL2::new(32).unwrap();
        let mut index = IndexIVFScalarQuantizer::new_with_metric(
            &quantizer,
            32,
            10,
            QuantizerType::QT8bit,
            MetricType::L2,
            false,
        ).unwrap();

        let initial_nprobe = index.nprobe();
        assert!(initial_nprobe > 0);

        index.set_nprobe(5);
        assert_eq!(index.nprobe(), 5);
    }
}
