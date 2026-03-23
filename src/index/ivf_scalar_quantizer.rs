use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexIVFScalarQuantizer};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::{Index, IvfIndex};
use crate::metric::MetricType;

use super::QuantizerType;

pub struct IndexIVFScalarQuantizer {
    inner: InnerPtr<FaissIndexIVFScalarQuantizer>,
}

impl IndexIVFScalarQuantizer {
    pub fn new(
        quantizer: super::IndexFlat,
        nlist: usize,
        qtype: QuantizerType,
        metric: MetricType,
    ) -> Result<Self> {
        Self::with_options(quantizer, nlist, qtype, metric, false)
    }

    pub fn with_options(
        quantizer: super::IndexFlat,
        nlist: usize,
        qtype: QuantizerType,
        metric: MetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        let d = quantizer.d();
        let quantizer_ptr = quantizer.inner_ptr();

        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(
                faiss_next_sys::faiss_IndexIVFScalarQuantizer_new_with_metric(
                    &mut inner,
                    quantizer_ptr as *mut FaissIndex,
                    d as usize,
                    nlist,
                    qtype.as_native(),
                    metric.as_native(),
                    encode_residual as i32,
                ),
            )?;

            std::mem::forget(quantizer);

            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let ivf_sq_ptr = faiss_next_sys::faiss_IndexIVFScalarQuantizer_cast(index.inner_ptr());
            if ivf_sq_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexIVFScalarQuantizer",
                    "index is not an IVF ScalarQuantizer index",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(ivf_sq_ptr)?,
            })
        }
    }

    pub fn quantizer(&self) -> *mut FaissIndex {
        unsafe { faiss_next_sys::faiss_IndexIVFScalarQuantizer_quantizer(self.inner.as_ptr()) }
    }
}

impl Index for IndexIVFScalarQuantizer {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl IvfIndex for IndexIVFScalarQuantizer {
    fn nlist(&self) -> usize {
        unsafe { faiss_next_sys::faiss_IndexIVFScalarQuantizer_nlist(self.inner.as_ptr()) }
    }

    fn nprobe(&self) -> usize {
        unsafe { faiss_next_sys::faiss_IndexIVFScalarQuantizer_nprobe(self.inner.as_ptr()) }
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe {
            faiss_next_sys::faiss_IndexIVFScalarQuantizer_set_nprobe(self.inner.as_ptr(), nprobe)
        }
    }
}

impl Drop for IndexIVFScalarQuantizer {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexIVFScalarQuantizer");
        unsafe {
            faiss_next_sys::faiss_IndexIVFScalarQuantizer_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexIVFScalarQuantizer {}
unsafe impl Sync for IndexIVFScalarQuantizer {}
