use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexScalarQuantizer, FaissQuantizerType};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;
use crate::metric::MetricType;

pub struct IndexScalarQuantizer {
    inner: InnerPtr<FaissIndexScalarQuantizer>,
}

impl IndexScalarQuantizer {
    pub fn new(d: u32, qtype: QuantizerType, metric: MetricType) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexScalarQuantizer_new_with(
                &mut inner,
                d as i64,
                qtype.as_native(),
                metric.as_native(),
            ))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let sq_ptr = faiss_next_sys::faiss_IndexScalarQuantizer_cast(index.inner_ptr());
            if sq_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexScalarQuantizer",
                    "index is not a ScalarQuantizer index",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(sq_ptr)?,
            })
        }
    }
}

impl Index for IndexScalarQuantizer {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexScalarQuantizer {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexScalarQuantizer");
        unsafe {
            faiss_next_sys::faiss_IndexScalarQuantizer_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexScalarQuantizer {}
unsafe impl Sync for IndexScalarQuantizer {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerType {
    Qt8bit,
    Qt4bit,
    Qt8bitUniform,
    Qt4bitUniform,
    QtFp16,
    Qt8bitDirect,
    Qt6bit,
}

impl QuantizerType {
    pub fn as_native(&self) -> FaissQuantizerType {
        match self {
            QuantizerType::Qt8bit => FaissQuantizerType::QT_8bit,
            QuantizerType::Qt4bit => FaissQuantizerType::QT_4bit,
            QuantizerType::Qt8bitUniform => FaissQuantizerType::QT_8bit_uniform,
            QuantizerType::Qt4bitUniform => FaissQuantizerType::QT_4bit_uniform,
            QuantizerType::QtFp16 => FaissQuantizerType::QT_fp16,
            QuantizerType::Qt8bitDirect => FaissQuantizerType::QT_8bit_direct,
            QuantizerType::Qt6bit => FaissQuantizerType::QT_6bit,
        }
    }
}
