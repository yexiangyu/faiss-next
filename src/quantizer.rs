use std::ptr::null_mut;

use crate::error::Result;
use crate::index::{Index, IndexPtr};
use faiss_next_sys as sys;
use sys::FaissMetricType;
pub use sys::FaissQuantizerType;

use crate::macros::{define_index_impl, faiss_rc};

define_index_impl!(FaissIndexScalarQuantizer, faiss_IndexScalarQuantizer_free);

impl FaissIndexScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexScalarQuantizer_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, typ: FaissQuantizerType, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexScalarQuantizer_new_with(&mut inner, d, typ, metric) })?;
        Ok(Self { inner })
    }

    pub fn downcast(index: impl IndexPtr) -> Result<Self> {
        let index = index.into_ptr();
        let index = unsafe { sys::faiss_IndexScalarQuantizer_cast(index) };
        Ok(Self { inner: index })
    }
}

define_index_impl!(
    FaissIndexIVFScalarQuantizer,
    faiss_IndexIVFScalarQuantizer_free
);

impl FaissIndexIVFScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexIVFScalarQuantizer_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with_metric(
        quantizer: FaissIndexScalarQuantizer,
        d: i64,
        nlist: usize,
        typ: FaissQuantizerType,
        metric: FaissMetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexScalarQuantizer_new_with(&mut inner, d, typ, metric) })?;
        Ok(Self { inner })
    }
}
