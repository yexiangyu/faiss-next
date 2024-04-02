use faiss_next_sys as sys;
use std::ptr::null_mut;

use crate::{
    error::Result,
    index::{impl_index, IndexTrait},
    macros::rc,
    metric::MetricType,
};
pub use sys::FaissQuantizerType as QuantizerType;

pub trait IndexScalarQuantizerTrait: IndexTrait {}

pub struct IndexScalarQuantizer {
    inner: *mut sys::FaissIndexScalarQuantizer,
}

impl_index!(IndexScalarQuantizer);
impl IndexScalarQuantizerTrait for IndexScalarQuantizer {}

impl IndexScalarQuantizer {
    pub fn new(d: i64, typ: QuantizerType, metric: MetricType) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexScalarQuantizer_new_with(&mut inner, d, typ, metric.into()) })?;
        Ok(Self { inner })
    }
    pub fn cast(index: impl IndexTrait) -> Self {
        let inner = unsafe { sys::faiss_IndexScalarQuantizer_cast(index.ptr()) };
        Self { inner }
    }
}

pub struct IndexIVFScalarQuantizer {
    inner: *mut sys::FaissIndexIVFScalarQuantizer,
}

impl_index!(IndexIVFScalarQuantizer);

impl IndexIVFScalarQuantizer {
    pub fn new(
        quantizer: impl IndexTrait,
        d: usize,
        nlist: usize,
        qt: QuantizerType,
        metric: MetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexIVFScalarQuantizer_new_with_metric(
                &mut inner,
                quantizer.ptr(),
                d,
                nlist,
                qt,
                metric.into(),
                encode_residual as i32,
            )
        })?;
        let mut r = Self { inner };
        r.set_own_fields(true);
        Ok(r)
    }

    fn set_own_fields(&mut self, own: bool) {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_set_own_fields(self.inner, own as i32) }
    }

    pub fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_nlist(self.inner) }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_nprobe(self.inner) }
    }

    pub fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        rc!({
            sys::faiss_IndexIVFScalarQuantizer_add_core(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                xids.as_ref().as_ptr(),
                precomputed_idx.as_ref().as_ptr(),
            )
        })
    }

    // pub fn train_residual(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
    //     let n = x.as_ref().len() as i64 / self.d() as i64;
    //     rc!({
    //         sys::faiss_IndexIVFScalarQuantizer_train_residual(self.ptr(), n, x.as_ref().as_ptr())
    //     })
    // }
}
