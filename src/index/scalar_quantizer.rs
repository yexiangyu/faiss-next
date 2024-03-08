use std::ptr::null_mut;

use faiss_next_sys as sys;

pub use sys::FaissQuantizerType;

use crate::{
    error::{Error, Result},
    rc,
};

use super::{
    common::{impl_index_drop, impl_index_trait, FaissIndexTrait},
    metric::FaissMetricType,
};

pub trait FaissIndexScalarQuantizerTrait: FaissIndexTrait {}

pub struct FaissIndexScalarQuantizerImpl {
    inner: *mut sys::FaissIndexScalarQuantizer,
}

impl_index_drop!(
    FaissIndexScalarQuantizerImpl,
    faiss_IndexScalarQuantizer_free
);
impl_index_trait!(FaissIndexScalarQuantizerImpl);

impl FaissIndexScalarQuantizerImpl {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexScalarQuantizer_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, typ: FaissQuantizerType, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexScalarQuantizer_new_with(&mut inner, d, typ, metric) })?;
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let inner = unsafe { sys::faiss_IndexScalarQuantizer_cast(index.into_inner()) };
        match inner.is_null() {
            true => Err(Error::DowncastFailure),
            false => Ok(Self { inner }),
        }
    }
}

pub struct FaissIndexIVFScalarQuantizer {
    inner: *mut sys::FaissIndexIVFScalarQuantizer,
}

impl_index_drop!(
    FaissIndexIVFScalarQuantizer,
    faiss_IndexIVFScalarQuantizer_free
);
impl_index_trait!(FaissIndexIVFScalarQuantizer);
impl FaissIndexScalarQuantizerTrait for FaissIndexIVFScalarQuantizer {}

impl FaissIndexIVFScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexIVFScalarQuantizer_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(
        d: i64,
        quantizer: impl FaissIndexTrait,
        nlist: usize,
        qt: FaissQuantizerType,
    ) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexIVFScalarQuantizer_new_with(
                &mut inner,
                quantizer.into_inner(),
                d,
                nlist,
                qt,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn new_with_metric(
        quantizer: impl FaissIndexTrait,
        d: usize,
        nlist: usize,
        qt: FaissQuantizerType,
        metric: FaissMetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexIVFScalarQuantizer_new_with_metric(
                &mut inner,
                quantizer.into_inner(),
                d,
                nlist,
                qt,
                metric,
                encode_residual as i32,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let inner = unsafe { sys::faiss_IndexIVFScalarQuantizer_cast(index.into_inner()) };
        match inner.is_null() {
            true => Err(Error::DowncastFailure),
            false => Ok(Self { inner }),
        }
    }

    pub fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_nlist(self.inner()) }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_nprobe(self.inner()) }
    }

    pub fn quantizer(&self) -> *mut sys::FaissIndex {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_quantizer(self.inner()) }
    }

    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_own_fields(self.inner()) != 0 }
    }
    pub fn set_own_fields(&mut self, value: bool) {
        unsafe { sys::faiss_IndexIVFScalarQuantizer_set_own_fields(self.inner(), value as i32) }
    }

    pub fn add_core(&mut self, x: &[f32], ids: &[i64], precomputed_idx: &[i64]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        rc!({
            sys::faiss_IndexIVFScalarQuantizer_add_core(
                self.inner(),
                n,
                x.as_ptr(),
                ids.as_ptr(),
                precomputed_idx.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn train_residual(&mut self, x: &[f32]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        rc!({ sys::faiss_IndexIVFScalarQuantizer_train_residual(self.inner(), n, x.as_ptr()) })?;
        Ok(())
    }
}
