use faiss_next_sys::{self as ffi, FaissMetricType};

use crate::{error::*, index::FaissIndexTrait, macros::*};
pub use ffi::FaissQuantizerType;
use std::{mem::forget, ptr::null_mut};

#[derive(Debug)]
pub struct FaissIndexScalarQuantizer {
    inner: *mut ffi::FaissIndexScalarQuantizer,
}
impl FaissIndexTrait for FaissIndexScalarQuantizer {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut _
    }
}

impl_faiss_drop!(FaissIndexScalarQuantizer, faiss_IndexScalarQuantizer_free);
impl FaissIndexScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexScalarQuantizer_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, qt: FaissQuantizerType, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexScalarQuantizer_new_with(&mut inner, d, qt, metric) })?;
        Ok(Self { inner })
    }

    pub fn downcast(rhs: impl FaissIndexTrait) -> Self {
        let inner = rhs.inner() as *mut _;
        let inner = unsafe { ffi::faiss_IndexScalarQuantizer_cast(inner) };
        forget(rhs);
        Self { inner }
    }
}

#[derive(Debug)]
pub struct FaissIndexIVFScalarQuantizer {
    pub inner: *mut ffi::FaissIndexIVFScalarQuantizer,
}
impl FaissIndexTrait for FaissIndexIVFScalarQuantizer {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut _
    }
}
impl_faiss_drop!(
    FaissIndexIVFScalarQuantizer,
    faiss_IndexIVFScalarQuantizer_free
);
impl FaissIndexIVFScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexIVFScalarQuantizer_new(&mut inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with(
        d: i64,
        quantizer: impl FaissIndexTrait,
        nlist: usize,
        qt: FaissQuantizerType,
    ) -> Result<Self> {
        let quantizer_inner = quantizer.inner() as *mut _;
        forget(quantizer);
        let mut inner = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexIVFScalarQuantizer_new_with(&mut inner, quantizer_inner, d, nlist, qt)
        })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with_metric(
        d: usize,
        quantizer: impl FaissIndexTrait,
        nlist: usize,
        qt: FaissQuantizerType,
        metric: FaissMetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        let quantizer_inner = quantizer.inner() as *mut _;
        forget(quantizer);
        let mut inner = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexIVFScalarQuantizer_new_with_metric(
                &mut inner,
                quantizer_inner,
                d,
                nlist,
                qt,
                metric,
                encode_residual as i32,
            )
        })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn downcast(rhs: impl FaissIndexTrait) -> Self {
        let inner = rhs.inner() as *mut _;
        let inner = unsafe { ffi::faiss_IndexIVFScalarQuantizer_cast(inner) };
        forget(rhs);
        Self { inner }
    }

    fn set_own_fields(&mut self, own: bool) {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_set_own_fields(self.inner, own as i32) }
    }

    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_own_fields(self.inner) > 0 }
    }
    pub fn quantizer(&self) -> Option<FaissIndexScalarQuantizer> {
        let inner = unsafe { ffi::faiss_IndexIVFScalarQuantizer_quantizer(self.inner) };
        match inner.is_null() {
            true => None,
            false => Some(FaissIndexScalarQuantizer { inner }),
        }
    }
    pub fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(xids.as_ref().len(), n);
        assert_eq!(precomputed_idx.as_ref().len(), n);
        faiss_rc(unsafe {
            ffi::faiss_IndexIVFScalarQuantizer_add_core(
                self.inner,
                n as i64,
                x.as_ref().as_ptr(),
                xids.as_ref().as_ptr(),
                precomputed_idx.as_ref().as_ptr(),
            )
        })
    }
}
impl_faiss_getter!(
    FaissIndexIVFScalarQuantizer,
    nlist,
    faiss_IndexIVFScalarQuantizer_nlist,
    usize
);
impl_faiss_getter!(
    FaissIndexIVFScalarQuantizer,
    nprobe,
    faiss_IndexIVFScalarQuantizer_nprobe,
    usize
);
impl_faiss_setter!(
    FaissIndexIVFScalarQuantizer,
    set_nprobe,
    faiss_IndexIVFScalarQuantizer_set_nprobe,
    probe,
    usize
);
