use std::{mem::forget, ptr::null_mut};

use faiss_next_sys::{self as ffi, FaissMetricType};

use crate::{error::*, index::FaissIndexTrait, macros::*, traits::FaissIndexIVFTrait};

#[derive(Debug)]
pub struct FaissIndexIVFFlat {
    pub inner: *mut ffi::FaissIndexIVFFlat,
}

impl FaissIndexTrait for FaissIndexIVFFlat {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut _
    }
}

impl_faiss_drop!(FaissIndexIVFFlat, faiss_IndexIVFFlat_free);

impl FaissIndexIVFTrait for FaissIndexIVFFlat {}

impl FaissIndexIVFFlat {
    pub fn downcast(rhs: impl FaissIndexTrait) -> Self {
        let inner = rhs.inner() as *mut _;
        let inner = unsafe { ffi::faiss_IndexIVFFlat_cast(inner) };
        forget(rhs);
        Self { inner }
    }

    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexIVFFlat_new(&mut inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with(quantizer: impl FaissIndexTrait, d: usize, nlist: usize) -> Result<Self> {
        let quantizer_inner = quantizer.inner();
        forget(quantizer);
        let mut inner = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexIVFFlat_new_with(&mut inner, quantizer_inner, d, nlist)
        })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with_metric(
        quantizer: impl FaissIndexTrait,
        d: usize,
        nlist: usize,
        metric: FaissMetricType,
    ) -> Result<Self> {
        let quantizer_inner = quantizer.inner();
        forget(quantizer);
        let mut inner = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexIVFFlat_new_with_metric(&mut inner, quantizer_inner, d, nlist, metric)
        })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
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
            ffi::faiss_IndexIVFFlat_add_core(
                self.inner,
                n as i64,
                x.as_ref().as_ptr(),
                xids.as_ref().as_ptr(),
                precomputed_idx.as_ref().as_ptr(),
            )
        })
    }

    pub fn update_vectors(
        &mut self,
        mut idx: impl AsMut<[i64]>,
        x: impl AsRef<[f32]>,
    ) -> Result<()> {
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(idx.as_mut().len(), n);
        faiss_rc(unsafe {
            ffi::faiss_IndexIVFFlat_update_vectors(
                self.inner,
                n as i32,
                idx.as_mut().as_mut_ptr(),
                x.as_ref().as_ptr(),
            )
        })
    }

    pub fn nlist(&self) -> usize {
        unsafe { ffi::faiss_IndexIVFFlat_nlist(self.inner) }
    }
}
