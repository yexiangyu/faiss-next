use crate::error::Result;
use faiss_next_sys as ffi;

use std::{mem::forget, ptr::null_mut};

use crate::index::{IndexTrait, MetricType};
use crate::index_ivf::IndexIVFTrait;

pub trait IndexIVFFlatTrait: IndexIVFTrait {
    fn nlist(&self) -> usize {
        ffi::run!(faiss_IndexIVFFlat_nlist, self.inner() as *mut _)
    }

    fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(
            xids.as_ref().len(),
            n,
            "xids length ({}) must match number of vectors ({})",
            xids.as_ref().len(),
            n
        );
        assert_eq!(
            precomputed_idx.as_ref().len(),
            n,
            "precomputed_idx length ({}) must match number of vectors ({})",
            precomputed_idx.as_ref().len(),
            n
        );

        ffi::ok!(
            faiss_IndexIVFFlat_add_core,
            self.inner() as *mut _,
            n as i64,
            x.as_ref().as_ptr(),
            xids.as_ref().as_ptr(),
            precomputed_idx.as_ref().as_ptr()
        )?;
        Ok(())
    }

    fn update_vectors(&mut self, mut idx: impl AsMut<[i64]>, x: impl AsRef<[f32]>) -> Result<()> {
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(
            idx.as_mut().len(),
            n,
            "idx length ({}) must match number of vectors ({})",
            idx.as_mut().len(),
            n
        );

        ffi::ok!(
            faiss_IndexIVFFlat_update_vectors,
            self.inner() as *mut _,
            n as i32,
            idx.as_mut().as_mut_ptr(),
            x.as_ref().as_ptr()
        )?;
        Ok(())
    }
}

macro_rules! impl_index_ivf_flat {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner as *mut _
            }
        }

        impl IndexIVFTrait for $cls {}

        impl IndexIVFFlatTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexIVFFlat {
    inner: *mut ffi::FaissIndexIVFFlat,
}

impl_index_ivf_flat!(IndexIVFFlat);
ffi::impl_drop!(IndexIVFFlat, faiss_IndexIVFFlat_free);

impl IndexIVFFlat {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexIVFFlat_new, &mut inner)?;
        let ret = Self { inner };
        // Set own_fields to true so that the index owns its components
        ffi::run!(faiss_IndexIVF_set_own_fields, ret.inner as *mut _, 1i32);
        Ok(ret)
    }

    pub fn new_with(quantizer: impl IndexTrait, d: i64, nlist: i64) -> Result<Self> {
        let quantizer_inner = quantizer.inner();
        forget(quantizer);
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexIVFFlat_new_with,
            &mut inner,
            quantizer_inner,
            d as usize,
            nlist as usize
        )?;
        let ret = Self { inner };
        // Set own_fields to true so that the index owns its components
        ffi::run!(faiss_IndexIVF_set_own_fields, ret.inner as *mut _, 1i32);
        Ok(ret)
    }

    pub fn new_with_metric(
        quantizer: impl IndexTrait,
        d: i64,
        nlist: i64,
        metric: MetricType,
    ) -> Result<Self> {
        let quantizer_inner = quantizer.inner();
        forget(quantizer);
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexIVFFlat_new_with_metric,
            &mut inner,
            quantizer_inner,
            d as usize,
            nlist as usize,
            metric
        )?;
        let ret = Self { inner };
        // Set own_fields to true so that the index owns its components
        ffi::run!(faiss_IndexIVF_set_own_fields, ret.inner as *mut _, 1i32);
        Ok(ret)
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexIVFFlat_cast, index.inner() as *mut _);
        assert!(!inner.is_null(), "Failed to cast index to IndexIVFFlat");
        Ok(Self { inner })
    }
}
