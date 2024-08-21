use std::mem::forget;

use crate::{error::*, macros::*, traits::FaissIndexTrait};
use faiss_next_sys as ffi;

pub struct FaissIndexLSH {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexLSH, faiss_IndexLSH_free);
impl FaissIndexTrait for FaissIndexLSH {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}
impl FaissIndexLSH {
    pub fn new(d: i64, nbits: i32) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexLSH_new(&mut inner, d, nbits) })?;
        Ok(Self { inner })
    }

    pub fn new_with_options(
        d: i64,
        nbits: i32,
        rotate_data: i32,
        train_thresholds: i32,
    ) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexLSH_new_with_options(
                &mut inner,
                d,
                nbits,
                rotate_data,
                train_thresholds,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn downcast(rhs: impl FaissIndexTrait) -> Self {
        let inner = rhs.inner();
        let inner = unsafe { ffi::faiss_IndexLSH_cast(inner) };
        forget(rhs);
        Self { inner }
    }
    pub fn nbits(&self) -> i32 {
        unsafe { ffi::faiss_IndexLSH_nbits(self.inner) }
    }
    pub fn code_size(&self) -> i32 {
        unsafe { ffi::faiss_IndexLSH_code_size(self.inner) }
    }
    pub fn rotate_data(&self) -> i32 {
        unsafe { ffi::faiss_IndexLSH_rotate_data(self.inner) }
    }
    pub fn train_thresholds(&self) -> i32 {
        unsafe { ffi::faiss_IndexLSH_train_thresholds(self.inner) }
    }
}
