use super::{Index, IndexPtr};
use crate::error::{Error, Result};
use crate::macros::define_index_impl;
use faiss_next_sys as sys;

define_index_impl!(FaissIndexLSH, faiss_IndexLSH_free);

impl FaissIndexLSH {
    pub fn downcast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexLSH_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }

    pub fn nbits(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_nbits(self.ptr()) }
    }

    pub fn code_size(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_code_size(self.ptr()) }
    }

    pub fn rotate_data(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_rotate_data(self.ptr()) }
    }

    pub fn train_thresholds(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_train_thresholds(self.ptr()) }
    }

    pub fn new_with_options(
        d: i64,
        nbits: i32,
        rotate_data: i32,
        train_thresholds: i32,
    ) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::faiss_IndexLSH_new_with_options(
                &mut inner,
                d,
                nbits,
                rotate_data,
                train_thresholds,
            );
        }
        Ok(Self { inner })
    }
}
