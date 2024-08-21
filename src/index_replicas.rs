use std::mem::forget;

use crate::{error::*, index::FaissIndexBorrowed, macros::*, traits::FaissIndexTrait};
use faiss_next_sys as ffi;

#[derive(Debug)]
pub struct FaissIndexReplicas {
    inner: *mut ffi::FaissIndexReplicas,
}
impl_faiss_drop!(FaissIndexReplicas, faiss_IndexReplicas_free);

impl FaissIndexTrait for FaissIndexReplicas {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut _
    }
}

impl FaissIndexReplicas {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexReplicas_new(&mut inner, d) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_iwth_options(d: i64, threaded: bool) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexReplicas_new_with_options(&mut inner, d, threaded as i32)
        })?;
        Ok(Self { inner })
    }

    pub fn add_replica(&mut self, index: impl FaissIndexTrait) -> Result<()> {
        let inner = index.inner() as *mut _;
        forget(index);
        faiss_rc(unsafe { ffi::faiss_IndexReplicas_add_replica(self.inner, inner) })
    }

    pub fn remove_replica(&mut self, index: &impl FaissIndexTrait) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_IndexReplicas_remove_replica(self.inner, index.inner()) })
    }

    pub fn at(&self, index: usize) -> Option<FaissIndexBorrowed<'_, Self>> {
        let inner = unsafe { ffi::faiss_IndexReplicas_at(self.inner, index as i32) };
        match inner.is_null() {
            true => None,
            false => Some(FaissIndexBorrowed {
                inner,
                owner: std::marker::PhantomData,
            }),
        }
    }

    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexReplicas_own_fields(self.inner) > 0 }
    }

    fn set_own_fields(&mut self, value: bool) {
        unsafe { ffi::faiss_IndexReplicas_set_own_fields(self.inner, value as i32) }
    }
}
