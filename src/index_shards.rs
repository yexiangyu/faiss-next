use std::ptr::null_mut;
use std::{marker::PhantomData, mem::forget};

use crate::{error::*, index::FaissIndexBorrowed, macros::*, traits::FaissIndexTrait};
use faiss_next_sys as ffi;

pub struct FaissIndexShards {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexShards, faiss_IndexShards_free);

impl FaissIndexShards {
    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexShards_own_fields(self.inner) > 0 }
    }

    fn set_own_fields(&mut self, value: bool) {
        unsafe { ffi::faiss_IndexShards_set_own_fields(self.inner, value as i32) }
    }

    pub fn successive_ids(&self) -> i32 {
        unsafe { ffi::faiss_IndexShards_successive_ids(self.inner) }
    }

    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexShards_new(&mut inner, d) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with_options(d: i64, threaded: bool, successive_ids: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_IndexShards_new_with_options(&mut inner, d, threaded as i32, successive_ids)
        })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn add_shard(&mut self, index: impl FaissIndexTrait) -> Result<()> {
        let inner = index.inner();
        forget(index);
        faiss_rc(unsafe { ffi::faiss_IndexShards_add_shard(self.inner, inner) })
    }

    pub fn remove_shard(&mut self, index: &impl FaissIndexTrait) -> Result<()> {
        let inner = index.inner();
        faiss_rc(unsafe { ffi::faiss_IndexShards_remove_shard(self.inner, inner) })
    }
    pub fn at(&self, index: usize) -> Option<FaissIndexBorrowed<'_, Self>> {
        let inner = unsafe { ffi::faiss_IndexShards_at(self.inner, index as i32) };
        match inner.is_null() {
            true => None,
            false => Some(FaissIndexBorrowed {
                inner,
                owner: PhantomData,
            }),
        }
        // Ok(FaissIndexBorrowed { inner })
    }
}
