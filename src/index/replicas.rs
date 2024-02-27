use std::marker::PhantomData;
use std::ptr::null_mut;

use super::{Index, IndexPtr};
use crate::error::Result;
use crate::macros::faiss_rc;
use faiss_next_sys as sys;

pub struct FaissIndexReplicas {
    inner: *mut sys::FaissIndexReplicas,
    n: usize,
}

impl Drop for FaissIndexReplicas {
    fn drop(&mut self) {
        unsafe {
            sys::faiss_Index_free(self.inner);
        }
    }
}

impl IndexPtr for FaissIndexReplicas {
    fn ptr(&self) -> *const sys::FaissIndex {
        self.inner
    }

    fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_ptr(self) -> *mut sys::FaissIndex {
        let inner = self.inner;
        std::mem::forget(self);
        inner
    }
}

impl Index for FaissIndexReplicas {}

pub struct FaissIndexReplica<'a> {
    inner: *mut sys::FaissIndex,
    marker: PhantomData<&'a FaissIndexReplicas>,
}

impl IndexPtr for FaissIndexReplica<'_> {
    fn ptr(&self) -> *const sys::FaissIndex {
        self.inner
    }

    fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
        todo!()
    }

    fn into_ptr(self) -> *mut sys::FaissIndex {
        todo!()
    }
}

impl Index for FaissIndexReplica<'_> {}

impl FaissIndexReplicas {
    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexReplicas_own_fields(self.inner) != 0 }
    }

    pub fn set_own_fields(&mut self, value: bool) {
        unsafe {
            sys::faiss_IndexReplicas_set_own_fields(self.inner, value as i32);
        }
    }

    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexReplicas_new(&mut inner, d) })?;
        Ok(Self { inner, n: 0 })
    }

    pub fn new_with_options(d: i64, threaded: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexReplicas_new_with_options(&mut inner, d, threaded as i32) })?;
        Ok(Self { inner, n: 0 })
    }

    pub fn add_index(&mut self, index: impl IndexPtr) -> Result<()> {
        faiss_rc!({ sys::faiss_IndexReplicas_add_replica(self.inner, index.into_ptr()) })?;
        self.n += 1;
        Ok(())
    }

    pub fn remove_replica(&mut self, index: &mut impl IndexPtr) -> Result<()> {
        faiss_rc!({ sys::faiss_IndexReplicas_remove_replica(self.inner, index.mut_ptr()) })
    }

    pub fn at(&mut self, i: i32) -> FaissIndexReplica {
        unsafe {
            let inner = sys::faiss_IndexReplicas_at(self.inner, i);
            FaissIndexReplica {
                inner,
                marker: PhantomData,
            }
        }
    }
}
