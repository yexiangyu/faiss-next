use crate::error::Result;
use faiss_next_sys as ffi;

use std::{mem::forget, ptr::null_mut};

use crate::index::{IndexBorrowed, IndexTrait};

pub trait IndexReplicasTrait: IndexTrait {
    fn add_replica(&mut self, index: impl IndexTrait) -> Result<()> {
        let inner = index.inner();
        forget(index);
        ffi::ok!(faiss_IndexReplicas_add_replica, self.inner(), inner)?;
        Ok(())
    }

    fn remove_replica(&mut self, index: &impl IndexTrait) -> Result<()> {
        ffi::ok!(
            faiss_IndexReplicas_remove_replica,
            self.inner(),
            index.inner()
        )?;
        Ok(())
    }

    fn at(&self, index: usize) -> Option<IndexBorrowed<'_>> {
        let inner = unsafe { ffi::faiss_IndexReplicas_at(self.inner(), index as i32) };
        match inner.is_null() {
            true => None,
            false => Some(IndexBorrowed::new(inner)),
        }
    }
    
    fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexReplicas_own_fields(self.inner()) != 0 }
    }
}

macro_rules! impl_index_replicas {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexReplicasTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexReplicas {
    inner: *mut ffi::FaissIndex,
}

impl_index_replicas!(IndexReplicas);
ffi::impl_drop!(IndexReplicas, faiss_IndexReplicas_free);

impl IndexReplicas {
    pub fn new(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexReplicas_new, &mut inner, d as i64)?;
        ffi::run!(faiss_IndexReplicas_set_own_fields, inner, true as i32);
        Ok(Self { inner })
    }

    pub fn new_with_options(d: i32, threaded: bool) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexReplicas_new_with_options,
            &mut inner,
            d as i64,
            threaded as i32
        )?;
        ffi::run!(faiss_IndexReplicas_set_own_fields, inner, true as i32);
        Ok(Self { inner })
    }


}
