use std::{mem::forget, ptr::null_mut};

use faiss_next_sys as sys;

use crate::{
    error::Result,
    index::{impl_index, IndexTrait},
    macros::rc,
};

pub struct IndexReplicas {
    inner: *mut sys::FaissIndexReplicas,
    num_replicas: usize,
}

impl_index!(IndexReplicas);

impl IndexReplicas {
    pub fn new(d: i64, threaded: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexReplicas_new_with_options(&mut inner, d, threaded as i32) })?;
        Ok(Self {
            inner,
            num_replicas: 0,
        })
    }

    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    pub fn add_replica(&mut self, index: impl IndexTrait) -> Result<()> {
        rc!({ sys::faiss_IndexReplicas_add_replica(self.inner, index.ptr()) })?;
        forget(index);
        self.set_own_fields(true);
        self.num_replicas += 1;
        Ok(())
    }

    pub fn remove_replica(&mut self, i: usize) -> Result<()> {
        let r = unsafe { sys::faiss_IndexReplicas_at(self.ptr(), i as i32) };
        rc!({ sys::faiss_IndexReplicas_remove_replica(self.inner, r) })?;
        self.num_replicas -= 1;
        Ok(())
    }

    fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexReplicas_set_own_fields(self.inner, own_fields as i32) }
    }
}
