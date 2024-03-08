use std::{marker::PhantomData, ptr::null_mut};

use faiss_next_sys as sys;

use crate::{error::Result, rc};

use super::common::{impl_index_drop, impl_index_trait, FaissIndexTrait};

pub struct FaissIndexReplicas {
    inner: *mut sys::FaissIndexReplicas,
}

impl_index_drop!(FaissIndexReplicas, faiss_IndexReplicas_free);
impl_index_trait!(FaissIndexReplicas);

pub struct FaissIndexReplicaIndex<'a> {
    inner: *mut sys::FaissIndex,
    marker: PhantomData<&'a FaissIndexReplicas>,
}

impl FaissIndexTrait for FaissIndexReplicaIndex<'_> {
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_inner(self) -> *mut sys::FaissIndex {
        unimplemented!()
    }
}

impl FaissIndexReplicas {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexReplicas_new(&mut inner, d) })?;
        Ok(Self { inner })
    }

    pub fn new_with_options(d: i64, threaded: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexReplicas_new_with_options(&mut inner, d, threaded as i32) })?;
        Ok(Self { inner })
    }

    pub fn add_replica(&mut self, index: impl FaissIndexTrait) -> Result<()> {
        let i = index.into_inner();
        rc!({ sys::faiss_IndexReplicas_add_replica(self.inner, i) })?;
        // unsafe { sys::faiss_IndexReplicas_set_own_fields(self.inner(), true as i32) }
        Ok(())
    }

    pub fn remove_replica(&mut self, index: FaissIndexReplicaIndex) -> Result<()> {
        rc!({ sys::faiss_IndexReplicas_remove_replica(self.inner, index.inner()) })
    }

    pub fn at(&mut self, i: i32) -> Result<FaissIndexReplicaIndex<'_>> {
        let inner = unsafe { sys::faiss_IndexReplicas_at(self.inner, i) };
        Ok(FaissIndexReplicaIndex {
            inner,
            marker: PhantomData,
        })
    }

    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexReplicas_own_fields(self.inner) != 0 }
    }
}
