use std::ptr::{addr_of_mut, null_mut};

use super::common::{impl_index_drop, impl_index_trait, FaissIndexTrait};
use crate::error::Result;
use crate::rc;
use faiss_next_sys as sys;
use tracing::trace;

pub struct FaissIndexLSH {
    inner: *mut sys::FaissIndexLSH,
}

impl_index_drop!(FaissIndexLSH, faiss_IndexLSH_free);
impl_index_trait!(FaissIndexLSH);

impl FaissIndexLSH {
    pub fn nbits(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_nbits(self.inner) }
    }

    pub fn code_size(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_code_size(self.inner) }
    }

    pub fn rotate_data(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_rotate_data(self.inner) }
    }

    pub fn train_thresholds(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_train_thresholds(self.inner) }
    }

    pub fn new(d: i64, nbits: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexLSH_new(addr_of_mut!(inner), d, nbits) })?;
        trace!("create faiss lsh index inner={:?}", inner);
        Ok(Self { inner })
    }

    pub fn new_with_options(
        d: i64,
        nbits: i32,
        rotate_data: i32,
        train_thresholds: i32,
    ) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexLSH_new_with_options(
                addr_of_mut!(inner),
                d,
                nbits,
                rotate_data,
                train_thresholds,
            )
        })?;
        Ok(Self { inner })
    }
}
