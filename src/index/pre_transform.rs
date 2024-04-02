use faiss_next_sys as sys;
use std::ptr::null_mut;

use crate::{
    error::Result,
    index::{impl_index, IndexTrait},
    macros::rc,
    vector_transform::VectorTransformTrait,
};

pub struct IndexPreTransform {
    inner: *mut sys::FaissIndexPreTransform,
}
impl_index!(IndexPreTransform);

impl IndexPreTransform {
    pub fn new(ltrans: impl VectorTransformTrait, index: impl IndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexPreTransform_new_with_transform(&mut inner, ltrans.ptr(), index.ptr())
        })?;

        let mut r = Self { inner };
        r.set_own_fields(true);
        Ok(r)
    }

    fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexPreTransform_set_own_fields(self.inner, own_fields as i32) }
    }
}
