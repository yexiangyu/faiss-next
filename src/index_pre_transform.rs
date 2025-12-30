use crate::error::Result;
use faiss_next_sys as ffi;

use std::{mem::forget, ptr::null_mut};

use crate::index::{IndexBorrowed, IndexTrait};
use crate::vector_transform::VectorTransformTrait;

pub trait IndexPreTransformTrait: IndexTrait {
    fn prepend_transform(&mut self, trans: impl VectorTransformTrait) -> Result<()> {
        ffi::ok!(
            faiss_IndexPreTransform_prepend_transform,
            self.inner(),
            trans.inner()
        )?;
        ffi::run!(
            faiss_IndexPreTransform_set_own_fields,
            self.inner(),
            true as i32
        );
        Ok(())
    }

    fn index(&self) -> IndexBorrowed<'_> {
        let inner = ffi::run!(faiss_IndexPreTransform_index, self.inner());
        IndexBorrowed::new(inner)
    }
}

macro_rules! impl_index_pre_transform {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexPreTransformTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexPreTransform {
    inner: *mut ffi::FaissIndex,
}

impl_index_pre_transform!(IndexPreTransform);
ffi::impl_drop!(IndexPreTransform, faiss_IndexPreTransform_free);

impl IndexPreTransform {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexPreTransform_new, &mut inner)?;
        let ret = Self { inner };
        // Set own_fields to true so that the index owns its components by default
        ffi::run!(faiss_IndexPreTransform_set_own_fields, ret.inner, 1i32);
        Ok(ret)
    }

    pub fn new_with(index: impl IndexTrait) -> Result<Self> {
        let index_inner = index.inner();
        forget(index);
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexPreTransform_new_with, &mut inner, index_inner)?;
        let ret = Self { inner };
        // Set own_fields to true so that the index owns its components by default
        ffi::run!(faiss_IndexPreTransform_set_own_fields, ret.inner, 1i32);
        Ok(ret)
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexPreTransform_cast, index.inner());
        assert!(
            !inner.is_null(),
            "Failed to cast index to IndexPreTransform"
        );
        Ok(Self { inner })
    }
}
