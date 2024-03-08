use crate::{
    error::{Error, Result},
    rc,
    vector_transform::FaissVectorTransformTrait,
};
use faiss_next_sys as sys;
use std::{marker::PhantomData, ptr::addr_of_mut};

use super::common::{impl_index_drop, impl_index_trait, FaissIndexTrait};

pub struct FaissIndexPreTransformImpl {
    inner: *mut sys::FaissIndexPreTransform,
}

impl_index_trait!(FaissIndexPreTransformImpl);
impl_index_drop!(FaissIndexPreTransformImpl, faiss_IndexPreTransform_free);

impl FaissIndexPreTransformImpl {
    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let i = unsafe { sys::faiss_IndexPreTransform_cast(i) };
        match i.is_null() {
            true => Err(Error::DowncastFailure),
            false => Ok(Self { inner: i }),
        }
    }

    pub fn index(&mut self) -> FaissIndexPreTransformTransform<'_> {
        let inner = unsafe { sys::faiss_IndexPreTransform_index(self.inner) };
        FaissIndexPreTransformTransform {
            inner,
            marker: PhantomData,
        }
    }

    pub fn new() -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_IndexPreTransform_new(addr_of_mut!(inner)) })?;
        Ok(Self { inner })
    }

    pub fn new_with(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_IndexPreTransform_new_with(addr_of_mut!(inner), i) })?;
        unsafe { sys::faiss_IndexPreTransform_set_own_fields(inner, true as i32) }
        Ok(Self { inner })
    }

    pub fn new_with_transform(
        trans: impl FaissVectorTransformTrait,
        index: impl FaissIndexTrait,
    ) -> Result<Self> {
        let i = index.into_inner();
        let t = trans.into_inner();
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_IndexPreTransform_new_with_transform(addr_of_mut!(inner), t, i) })?;
        unsafe { sys::faiss_IndexPreTransform_set_own_fields(inner, true as i32) }
        Ok(Self { inner })
    }

    pub fn prepend_transform(&mut self, trans: impl FaissVectorTransformTrait) -> Result<()> {
        let t = trans.inner();
        rc!({ sys::faiss_IndexPreTransform_prepend_transform(self.inner, t) })?;
        unsafe { sys::faiss_IndexPreTransform_set_own_fields(self.inner(), true as i32) }
        Ok(())
    }
}

pub struct FaissIndexPreTransformTransform<'a> {
    inner: *mut sys::FaissIndex,
    marker: std::marker::PhantomData<&'a FaissIndexPreTransformImpl>,
}

impl FaissIndexTrait for FaissIndexPreTransformTransform<'_> {
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_inner(self) -> *mut sys::FaissIndex {
        unimplemented!()
    }
}
