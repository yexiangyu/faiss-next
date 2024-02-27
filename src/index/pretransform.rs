use std::marker::PhantomData;
use std::ptr::null_mut;

use faiss_next_sys as sys;

use crate::error::{Error, Result};
use crate::macros::faiss_rc;
use crate::transform::VectorTransformPtr;

use super::factory::FaissIndexImpl;
use super::{Index, IndexPtr};

pub struct FaissIndexPreTransform<'a, 'b> {
    inner: *mut sys::FaissIndexPreTransform,
    index: PhantomData<&'a dyn IndexPtr>,
    transform: PhantomData<&'b dyn VectorTransformPtr>,
}

impl<'a, 'b> Drop for FaissIndexPreTransform<'a, 'b> {
    fn drop(&mut self) {
        unsafe { sys::faiss_IndexPreTransform_free(self.inner) }
    }
}

impl<'a, 'b> IndexPtr for FaissIndexPreTransform<'a, 'b> {
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
impl<'a, 'b> Index for FaissIndexPreTransform<'a, 'b> {}

impl<'a, 'b> FaissIndexPreTransform<'a, 'b> {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexPreTransform_new(&mut inner) })?;
        Ok(Self {
            inner,
            index: PhantomData,
            transform: PhantomData,
        })
    }

    pub fn new_with(index: &'a mut impl IndexPtr) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexPreTransform_new_with(&mut inner, index.mut_ptr()) })?;
        Ok(Self {
            inner,
            index: PhantomData,
            transform: PhantomData,
        })
    }

    pub fn new_with_transform(
        index: &'a mut impl IndexPtr,
        transform: &'b mut impl VectorTransformPtr,
    ) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_IndexPreTransform_new_with_transform(
                &mut inner,
                transform.mut_ptr(),
                index.mut_ptr(),
            )
        })?;
        Ok(Self {
            inner,
            index: PhantomData,
            transform: PhantomData,
        })
    }

    pub fn cast(index: FaissIndexImpl) -> Result<Self> {
        let inner = index.into_ptr();
        let inner = unsafe { sys::faiss_IndexPreTransform_cast(inner) };
        match inner.is_null() {
            true => Err(Error::CastFailed),
            false => Ok(Self {
                inner,
                index: PhantomData,
                transform: PhantomData,
            }),
        }
    }

    pub fn index(self) -> FaissIndexImpl {
        let inner = self.into_ptr();
        let inner = unsafe { sys::faiss_IndexPreTransform_index(inner) };
        FaissIndexImpl { inner }
    }
}

#[cfg(test)]
#[test]
fn test_pretransform_index_ok() -> Result<()> {
    use crate::transform::VectorTransform;
    use ndarray::{s, Array2};
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;
    let x = Array2::random([1024, 128], Uniform::new(0.0, 1.0));
    let mut transform = crate::transform::FaissRandomRotationMatrix::new_with(128, 64)?;
    transform.train(x.as_slice_memory_order().unwrap())?;
    let mut index = crate::index::flat::FaissIndexFlatL2::new_with(64)?;
    let mut index = FaissIndexPreTransform::new_with_transform(&mut index, &mut transform)?;
    index.add(x.as_slice_memory_order().unwrap())?;
    let query = x.slice(s![42, ..]);
    let (labels, scores) = index.search(query.as_slice_memory_order().unwrap(), 1)?;
    assert_eq!(labels[0], 42);
    assert!(scores[0].abs() < 1e-10);
    Ok(())
}
