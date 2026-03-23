use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::error::{Error, Result};

pub(crate) struct InnerPtr<T> {
    ptr: NonNull<T>,
    _marker: PhantomData<T>,
}

impl<T> InnerPtr<T> {
    pub fn new(ptr: *mut T) -> Result<Self> {
        NonNull::new(ptr)
            .map(|ptr| Self {
                ptr,
                _marker: PhantomData,
            })
            .ok_or(Error::NullPointer)
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Clone for InnerPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for InnerPtr<T> {}

unsafe impl<T> Send for InnerPtr<T> {}
unsafe impl<T> Sync for InnerPtr<T> {}
