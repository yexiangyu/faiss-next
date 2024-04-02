use std::ffi::CString;
use std::ptr::null_mut;

use crate::error::{Error, Result};
use crate::index::factory::IndexImpl;
use crate::index::IndexTrait;
use crate::macros::rc;

use faiss_next_sys as sys;
use tracing::trace;

#[repr(i32)]
pub enum IOFlag {
    /// mmap the file
    MMAP = 1,
    /// read-only
    ReadOnly = 2,
}

pub fn write_index(index: &impl IndexTrait, path: impl AsRef<str>) -> Result<()> {
    let path_ = path.as_ref();
    let path = CString::new(path_).map_err(|_| Error::InvalidPath {
        path: path.as_ref().to_string(),
    })?;
    trace!("wrting index={:?} to file={}", index.ptr(), path_);
    rc!({ sys::faiss_write_index_fname(index.ptr(), path.as_ptr()) })
}

pub fn read_index(path: impl AsRef<str>, flag: IOFlag) -> Result<IndexImpl> {
    let path_ = path.as_ref();
    let path = CString::new(path_).map_err(|_| Error::InvalidPath {
        path: path.as_ref().to_string(),
    })?;
    let mut inner = null_mut();
    rc!({ sys::faiss_read_index_fname(path.as_ptr(), flag as i32, &mut inner) })?;
    trace!("read index={:?} to file={}", inner, path_);
    Ok(IndexImpl::new(inner))
}
