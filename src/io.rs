use std::ffi::CString;
use std::ptr;

use crate::error::{check_return_code, Result};
use crate::index::{BinaryIndex, Index, IndexImpl};

pub fn write_index(index: &impl Index, path: &str) -> Result<()> {
    let c_path = CString::new(path)?;
    check_return_code(unsafe {
        faiss_next_sys::faiss_write_index_fname(index.inner_ptr(), c_path.as_ptr())
    })
}

pub fn read_index(path: &str) -> Result<IndexImpl> {
    let c_path = CString::new(path)?;
    let mut inner = ptr::null_mut();
    check_return_code(unsafe {
        faiss_next_sys::faiss_read_index_fname(c_path.as_ptr(), 0, &mut inner)
    })?;
    IndexImpl::from_raw(inner)
}

pub fn write_index_binary(index: &crate::index::IndexBinary, path: &str) -> Result<()> {
    let c_path = CString::new(path)?;
    check_return_code(unsafe {
        faiss_next_sys::faiss_write_index_binary_fname(index.inner_ptr(), c_path.as_ptr())
    })
}

pub fn read_index_binary(path: &str) -> Result<crate::index::IndexBinary> {
    let c_path = CString::new(path)?;
    let mut inner = ptr::null_mut();
    check_return_code(unsafe {
        faiss_next_sys::faiss_read_index_binary_fname(c_path.as_ptr(), 0, &mut inner)
    })?;
    crate::index::IndexBinary::from_raw(inner)
}
