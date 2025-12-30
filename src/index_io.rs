use std::{ffi::CString, ptr::null_mut};

use crate::{
    error::*,
    index::{IndexOwned},
};
use faiss_next_sys as ffi;

/// IO flags for index read/write operations
pub const IO_FLAG_MMAP: i32 = 1;
pub const IO_FLAG_READ_ONLY: i32 = 2;

/// Write index to a file.
/// This is equivalent to `faiss::write_index` when a file descriptor is provided.
pub fn write_index_to_fd(_index: &impl crate::index::IndexTrait, _fd: i32) -> Result<()> {
    // Note: This function requires a FILE* which needs to be obtained from the C runtime
    // For direct file descriptor usage, we'd need a more complex binding setup
    // We'll implement it using the filename approach for now
    todo!("write_index_to_fd requires more complex binding setup with FILE* from C runtime")
}

/// Write index to a file.
/// This is equivalent to `faiss::write_index` when a file path is provided.
pub fn write_index_to_file(index: &impl crate::index::IndexTrait, filename: impl AsRef<str>) -> Result<()> {
    let filename = filename.as_ref();
    let filename = CString::new(filename)?;
    ffi::ok!(faiss_write_index_fname, index.inner(), filename.as_ptr())?;
    Ok(())
}

/// Write index to a custom writer.
pub fn write_index_to_custom_writer(
    index: &impl crate::index::IndexTrait,
    io_writer: *mut ffi::FaissIOWriter,
    io_flags: i32,
) -> Result<()> {
    ffi::ok!(faiss_write_index_custom, index.inner(), io_writer, io_flags)?;
    Ok(())
}

/// Read index from a file descriptor.
/// This is equivalent to `faiss::read_index` when a file descriptor is given.
pub fn read_index_from_fd(_fd: i32, _io_flags: i32) -> Result<IndexOwned> {
    // Note: This function requires a FILE* which needs to be obtained from the C runtime
    // For direct file descriptor usage, we'd need a more complex binding setup 
    todo!("read_index_from_fd requires more complex binding setup with FILE* from C runtime")
}

/// Read index from a file.
/// This is equivalent to `faiss::read_index` when a file path is given.
pub fn read_index_from_file(filename: impl AsRef<str>, io_flags: i32) -> Result<IndexOwned> {
    let filename = filename.as_ref();
    let filename = CString::new(filename)?;
    let mut inner = null_mut();
    ffi::ok!(faiss_read_index_fname, filename.as_ptr(), io_flags, &mut inner)?;
    Ok(IndexOwned::new(inner))
}

/// Read index from a custom reader.
pub fn read_index_from_custom_reader(
    io_reader: *mut ffi::FaissIOReader,
    io_flags: i32,
) -> Result<IndexOwned> {
    let mut inner = null_mut();
    ffi::ok!(faiss_read_index_custom, io_reader, io_flags, &mut inner)?;
    Ok(IndexOwned::new(inner))
}

// Binary index I/O functions (commented out until index_binary module is properly implemented)
// use crate::index_binary::{FaissIndexBinaryOwned, FaissIndexBinaryTrait};

// /// Write binary index to a file.
// /// This is equivalent to `faiss::write_index_binary` when a file descriptor is provided.
// pub fn write_index_binary_to_fd(index: &impl FaissIndexBinaryTrait, fd: i32) -> Result<()> {
//     // Note: This function requires a FILE* which needs to be obtained from the C runtime
//     todo!("write_index_binary_to_fd requires more complex binding setup with FILE* from C runtime")
// }

// /// Write binary index to a file.
// /// This is equivalent to `faiss::write_index_binary` when a file path is provided.
// pub fn write_index_binary_to_file(index: &impl FaissIndexBinaryTrait, filename: impl AsRef<str>) -> Result<()> {
//     let filename = filename.as_ref();
//     let filename = CString::new(filename)?;
//     ffi::ok!(faiss_write_index_binary_fname, index.inner(), filename.as_ptr())?;
//     Ok(())
// }

// /// Write binary index to a custom writer.
// pub fn write_index_binary_to_custom_writer(
//     index: &impl FaissIndexBinaryTrait,
//     io_writer: *mut ffi::FaissIOWriter,
// ) -> Result<()> {
//     ffi::ok!(faiss_write_index_binary_custom, index.inner(), io_writer)?;
//     Ok(())
// }

// /// Read binary index from a file descriptor.
// /// This is equivalent to `faiss::read_index_binary` when a file descriptor is given.
// pub fn read_index_binary_from_fd(fd: i32, io_flags: i32) -> Result<FaissIndexBinaryOwned> {
//     // Note: This function requires a FILE* which needs to be obtained from the C runtime
//     todo!("read_index_binary_from_fd requires more complex binding setup with FILE* from C runtime")
// }

// /// Read binary index from a file.
// /// This is equivalent to `faiss::read_index_binary` when a file path is given.
// pub fn read_index_binary_from_file(filename: impl AsRef<str>, io_flags: i32) -> Result<FaissIndexBinaryOwned> {
//     let filename = filename.as_ref();
//     let filename = CString::new(filename)?;
//     let mut inner = null_mut();
//     ffi::ok!(faiss_read_index_binary_fname, filename.as_ptr(), io_flags, &mut inner)?;
//     Ok(FaissIndexBinaryOwned { inner })
// }

// /// Read binary index from a custom reader.
// pub fn read_index_binary_from_custom_reader(
//     io_reader: *mut ffi::FaissIOReader,
//     io_flags: i32,
// ) -> Result<FaissIndexBinaryOwned> {
//     let mut inner = null_mut();
//     ffi::ok!(faiss_read_index_binary_custom, io_reader, io_flags, &mut inner)?;
//     Ok(FaissIndexBinaryOwned { inner })
// }

// Vector Transform I/O function
#[derive(Debug)]
pub struct VectorTransformOwned {
    inner: *mut ffi::FaissVectorTransform,
}

impl VectorTransformOwned {
    pub(crate) fn new(inner: *mut ffi::FaissVectorTransform) -> Self {
        Self { inner }
    }
}

ffi::impl_drop!(VectorTransformOwned, faiss_VectorTransform_free);

impl VectorTransformOwned {
    pub fn inner(&self) -> *mut ffi::FaissVectorTransform {
        self.inner
    }
}

/// Read vector transform from a file.
/// This is equivalent to `faiss::read_VectorTransform` when a file path is given.
pub fn read_vector_transform_from_file(filename: impl AsRef<str>) -> Result<VectorTransformOwned> {
    let filename = filename.as_ref();
    let filename = CString::new(filename)?;
    let mut inner = null_mut();
    ffi::ok!(faiss_read_VectorTransform_fname, filename.as_ptr(), &mut inner)?;
    Ok(VectorTransformOwned::new(inner))
}