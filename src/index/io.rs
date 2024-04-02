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

#[cfg(test)]
#[test]
fn test_index_io_ok() -> Result<()> {
    use crate::{index::factory::index_factory, metric::MetricType, prelude::*};
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use tracing::*;
    std::env::set_var("RUST_LOG", "trace");

    let _ = tracing_subscriber::fmt::try_init();
    let (n, d) = (1024usize, 128usize);
    let base = Array2::<f32>::random([n, d], Uniform::new(-1.0, 1.0));
    let base = base.as_slice().ok_or(Error::NotStandardLayout)?;

    let mut index = index_factory(d, "Flat", MetricType::L2)?;
    index.add(base, Option::<&[_]>::None)?;

    write_index(&index, "index_l2.idx")?;

    let index_ = read_index("index_l2.idx", IOFlag::ReadOnly)?;
    let mut index_ = crate::index::flat::IndexFlat::cast(index_);

    let base_ = index_.xb();
    info!("read index == write_index: {}", base_ == base);
    Ok(())
}
