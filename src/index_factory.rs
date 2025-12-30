use std::{ffi::CString, ptr::null_mut};

use crate::{
    error::*,
    index::{IndexOwned, MetricType},
};
use faiss_next_sys as ffi;

pub fn index_factory(
    d: i32,
    description: impl AsRef<str>,
    metric: MetricType,
) -> Result<IndexOwned> {
    let description = description.as_ref();
    let description = CString::new(description)?;
    let mut inner = null_mut();
    ffi::ok!(
        faiss_index_factory,
        &mut inner,
        d,
        description.as_ptr(),
        metric
    )?;
    Ok(IndexOwned::new(inner))
}

// pub fn faiss_index_binary_factory(
//     d: i32,
//     description: impl AsRef<str>,
// ) -> Result<FaissIndexBinaryOwned> {
//     let description = description.as_ref();
//     let description = std::ffi::CString::new(description).unwrap();
//     let mut inner = std::ptr::null_mut();
//     faiss_rc(ffi::extension::faiss_index_binary_factory(
//         d,
//         description.as_ptr(),
//         &mut inner,
//     ))?;
//     Ok(FaissIndexBinaryOwned { inner })
// }
