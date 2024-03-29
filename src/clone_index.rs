use std::ptr::null_mut;

use crate::error::Result;
use crate::index::IndexTrait;
use crate::macros::rc;
use faiss_next_sys as sys;
use tracing::trace;

pub fn clone_index(index: &impl IndexTrait) -> Result<*mut sys::FaissIndex> {
    let mut inner = null_mut();
    rc!({ sys::faiss_clone_index(index.ptr(), &mut inner) })?;
    trace!("clone index, source={:?}, target={:?}", index.ptr(), inner);
    Ok(inner)
}
