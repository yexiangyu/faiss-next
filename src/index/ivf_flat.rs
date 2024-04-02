use std::mem::forget;
use std::ptr::null_mut;

use crate::error::Result;
use crate::index::ivf::IndexIVFTrait;
use crate::index::{impl_index, IndexTrait};
use crate::macros::rc;
use crate::metric::MetricType;
use faiss_next_sys as sys;

pub struct IndexIVFFlat {
    inner: *mut sys::FaissIndexIVFFlat,
}

impl_index!(IndexIVFFlat);

impl IndexIVFTrait for IndexIVFFlat {}

impl IndexIVFFlat {
    pub fn new(
        quantizer: impl IndexTrait,
        d: usize,
        nlist: usize,
        metric: MetricType,
    ) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexIVFFlat_new_with_metric(
                &mut inner,
                quantizer.ptr(),
                d,
                nlist,
                metric.into(),
            )
        })?;
        let mut r = Self { inner };
        r.set_own_fields(true);
        Ok(r)
    }

    pub fn cast(index: impl IndexTrait) -> Self {
        let inner = index.ptr();
        let inner = unsafe { sys::faiss_IndexIVFFlat_cast(inner) };
        forget(index);
        Self { inner }
    }

    pub fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        rc!({
            sys::faiss_IndexIVFFlat_add_core(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                xids.as_ref().as_ptr(),
                precomputed_idx.as_ref().as_ptr(),
            )
        })
    }

    pub fn update_vectors(
        &mut self,
        v: impl AsRef<[f32]>,
        mut idx: impl AsMut<[i64]>,
    ) -> Result<()> {
        let n = v.as_ref().len() as i32 / self.d();
        rc!({
            sys::faiss_IndexIVFFlat_update_vectors(
                self.ptr(),
                n,
                idx.as_mut().as_mut_ptr(),
                v.as_ref().as_ptr(),
            )
        })
    }
}
