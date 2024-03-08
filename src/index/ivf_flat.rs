use crate::error::Result;
use crate::rc;
use std::ptr::{addr_of_mut, null_mut};
use tracing::*;

use faiss_next_sys as sys;

use super::{
    common::{impl_index_drop, impl_index_trait, FaissIndexTrait},
    ivf::FaissIndexIVFTrait,
    metric::FaissMetricType,
};

pub struct FaissIndexIVFFlat {
    inner: *mut sys::FaissIndexIVFFlat,
}

impl_index_trait!(FaissIndexIVFFlat);
impl_index_drop!(FaissIndexIVFFlat, faiss_IndexIVFFlat_free);

impl FaissIndexIVFTrait for FaissIndexIVFFlat {}

impl FaissIndexIVFFlat {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexIVFFlat_new(addr_of_mut!(inner)) })?;
        Ok(Self { inner })
    }

    pub fn new_with(quantizer: impl FaissIndexTrait, d: usize, nlist: usize) -> Result<Self> {
        let quantizer = quantizer.into_inner();
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexIVFFlat_new_with(addr_of_mut!(inner), quantizer, d, nlist) })?;
        unsafe { sys::faiss_IndexIVFFlat_set_own_fields(inner, true as i32) }
        trace!(
            "create ivf flat index {:?}, with quantizer={:?}",
            inner,
            quantizer
        );
        Ok(Self { inner })
    }

    pub fn new_with_metric(
        quantizer: impl FaissIndexTrait,
        d: usize,
        nlist: usize,
        metric: FaissMetricType,
    ) -> Result<Self> {
        let quantizer = quantizer.into_inner();
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexIVFFlat_new_with_metric(
                addr_of_mut!(inner),
                quantizer,
                d,
                nlist,
                metric,
            )
        })?;
        unsafe { sys::faiss_IndexIVFFlat_set_own_fields(inner, true as i32) }
        trace!(
            "create ivf flat index {:?}, with quantizer={:?}, metric={:?}",
            inner,
            quantizer,
            metric
        );
        Ok(Self { inner })
    }

    pub fn add_core(&mut self, x: &[f32], xids: &[i64], precomputed_idx: &[i64]) -> Result<()> {
        let n = x.len() as i64 / self.d() as i64;
        rc!({
            sys::faiss_IndexIVFFlat_add_core(
                self.inner,
                n,
                x.as_ptr(),
                xids.as_ptr(),
                precomputed_idx.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn update_vectors(&mut self, v: &[f32], idx: &mut [i64]) -> Result<()> {
        let n = v.len() as i32 / self.d();
        assert_eq!(n, idx.len() as i32);
        rc!({
            sys::faiss_IndexIVFFlat_update_vectors(self.inner, n, idx.as_mut_ptr(), v.as_ptr())
        })?;
        Ok(())
    }
}
