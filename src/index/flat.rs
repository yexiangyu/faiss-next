use crate::error::Result;
use crate::rc;
use std::ptr::{addr_of_mut, null_mut};

use faiss_next_sys as sys;
use sys::faiss_IndexRefineFlat_set_own_fields;
use tracing::trace;

use super::{
    common::{impl_index_drop, impl_index_trait, FaissIndexTrait},
    metric::FaissMetricType,
};

pub trait FaissIndexFlatTrait: FaissIndexTrait {
    fn xb(&mut self) -> &[f32] {
        let mut size = 0usize;
        let mut data = std::ptr::null_mut();
        unsafe {
            sys::faiss_IndexFlat_xb(
                self.inner(),
                std::ptr::addr_of_mut!(data),
                std::ptr::addr_of_mut!(size),
            );
            std::slice::from_raw_parts(data, size)
        }
    }
}

pub struct FaissIndexFlatImpl {
    inner: *mut sys::FaissIndexFlat,
}
impl_index_drop!(FaissIndexFlatImpl, faiss_IndexFlat_free);
impl_index_trait!(FaissIndexFlatImpl);

impl FaissIndexFlatImpl {
    pub fn new(d: i64, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlat_new_with(addr_of_mut!(inner), d, metric) })?;
        trace!("create faiss index flat impl inner={:?}", inner);
        Ok(Self { inner })
    }

    pub fn compute_distance_subset(&mut self, x: &[f32], labels: &[i64]) -> Result<Vec<f32>> {
        let n = x.len() as i64 / self.d() as i64;
        let k = labels.len() as i64 / n;
        let mut distances = vec![0.0; n as usize * k as usize];
        rc!({
            sys::faiss_IndexFlat_compute_distance_subset(
                self.inner(),
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_ptr(),
            )
        })?;
        Ok(distances)
    }
}

pub struct FaissIndexFlatIP {
    inner: *mut sys::FaissIndexFlatIP,
}

impl_index_drop!(FaissIndexFlatIP, faiss_IndexFlatIP_free);
impl_index_trait!(FaissIndexFlatIP);

impl FaissIndexFlatIP {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlatIP_new_with(addr_of_mut!(inner), d) })?;
        trace!("create faiss index flat ip inner={:?}", inner);
        Ok(Self { inner })
    }
}

pub struct FaissIndexFlatL2 {
    inner: *mut sys::FaissIndexFlatL2,
}

impl_index_drop!(FaissIndexFlatL2, faiss_IndexFlatL2_free);
impl_index_trait!(FaissIndexFlatL2);

impl FaissIndexFlatL2 {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlatL2_new_with(addr_of_mut!(inner), d) })?;
        trace!("create faiss index flat l2 inner={:?}", inner);
        Ok(Self { inner })
    }
}

pub trait FaissIndexRefineFlatTrait: FaissIndexTrait {
    fn k_factor(&self) -> f32 {
        unsafe { sys::faiss_IndexRefineFlat_k_factor(self.inner()) }
    }

    fn set_k_factor(&mut self, k_factor: f32) {
        unsafe { sys::faiss_IndexRefineFlat_set_k_factor(self.inner(), k_factor) }
    }
}

pub struct FaissIndexRefineFlatImpl {
    inner: *mut sys::FaissIndexRefineFlat,
}

impl_index_drop!(FaissIndexRefineFlatImpl, faiss_IndexRefineFlat_free);
impl_index_trait!(FaissIndexRefineFlatImpl);

impl FaissIndexRefineFlatImpl {
    pub fn new(base: impl FaissIndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        let base = base.into_inner();
        rc!({ sys::faiss_IndexRefineFlat_new(addr_of_mut!(inner), base) })?;
        trace!(
            "create faiss index refine flat inner={:?}, base={:?}",
            inner,
            base
        );
        unsafe { faiss_IndexRefineFlat_set_own_fields(inner, true as i32) };
        Ok(Self { inner })
    }

    pub fn k_factor(&self) -> f32 {
        unsafe { sys::faiss_IndexRefineFlat_k_factor(self.inner()) }
    }

    pub fn set_k_factor(&mut self, k_factor: f32) {
        unsafe { sys::faiss_IndexRefineFlat_set_k_factor(self.inner(), k_factor) }
    }
}

pub struct FaissIndexFlat1D {
    inner: *mut sys::FaissIndexFlat1D,
}
impl_index_drop!(FaissIndexFlat1D, faiss_IndexFlat1D_free);
impl_index_trait!(FaissIndexFlat1D);

impl FaissIndexFlat1D {
    pub fn new(continuous_update: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlat1D_new_with(addr_of_mut!(inner), continuous_update as i32) })?;
        Ok(Self { inner })
    }

    pub fn update_permutation(&mut self) -> Result<()> {
        rc!({ sys::faiss_IndexFlat1D_update_permutation(self.inner()) })?;
        Ok(())
    }
}
