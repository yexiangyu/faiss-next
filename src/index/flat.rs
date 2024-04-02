use std::mem::forget;
use std::ptr::null_mut;
use std::slice::from_raw_parts_mut;

use faiss_next_sys as sys;
use tracing::trace;

use crate::error::Result;
use crate::index::{impl_index, IndexTrait};
use crate::macros::rc;
use crate::metric::MetricType;

pub trait IndexFlatTrait: IndexTrait {
    fn xb(&mut self) -> &mut [f32] {
        let mut data = null_mut();
        let mut len = 0usize;
        unsafe {
            sys::faiss_IndexFlat_xb(self.ptr(), &mut data, &mut len);
            from_raw_parts_mut(data, len)
        }
    }

    fn compute_distance_subset(
        &mut self,
        x: impl AsRef<[f32]>,
        k: i64,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
    ) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        let x = x.as_ref();
        let distances = distances.as_mut();
        let labels = labels.as_mut();
        rc!({
            sys::faiss_IndexFlat_compute_distance_subset(
                self.ptr(),
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })
    }
}

macro_rules! impl_index_flat {
    ($cls: ty) => {
        impl_index!($cls);
        impl IndexFlatTrait for $cls {}
    };
}

pub struct IndexFlat {
    inner: *mut sys::FaissIndexFlat,
}

impl_index_flat!(IndexFlat);

impl IndexFlat {
    pub fn new(d: usize, metric: MetricType) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlat_new_with(&mut inner, d as _, metric.into()) })?;
        trace!(?d, ?metric, "new IndexFlat inner={:?}", inner);
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Self {
        let ptr = index.ptr();
        let inner = unsafe { sys::faiss_IndexFlat_cast(ptr) };
        trace!(
            "cast index={:?} to IndexFlat inner={:?}",
            index.ptr(),
            inner
        );
        forget(index);
        Self { inner }
    }
}

pub struct IndexFlatIP {
    inner: *mut sys::FaissIndexFlatIP,
}
impl_index_flat!(IndexFlatIP);
impl IndexFlatIP {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlatIP_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}

pub struct IndexFlatL2 {
    inner: *mut sys::FaissIndexFlatL2,
}
impl_index_flat!(IndexFlatL2);
impl IndexFlatL2 {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlatL2_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}

pub struct IndexRefineFlat {
    inner: *mut sys::FaissIndexRefineFlat,
}

impl_index_flat!(IndexRefineFlat);

impl IndexRefineFlat {
    pub fn new(base: impl IndexTrait, k_factor: Option<f32>) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexRefineFlat_new(&mut inner, base.ptr()) })?;
        let mut r = Self { inner };
        if let Some(k_factor) = k_factor {
            r.set_k_factor(k_factor);
        }
        r.set_own_fields(true);
        Ok(r)
    }

    fn set_own_fields(&mut self, own: bool) {
        unsafe { sys::faiss_IndexRefineFlat_set_own_fields(self.inner, own as i32) }
    }

    fn set_k_factor(&mut self, factor: f32) {
        unsafe { sys::faiss_IndexRefineFlat_set_k_factor(self.inner, factor) }
    }
}

pub struct IndexFlat1D {
    inner: *mut sys::FaissIndexFlat1D,
}
impl_index_flat!(IndexFlat1D);

impl IndexFlat1D {
    pub fn new(continuous_update: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlat1D_new_with(&mut inner, continuous_update as i32) })?;
        Ok(Self { inner })
    }

    pub fn update_permutation(&mut self) -> Result<()> {
        rc!({ sys::faiss_IndexFlat1D_update_permutation(self.inner) })
    }
}
