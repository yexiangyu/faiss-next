use crate::error::Result;
use faiss_next_sys as ffi;

use std::{
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use crate::index::{IndexTrait, MetricType};

pub trait IndexFlatTrait: IndexTrait {
    fn xb(&self) -> &[f32] {
        let mut ptr = null_mut();
        let mut size = 0usize;
        ffi::run!(faiss_IndexFlat_xb, self.inner(), &mut ptr, &mut size);
        unsafe { from_raw_parts(ptr, size) }
    }

    fn xb_mut(&mut self) -> &mut [f32] {
        let mut ptr = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_IndexFlat_xb(self.inner(), &mut ptr, &mut size) };
        unsafe { from_raw_parts_mut(ptr, size) }
    }

    // TODO: change return to number of result, other than a ()
    fn compute_distance_subset(
        &mut self,
        x: impl AsRef<[f32]>,
        k: i64,
        mut distances: impl AsMut<[f32]>,
        labels: impl AsRef<[i64]>,
    ) -> Result<()> {
        assert_eq!(
            x.as_ref().len() % self.d() as usize,
            0,
            "input vector length must be a multiple of the dimension"
        );
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(
            distances.as_mut().len(),
            n * k as usize,
            "output distance length must be a multiple of the number of input vectors"
        );
        ffi::ok!(
            faiss_IndexFlat_compute_distance_subset,
            self.inner(),
            n as i64,
            x.as_ref().as_ptr(),
            k,
            distances.as_mut().as_mut_ptr(),
            labels.as_ref().as_ptr()
        )?;
        Ok(())
    }
}

macro_rules! impl_index_flat {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexFlatTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexFlat {
    inner: *mut ffi::FaissIndex,
}

impl_index_flat!(IndexFlat);
ffi::impl_drop!(IndexFlat, faiss_IndexFlat_free);

impl IndexFlat {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlat_new, &mut inner)?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, metric: MetricType) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlat_new_with, &mut inner, d, metric)?;
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexFlat_cast, index.inner());
        assert!(inner.is_null(), "Failed to cast index");
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IndexFlatIP {
    inner: *mut ffi::FaissIndex,
}
ffi::impl_drop!(IndexFlatIP, faiss_IndexFlatIP_free);
impl_index_flat!(IndexFlatIP);

impl IndexFlatIP {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlatIP_new, &mut inner)?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlatIP_new_with, &mut inner, d)?;
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexFlatIP_cast, index.inner());
        assert!(inner.is_null(), "Failed to cast index");
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IndexFlatL2 {
    inner: *mut ffi::FaissIndex,
}

ffi::impl_drop!(IndexFlatL2, faiss_IndexFlatL2_free);
impl_index_flat!(IndexFlatL2);

impl IndexFlatL2 {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlatL2_new, &mut inner)?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlatL2_new_with, &mut inner, d)?;
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexFlatL2_cast, index.inner());
        assert!(inner.is_null(), "Failed to cast index");
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IndexRefineFlat {
    inner: *mut ffi::FaissIndex,
}
ffi::impl_drop!(IndexRefineFlat, faiss_IndexRefineFlat_free);
#[rustfmt::skip]
ffi::impl_getter!(IndexRefineFlat, k_factor, faiss_IndexRefineFlat_k_factor, usize);
#[rustfmt::skip]
ffi::impl_setter!( IndexRefineFlat, set_k_factor, faiss_IndexRefineFlat_set_k_factor, val, f32);
impl_index_flat!(IndexRefineFlat);

impl IndexRefineFlat {
    pub fn new(index: impl IndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexRefineFlat_new, &mut inner, index.inner())?;
        ffi::run!(faiss_IndexRefineFlat_set_own_fields, inner, true as i32);
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexRefineFlat_cast, index.inner());
        assert!(inner.is_null(), "Failed to cast index");
        Ok(Self { inner })
    }
}
#[derive(Debug)]
pub struct IndexFlat1D {
    inner: *mut ffi::FaissIndex,
}
ffi::impl_drop!(IndexFlat1D, faiss_IndexFlat1D_free);
impl_index_flat!(IndexFlat1D);
impl IndexFlat1D {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexFlat1D_new, &mut inner)?;
        Ok(Self { inner })
    }

    pub fn new_with(continuous_update: bool) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexFlat1D_new_with,
            &mut inner,
            continuous_update as i32
        )?;
        Ok(Self { inner })
    }
}
