use std::mem::forget;
use std::ptr::null_mut;

use crate::error::Result;
use crate::{error::*, macros::*, traits::FaissIndexTrait};
use faiss_next_sys::{self as ffi, FaissMetricType};

pub trait FaissIndexFlatTrait: FaissIndexTrait {
    fn xb(&self) -> &[f32] {
        let mut ptr = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_IndexFlat_xb(self.inner(), &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    fn xb_mut(&mut self) -> &mut [f32] {
        let mut ptr = null_mut();
        let mut size = 0usize;
        unsafe { ffi::faiss_IndexFlat_xb(self.inner(), &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    // TODO: change return to number of result, other than a ()
    fn compute_distance_subset(
        &mut self,
        x: impl AsRef<[f32]>,
        k: i64,
        mut distances: impl AsMut<[f32]>,
        labels: impl AsRef<[i64]>,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() % self.d() as usize, 0);
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(distances.as_mut().len(), n * k as usize);
        faiss_rc(unsafe {
            ffi::faiss_IndexFlat_compute_distance_subset(
                self.inner(),
                n as i64,
                x.as_ref().as_ptr(),
                k,
                distances.as_mut().as_mut_ptr(),
                labels.as_ref().as_ptr(),
            )
        })
    }
}

/// FaissIndexFlatIP
/// ```rust
/// use faiss_next::prelude::*;
/// use ndarray::{Array2, s};
/// use ndarray_rand::{rand_distr::Uniform, RandomExt};
///
/// let mut index = FaissIndexFlat::new(128, FaissMetricType::METRIC_L2).unwrap();
///
/// let base = Array2::<f32>::random((1024, 128), Uniform::new(-1.0, 1.0));
///
/// let query = base.slice(s![42..43, ..]);
///
/// index.add(base.as_slice().unwrap()).unwrap();
///
/// let mut distances = vec![0.0];
/// let mut labels = vec![-1];
///
/// index.search(query.as_slice().unwrap(), 1, &mut distances, &mut labels).unwrap();
/// assert_eq!(labels, &[42]);
/// assert_eq!(index.xb().len(), 1024 * 128);
/// ```
#[derive(Debug)]
pub struct FaissIndexFlat {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexFlat, faiss_IndexFlat_free);
impl FaissIndexTrait for FaissIndexFlat {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}
impl FaissIndexFlatTrait for FaissIndexFlat {}

impl FaissIndexFlat {
    // pub fn new() -> Result<Self> {
    //     let mut inner = null_mut();
    //     faiss_rc(unsafe { ffi::faiss_IndexFlat_new(&mut inner) })?;
    //     Ok(Self { inner })
    // }

    pub fn new(d: i64, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexFlat_new_with(&mut inner, d, metric) })?;
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<FaissIndexFlat> {
        let inner = index.inner();
        forget(index);
        let inner = unsafe { ffi::faiss_IndexFlat_cast(inner) };
        Ok(FaissIndexFlat { inner })
    }
}

/// FaissIndexFlatIP
/// ```rust
/// use faiss_next::prelude::*;
/// use ndarray::{Array2, s};
/// use ndarray_rand::{rand_distr::Uniform, RandomExt};
///
/// let mut index = FaissIndexFlatIP::new(128).unwrap();
///
/// let base = Array2::<f32>::random((1024, 128), Uniform::new(-1.0, 1.0));
///
/// let query = base.slice(s![42..43, ..]);
///
/// index.add(base.as_slice().unwrap()).unwrap();
///
/// let mut distances = vec![0.0];
/// let mut labels = vec![-1];
///
/// index.search(query.as_slice().unwrap(), 1, &mut distances, &mut labels).unwrap();
/// assert_eq!(labels, &[42]);
/// ```
#[derive(Debug)]
pub struct FaissIndexFlatIP {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexFlatIP, faiss_IndexFlatIP_free);
impl FaissIndexFlatIP {
    // pub fn new() -> Result<Self> {
    //     let mut inner = null_mut();
    //     faiss_rc(unsafe { ffi::faiss_IndexFlatIP_new(&mut inner) })?;
    //     Ok(Self { inner })
    // }

    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexFlatIP_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<FaissIndexFlatIP> {
        let inner = index.inner();
        forget(index);
        let inner = unsafe { ffi::faiss_IndexFlatIP_cast(inner) };
        Ok(FaissIndexFlatIP { inner })
    }
}
impl FaissIndexTrait for FaissIndexFlatIP {
    fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
        self.inner
    }
}
impl FaissIndexFlatTrait for FaissIndexFlatIP {}

/// FaissIndexFlatL2
/// ```rust
/// use faiss_next::prelude::*;
/// use ndarray::{Array2, s};
/// use ndarray_rand::{rand_distr::Uniform, RandomExt};
///
/// let mut index = FaissIndexFlatL2::new(128).unwrap();
///
/// let base = Array2::<f32>::random((1024, 128), Uniform::new(-1.0, 1.0));
///
/// let query = base.slice(s![42..43, ..]);
///
/// index.add(base.as_slice().unwrap()).unwrap();
///
/// let mut distances = vec![0.0];
/// let mut labels = vec![-1];
///
/// index.search(query.as_slice().unwrap(), 1, &mut distances, &mut labels).unwrap();
/// assert_eq!(labels, &[42]);
/// ```
#[derive(Debug)]
pub struct FaissIndexFlatL2 {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexFlatL2, faiss_IndexFlatL2_free);
impl FaissIndexFlatL2 {
    // pub fn new() -> Result<Self> {
    //     let mut inner = null_mut();
    //     faiss_rc(unsafe { ffi::faiss_IndexFlatL2_new(&mut inner) })?;
    //     Ok(Self { inner })
    // }

    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexFlatL2_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<FaissIndexFlatL2> {
        let inner = index.inner();
        forget(index);
        let inner = unsafe { ffi::faiss_IndexFlatL2_cast(inner) };
        Ok(FaissIndexFlatL2 { inner })
    }
}
impl FaissIndexTrait for FaissIndexFlatL2 {
    fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
        self.inner
    }
}
impl FaissIndexFlatTrait for FaissIndexFlatL2 {}

#[derive(Debug)]
pub struct FaissIndexRefineFlat {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexRefineFlat, faiss_IndexRefineFlat_free);
impl FaissIndexTrait for FaissIndexRefineFlat {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}
impl FaissIndexFlatTrait for FaissIndexRefineFlat {}

impl FaissIndexRefineFlat {
    pub fn new(index: impl FaissIndexTrait) -> Result<Self> {
        let index_inner = index.inner();
        forget(index);
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexRefineFlat_new(&mut inner, index_inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<FaissIndexRefineFlat> {
        let inner = index.inner();
        forget(index);
        let inner = unsafe { ffi::faiss_IndexRefineFlat_cast(inner) };
        Ok(FaissIndexRefineFlat { inner })
    }

    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexRefineFlat_own_fields(self.inner) > 0 }
    }

    fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { ffi::faiss_IndexRefineFlat_set_own_fields(self.inner, own_fields as i32) }
    }

    pub fn k_factor(&self) -> f32 {
        unsafe { ffi::faiss_IndexRefineFlat_k_factor(self.inner) }
    }

    pub fn set_k_factor(&self, val: f32) {
        unsafe { ffi::faiss_IndexRefineFlat_set_k_factor(self.inner, val) }
    }
}

#[derive(Debug)]
pub struct FaissIndexFlat1D {
    inner: *mut ffi::FaissIndex,
}
impl_faiss_drop!(FaissIndexFlat1D, faiss_IndexFlat1D_free);
impl FaissIndexTrait for FaissIndexFlat1D {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}
impl FaissIndexFlatTrait for FaissIndexFlat1D {}
impl FaissIndexFlat1D {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexFlat1D_new(&mut inner) })?;
        Ok(Self { inner })
    }
    pub fn new_with(continuous_update: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexFlat1D_new_with(&mut inner, continuous_update) })?;
        Ok(Self { inner })
    }
}
