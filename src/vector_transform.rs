#![allow(non_snake_case)]

use crate::error::Result;
use faiss_next_sys as ffi;

use std::{ptr::null_mut, slice::from_raw_parts};

pub trait VectorTransformTrait: std::fmt::Debug {
    fn inner(&self) -> *mut ffi::FaissVectorTransform;

    fn is_trained(&self) -> bool {
        ffi::run!(faiss_VectorTransform_is_trained, self.inner()) > 0
    }

    fn d_in(&self) -> i32 {
        ffi::run!(faiss_VectorTransform_d_in, self.inner())
    }

    fn d_out(&self) -> i32 {
        ffi::run!(faiss_VectorTransform_d_out, self.inner())
    }

    fn train(&self, x: impl AsRef<[f32]>) -> Result<()> {
        let n = x.as_ref().len() as i64;
        ffi::ok!(faiss_VectorTransform_train, self.inner(), n, x.as_ref().as_ptr())?;
        Ok(())
    }

    fn apply(&self, x: impl AsRef<[f32]>) -> &[f32] {
        let n = x.as_ref().len() as i64;
        let ptr = ffi::run!(faiss_VectorTransform_apply, self.inner(), n, x.as_ref().as_ptr());
        unsafe { from_raw_parts(ptr, self.d_out() as usize) }
    }

    fn apply_noalloc(&self, x: impl AsRef<[f32]>, mut y: impl AsMut<[f32]>) {
        let n = x.as_ref().len() as i64;
        assert_eq!(
            y.as_mut().len(),
            self.d_out() as usize,
            "y length ({}) must match d_out ({})",
            y.as_mut().len(),
            self.d_out()
        );
        ffi::run!(
            faiss_VectorTransform_apply_noalloc,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            y.as_mut().as_mut_ptr()
        );
    }

    fn reverse_transform(&self, x: impl AsRef<[f32]>, mut y: impl AsMut<[f32]>) {
        let n = x.as_ref().len() as i64;
        ffi::run!(
            faiss_VectorTransform_reverse_transform,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            y.as_mut().as_mut_ptr()
        );
    }

    fn set_is_orthonormal(&self) {
        ffi::run!(faiss_LinearTransform_set_is_orthonormal, self.inner());
    }

    fn is_orthonormal(&self) -> bool {
        ffi::run!(faiss_LinearTransform_is_orthonormal, self.inner()) > 0
    }

    fn have_bias(&self) -> bool {
        ffi::run!(faiss_LinearTransform_have_bias, self.inner()) > 0
    }
}

macro_rules! impl_vector_transform {
    ($cls: ident) => {
        impl VectorTransformTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissVectorTransform {
                self.inner as *mut _
            }
        }
    };
}

#[derive(Debug)]
pub struct RandomRotationMatrix {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(RandomRotationMatrix);
ffi::impl_drop!(RandomRotationMatrix, faiss_RandomRotationMatrix_free);

impl RandomRotationMatrix {
    pub fn new(d_in: i32, d_out: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_RandomRotationMatrix_new_with, &mut inner, d_in, d_out)?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct PCAMatrix {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(PCAMatrix);
ffi::impl_drop!(PCAMatrix, faiss_PCAMatrix_free);

impl PCAMatrix {
    pub fn new(d_in: i32, d_out: i32, eigen_power: f32, random_rotation: bool) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_PCAMatrix_new_with,
            &mut inner,
            d_in,
            d_out,
            eigen_power,
            random_rotation as i32
        )?;
        Ok(Self { inner })
    }

    pub fn eigen_power(&self) -> f32 {
        ffi::run!(faiss_PCAMatrix_eigen_power, self.inner as *mut _)
    }

    pub fn random_rotation(&self) -> bool {
        ffi::run!(faiss_PCAMatrix_random_rotation, self.inner as *mut _) > 0
    }
}

#[derive(Debug)]
pub struct ITQMatrix {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(ITQMatrix);
ffi::impl_drop!(ITQMatrix, faiss_ITQMatrix_free);

impl ITQMatrix {
    pub fn new(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_ITQMatrix_new_with, &mut inner, d)?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct ITQTransform {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(ITQTransform);
ffi::impl_drop!(ITQTransform, faiss_ITQTransform_free);

impl ITQTransform {
    pub fn new(d_in: i32, d_out: i32, do_pca: bool) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_ITQTransform_new_with, &mut inner, d_in, d_out, do_pca as i32)?;
        Ok(Self { inner })
    }

    pub fn do_pca(&self) -> bool {
        ffi::run!(faiss_ITQTransform_do_pca, self.inner as *mut _) > 0
    }
}

#[derive(Debug)]
pub struct OPQMatrix {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(OPQMatrix);
ffi::impl_drop!(OPQMatrix, faiss_OPQMatrix_free);

impl OPQMatrix {
    pub fn new(d: i32, m: i32, d_out: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_OPQMatrix_new_with, &mut inner, d, m, d_out)?;
        Ok(Self { inner })
    }

    pub fn verbose(&self) -> bool {
        ffi::run!(faiss_OPQMatrix_verbose, self.inner as *mut _) > 0
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        ffi::run!(faiss_OPQMatrix_set_verbose, self.inner as *mut _, verbose as i32);
    }

    pub fn niter(&self) -> i32 {
        ffi::run!(faiss_OPQMatrix_niter, self.inner as *mut _)
    }

    pub fn set_niter(&mut self, niter: i32) {
        ffi::run!(faiss_OPQMatrix_set_niter, self.inner as *mut _, niter);
    }

    pub fn niter_pq(&self) -> i32 {
        ffi::run!(faiss_OPQMatrix_niter_pq, self.inner as *mut _)
    }

    pub fn set_niter_pq(&mut self, niter_pq: i32) {
        ffi::run!(faiss_OPQMatrix_set_niter_pq, self.inner as *mut _, niter_pq);
    }
}

#[derive(Debug)]
pub struct RemapDimensionsTransform {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(RemapDimensionsTransform);
ffi::impl_drop!(RemapDimensionsTransform, faiss_RemapDimensionsTransform_free);

impl RemapDimensionsTransform {
    pub fn new(d_in: i32, d_out: i32, uniform: bool) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_RemapDimensionsTransform_new_with, &mut inner, d_in, d_out, uniform as i32)?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct NormalizationTransform {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(NormalizationTransform);
ffi::impl_drop!(NormalizationTransform, faiss_NormalizationTransform_free);

impl NormalizationTransform {
    pub fn new(d: i32, norm: f32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_NormalizationTransform_new_with, &mut inner, d, norm)?;
        Ok(Self { inner })
    }

    pub fn norm(&self) -> f32 {
        ffi::run!(faiss_NormalizationTransform_norm, self.inner as *mut _)
    }
}

#[derive(Debug)]
pub struct CenteringTransform {
    inner: *mut ffi::FaissVectorTransform,
}

impl_vector_transform!(CenteringTransform);
ffi::impl_drop!(CenteringTransform, faiss_CenteringTransform_free);

impl CenteringTransform {
    pub fn new(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_CenteringTransform_new_with, &mut inner, d)?;
        Ok(Self { inner })
    }
}
