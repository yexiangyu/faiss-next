use std::ptr::null_mut;

use faiss_next_sys as sys;

use crate::error::Result;
use crate::macros::rc;

pub trait VectorTransformTrait {
    fn ptr(&self) -> *mut sys::FaissVectorTransform;

    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_VectorTransform_is_trained(self.ptr()) != 0 }
    }

    fn d_in(&self) -> i32 {
        unsafe { sys::faiss_VectorTransform_d_in(self.ptr()) }
    }

    fn d_out(&self) -> i32 {
        unsafe { sys::faiss_VectorTransform_d_out(self.ptr()) }
    }

    fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d_in() as i64;
        rc!({ sys::faiss_VectorTransform_train(self.ptr(), n, x.as_ref().as_ptr()) })
    }

    fn apply(&self, x: impl AsRef<[f32]>, mut out: impl AsMut<[f32]>) {
        let n = x.as_ref().len() as i64 / self.d_in() as i64;
        unsafe {
            sys::faiss_VectorTransform_apply_noalloc(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                out.as_mut().as_mut_ptr(),
            )
        };
    }

    fn reverse_transform(&self, xt: impl AsRef<[f32]>, mut x: impl AsMut<[f32]>) {
        let n = xt.as_ref().len() as i64 / self.d_out() as i64;
        unsafe {
            sys::faiss_VectorTransform_reverse_transform(
                self.ptr(),
                n,
                xt.as_ref().as_ptr(),
                x.as_mut().as_mut_ptr(),
            )
        }
    }
}

pub trait LinearTransform: VectorTransformTrait {
    fn transform_transpose(&self, y: impl AsRef<[f32]>, mut x: impl AsMut<[f32]>) {
        let n = y.as_ref().len() as i64 / self.d_in() as i64;
        unsafe {
            sys::faiss_LinearTransform_transform_transpose(
                self.ptr(),
                n,
                y.as_ref().as_ptr(),
                x.as_mut().as_mut_ptr(),
            )
        }
    }

    fn set_is_orthonormal(&mut self) {
        unsafe { sys::faiss_LinearTransform_set_is_orthonormal(self.ptr()) }
    }

    fn have_bias(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_have_bias(self.ptr()) != 0 }
    }

    fn is_orthonormal(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_is_orthonormal(self.ptr()) != 0 }
    }
}

macro_rules! impl_vector_transform {
    ($cls: ty) => {
        impl VectorTransformTrait for $cls {
            fn ptr(&self) -> *mut faiss_next_sys::FaissVectorTransform {
                self.inner
            }
        }
        impl Drop for $cls {
            fn drop(&mut self) {
                if !self.inner.is_null() {
                    unsafe { faiss_next_sys::faiss_VectorTransform_free(self.inner) };
                }
            }
        }
    };
}

pub struct RandomRotationMatrix {
    inner: *mut sys::FaissRandomRotationMatrix,
}
impl_vector_transform!(RandomRotationMatrix);
impl RandomRotationMatrix {
    pub fn new(d_in: i32, d_out: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RandomRotationMatrix_new_with(&mut inner, d_in, d_out) })?;
        Ok(Self { inner })
    }
}

pub struct PCAMatrix {
    inner: *mut sys::FaissPCAMatrix,
}
impl_vector_transform!(PCAMatrix);
impl PCAMatrix {
    pub fn new(d_in: i32, d_out: i32, eigen_power: f32, random_rotation: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_PCAMatrix_new_with(
                &mut inner,
                d_in,
                d_out,
                eigen_power,
                random_rotation as i32,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn eigen_power(&self) -> f32 {
        unsafe { sys::faiss_PCAMatrix_eigen_power(self.inner) }
    }

    pub fn random_rotation(&self) -> bool {
        unsafe { sys::faiss_PCAMatrix_random_rotation(self.inner) != 0 }
    }
}

pub struct ITQMatrix {
    inner: *mut sys::FaissITQMatrix,
}
impl_vector_transform!(ITQMatrix);
impl ITQMatrix {
    pub fn new(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_ITQMatrix_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}

pub struct ITQTransform {
    inner: *mut sys::FaissITQTransform,
}
impl_vector_transform!(ITQTransform);
impl ITQTransform {
    pub fn new(d_in: i32, d_out: i32, do_pca: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_ITQTransform_new_with(&mut inner, d_in, d_out, do_pca as i32) })?;
        Ok(Self { inner })
    }

    pub fn do_pca(&self) -> bool {
        unsafe { sys::faiss_ITQTransform_do_pca(self.inner) != 0 }
    }
}

pub struct OPQMatrix {
    inner: *mut sys::FaissOPQMatrix,
}
impl_vector_transform!(OPQMatrix);
impl OPQMatrix {
    #[allow(non_snake_case)]
    pub fn new(d: i32, M: i32, d2: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_OPQMatrix_new_with(&mut inner, d, M, d2) })?;
        Ok(Self { inner })
    }

    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_OPQMatrix_verbose(self.inner) != 0 }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_OPQMatrix_set_verbose(self.inner, verbose as i32) }
    }

    pub fn niter(&self) -> i32 {
        unsafe { sys::faiss_OPQMatrix_niter(self.inner) }
    }

    pub fn set_niter(&mut self, niter: i32) {
        unsafe { sys::faiss_OPQMatrix_set_niter(self.inner, niter) }
    }

    pub fn niter_pq(&self) -> i32 {
        unsafe { sys::faiss_OPQMatrix_niter_pq(self.inner) }
    }
    pub fn set_niter_pq(&mut self, niter_pq: i32) {
        unsafe { sys::faiss_OPQMatrix_set_niter_pq(self.inner, niter_pq) }
    }
}

pub struct RemapDimensionsTransform {
    inner: *mut sys::FaissRemapDimensionsTransform,
}
impl_vector_transform!(RemapDimensionsTransform);
impl RemapDimensionsTransform {
    pub fn new(d_in: i32, d_out: i32, uniform: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RemapDimensionsTransform_new_with(&mut inner, d_in, d_out, uniform) })?;
        Ok(Self { inner })
    }
}

pub struct NormalizationTransform {
    inner: *mut sys::FaissNormalizationTransform,
}
impl_vector_transform!(NormalizationTransform);
impl NormalizationTransform {
    pub fn new(d: i32, norm: f32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_NormalizationTransform_new_with(&mut inner, d, norm) })?;
        Ok(Self { inner })
    }
}

pub struct CenteringTransform {
    inner: *mut sys::FaissCenteringTransform,
}
impl_vector_transform!(CenteringTransform);

impl CenteringTransform {
    pub fn new(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_CenteringTransform_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}
