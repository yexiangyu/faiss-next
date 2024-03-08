use crate::error::Result;
use crate::rc;
use faiss_next_sys as sys;
use tracing::trace;

pub trait FaissVectorTransformTrait {
    fn inner(&self) -> *mut sys::FaissVectorTransform;

    fn into_inner(self) -> *mut sys::FaissVectorTransform;

    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_VectorTransform_is_trained(self.inner()) != 0 }
    }

    fn d_in(&self) -> i32 {
        unsafe { sys::faiss_VectorTransform_d_in(self.inner()) }
    }

    fn d_out(&self) -> i32 {
        unsafe { sys::faiss_VectorTransform_d_out(self.inner()) }
    }

    fn train(&mut self, x: &[f32]) -> Result<()> {
        let n = x.len() as i64 / self.d_in() as i64;
        trace!(
            "train transformer inner={:?}, n={}, x.len={}",
            self.inner(),
            n,
            x.len()
        );
        rc!({ sys::faiss_VectorTransform_train(self.inner(), n, x.as_ptr()) })?;
        Ok(())
    }

    fn apply(&mut self, x: &[f32]) -> &[f32] {
        let n = x.len() as i64 / self.d_in() as i64;
        let out = unsafe { sys::faiss_VectorTransform_apply(self.inner(), n, x.as_ptr()) };
        unsafe { std::slice::from_raw_parts(out, n as usize * self.d_out() as usize) }
    }

    fn apply_noalloc(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len() as i64 / self.d_in() as i64;
        unsafe {
            sys::faiss_VectorTransform_apply_noalloc(self.inner(), n, x.as_ptr(), out.as_mut_ptr())
        }
    }

    fn reverse_transform(&mut self, x: &[f32], y: &mut [f32]) {
        let n = x.len() as i64 / self.d_out() as i64;
        unsafe {
            sys::faiss_VectorTransform_reverse_transform(
                self.inner(),
                n,
                x.as_ptr(),
                y.as_mut_ptr(),
            )
        }
    }
}

pub trait FaissLinearTransformTrait: FaissVectorTransformTrait {
    fn transform_transpose(&self, y: &[f32], x: &mut [f32]) {
        let n = y.len() as i64 / self.d_out() as i64;
        unsafe {
            sys::faiss_LinearTransform_transform_transpose(
                self.inner(),
                n,
                y.as_ptr(),
                x.as_mut_ptr(),
            )
        }
    }

    fn is_orthonormal(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_is_orthonormal(self.inner()) != 0 }
    }

    fn set_is_orthornormal(&mut self) {
        unsafe { sys::faiss_LinearTransform_set_is_orthonormal(self.inner()) }
    }

    fn have_bias(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_have_bias(self.inner()) != 0 }
    }
}

macro_rules! impl_vector_transform {
    ($kls: ident, $free: ident) => {
        impl FaissVectorTransformTrait for $kls {
            fn inner(&self) -> *mut sys::FaissVectorTransform {
                self.inner
            }

            fn into_inner(self) -> *mut sys::FaissVectorTransform {
                let mut s = self;
                let inner = s.inner;
                s.inner = std::ptr::null_mut();
                inner
            }
        }

        impl Drop for $kls {
            fn drop(&mut self) {
                trace!("drop vector transform inner={:?}", self.inner);
                if !self.inner.is_null() {
                    unsafe { sys::$free(self.inner) }
                }
            }
        }
    };
}

macro_rules! impl_linear_transform {
    ($kls: ident) => {
        impl FaissLinearTransformTrait for $kls {}
    };
}

pub struct FaissVectorTransformImpl {
    inner: *mut sys::FaissVectorTransform,
}

impl_vector_transform!(FaissVectorTransformImpl, faiss_VectorTransform_free);

pub struct FaissLinearTransformImpl {
    inner: *mut sys::FaissLinearTransform,
}

impl_vector_transform!(FaissLinearTransformImpl, faiss_LinearTransform_free);
impl_linear_transform!(FaissLinearTransformImpl);

pub struct FaissRandomRotationMatrix {
    inner: *mut sys::FaissRandomRotationMatrix,
}

impl_vector_transform!(FaissRandomRotationMatrix, faiss_RandomRotationMatrix_free);
impl_linear_transform!(FaissRandomRotationMatrix);

impl FaissRandomRotationMatrix {
    pub fn new_with(d_in: i32, d_out: i32) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_RandomRotationMatrix_new_with(&mut inner, d_in, d_out) })?;
        Ok(Self { inner })
    }
}

pub struct FaissPCAMatrix {
    inner: *mut sys::FaissPCAMatrix,
}

impl_vector_transform!(FaissPCAMatrix, faiss_PCAMatrix_free);
impl_linear_transform!(FaissPCAMatrix);

impl FaissPCAMatrix {
    pub fn new_with(
        d_in: i32,
        d_out: i32,
        eigen_power: f32,
        random_rotation: bool,
    ) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
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

pub struct FaissITQMatrix {
    inner: *mut sys::FaissITQMatrix,
}

impl_vector_transform!(FaissITQMatrix, faiss_ITQMatrix_free);
impl_linear_transform!(FaissITQMatrix);

impl FaissITQMatrix {
    pub fn new(d: i32) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_ITQMatrix_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}

pub struct FaissITQTransform {
    inner: *mut sys::FaissITQTransform,
}

impl_vector_transform!(FaissITQTransform, faiss_ITQTransform_free);
impl_linear_transform!(FaissITQTransform);

impl FaissITQMatrix {
    pub fn new_with(d_in: i32, d_out: i32, do_pca: bool) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_ITQTransform_new_with(&mut inner, d_in, d_out, do_pca as i32) })?;
        Ok(Self { inner })
    }

    pub fn do_pca(&self) -> bool {
        unsafe { sys::faiss_ITQTransform_do_pca(self.inner) != 0 }
    }
}

pub struct FaissOPQMatrix {
    inner: *mut sys::FaissOPQMatrix,
}

impl_vector_transform!(FaissOPQMatrix, faiss_OPQMatrix_free);
impl_linear_transform!(FaissOPQMatrix);

impl FaissOPQMatrix {
    pub fn new_with(d: i32, m: i32, d2: i32) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_OPQMatrix_new_with(&mut inner, d, m, d2) })?;
        Ok(Self { inner })
    }

    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_OPQMatrix_verbose(self.inner) != 0 }
    }

    pub fn niter(&self) -> i32 {
        unsafe { sys::faiss_OPQMatrix_niter(self.inner) }
    }

    pub fn niter_pq(&self) -> i32 {
        unsafe { sys::faiss_OPQMatrix_niter_pq(self.inner) }
    }
}

pub struct FaissRemapDimensionsTransform {
    inner: *mut sys::FaissRemapDimensionsTransform,
}

impl_vector_transform!(
    FaissRemapDimensionsTransform,
    faiss_RemapDimensionsTransform_free
);
impl_linear_transform!(FaissRemapDimensionsTransform);

impl FaissRemapDimensionsTransform {
    pub fn new_with(d_in: i32, d_out: i32, uniform: bool) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({
            sys::faiss_RemapDimensionsTransform_new_with(&mut inner, d_in, d_out, uniform as i32)
        })?;
        Ok(Self { inner })
    }
}

pub struct FaissNormalizationTransform {
    inner: *mut sys::FaissNormalizationTransform,
}

impl_vector_transform!(
    FaissNormalizationTransform,
    faiss_NormalizationTransform_free
);
impl_linear_transform!(FaissNormalizationTransform);

impl FaissNormalizationTransform {
    pub fn new_with(d: i32, norm: f32) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_NormalizationTransform_new_with(&mut inner, d, norm) })?;
        Ok(Self { inner })
    }

    pub fn norm(&self) -> f32 {
        unsafe { sys::faiss_NormalizationTransform_norm(self.inner) }
    }
}

pub struct FaissCenteringTransform {
    inner: *mut sys::FaissCenteringTransform,
}

impl_vector_transform!(FaissCenteringTransform, faiss_CenteringTransform_free);
impl_linear_transform!(FaissCenteringTransform);

impl FaissCenteringTransform {
    pub fn new_with(d: i32) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_CenteringTransform_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}
