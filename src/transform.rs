use crate::{
    error::Result,
    macros::{define_vector_transform_impl, faiss_rc},
};
use faiss_next_sys as sys;
use std::ptr::null_mut;

pub trait VectorTransformPtr {
    fn ptr(&self) -> *const sys::FaissVectorTransform;
    fn mut_ptr(&mut self) -> *mut sys::FaissVectorTransform;
    fn into_ptr(self) -> *mut sys::FaissVectorTransform;
}

pub trait VectorTransform: VectorTransformPtr {
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
        let x = x.as_ref();
        let n = x.len() / self.d_in() as usize;
        faiss_rc!(unsafe {
            sys::faiss_VectorTransform_train(self.mut_ptr(), n as i64, x.as_ref().as_ptr())
        })
    }

    fn apply(&self, x: impl AsRef<[f32]>) -> &[f32] {
        let x = x.as_ref();
        let n = x.len() / self.d_in() as usize;
        let out = unsafe { sys::faiss_VectorTransform_apply(self.ptr(), n as i64, x.as_ptr()) };
        let n_out = n * self.d_out() as usize;
        let out = unsafe { std::slice::from_raw_parts(out, n_out) };
        out
    }

    fn apply_noalloc(&self, x: impl AsRef<[f32]>, out: &mut [f32]) {
        let n = x.as_ref().len() as i64 / self.d_in() as i64;
        unsafe {
            sys::faiss_VectorTransform_apply_noalloc(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                out.as_mut_ptr(),
            )
        };
    }

    fn reverse_transform(&self, x: impl AsRef<[f32]>, out: &mut [f32]) {
        let n = x.as_ref().len() as i64 / self.d_out() as i64;
        unsafe {
            sys::faiss_VectorTransform_apply_noalloc(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                out.as_mut_ptr(),
            )
        };
    }
}

pub trait LinearTransform: VectorTransform {
    fn transform_transpose(&self, x: impl AsRef<[f32]>, out: &mut [f32]) {
        let n = x.as_ref().len() as i64 / self.d_out() as i64;
        unsafe {
            sys::faiss_LinearTransform_transform_transpose(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    fn have_bias(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_have_bias(self.ptr()) != 0 }
    }

    fn is_orthonormal(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_is_orthonormal(self.ptr()) != 0 }
    }
}

define_vector_transform_impl!(
    /// Random rotation matrix
    /// ```rust
    /// use faiss_next as faiss;
    /// use faiss::transform::{VectorTransform, FaissRandomRotationMatrix, LinearTransform};
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Uniform;
    ///
    /// let mut t = FaissRandomRotationMatrix::new_with(128, 64).unwrap();
    /// assert_eq!(t.d_in(), 128);
    /// assert_eq!(t.d_out(), 64);
    ///
    /// let x = ndarray::Array2::random([1024, 128], Uniform::new(0.0, 1.0));
    /// t.train(x.as_slice_memory_order().unwrap()).unwrap();
    /// assert!(t.is_trained());
    /// let y = ndarray::Array2::random([1, 128], Uniform::new(0.0, 1.0));
    /// let y_ = t.apply(y.as_slice_memory_order().unwrap());
    /// assert_eq!(y_.len(), 64);
    ///
    /// assert!(!t.have_bias());
    /// assert!(t.is_orthonormal());
    /// ```
    FaissRandomRotationMatrix,
    faiss_RandomRotationMatrix_free
);

impl FaissRandomRotationMatrix {
    pub fn new_with(d_in: i32, d_out: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_RandomRotationMatrix_new_with(&mut inner, d_in, d_out) })?;
        Ok(Self { inner })
    }
}

define_vector_transform_impl!(
    /// Random rotation matrix
    /// ```rust
    /// use faiss_next as faiss;
    /// use faiss::transform::{VectorTransform, FaissPCAMatrix, LinearTransform};
    /// use ndarray::Array2;
    /// use ndarray_rand::RandomExt;
    /// use rand::distributions::Uniform;
    ///
    /// let mut t = FaissPCAMatrix::new_with(128, 64, 0.0, false).unwrap();
    /// assert_eq!(t.d_in(), 128);
    /// assert_eq!(t.d_out(), 64);
    ///
    /// let x = ndarray::Array2::random([1024, 128], Uniform::new(0.0, 1.0));
    /// t.train(x.as_slice_memory_order().unwrap()).unwrap();
    /// assert!(t.is_trained());
    /// let y = ndarray::Array2::random([1, 128], Uniform::new(0.0, 1.0));
    /// let y_ = t.apply(y.as_slice_memory_order().unwrap());
    /// assert_eq!(y_.len(), 64);
    ///
    /// assert!(t.have_bias());
    /// assert!(t.is_orthonormal());
    /// ```
    FaissPCAMatrix,
    faiss_PCAMatrix_free
);

impl FaissPCAMatrix {
    pub fn new_with(
        d_in: i32,
        d_out: i32,
        eigen_power: f32,
        random_rotation: bool,
    ) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
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

define_vector_transform_impl!(FaissITQMatrix, faiss_ITQMatrix_free);
impl FaissITQMatrix {
    pub fn new_with(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_ITQMatrix_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}

define_vector_transform_impl!(FaissITQTransform, faiss_ITQTransform_free);

impl FaissITQTransform {
    pub fn new_with(d_in: i32, d_out: i32, do_pca: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_ITQTransform_new_with(&mut inner, d_in, d_out, do_pca as i32) })?;
        Ok(Self { inner })
    }

    pub fn do_pca(&self) -> bool {
        unsafe { sys::faiss_ITQTransform_do_pca(self.inner) != 0 }
    }
}

define_vector_transform_impl!(FaissOPQMatrix, faiss_OPQMatrix_free);

impl FaissOPQMatrix {
    pub fn new_with(d_in: i32, d_out: i32, n_ions: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_OPQMatrix_new_with(&mut inner, d_in, d_out, n_ions) })?;
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

define_vector_transform_impl!(
    FaissRemapDimensionsTransform,
    faiss_RemapDimensionsTransform_free
);

impl FaissRemapDimensionsTransform {
    pub fn new_with(d_in: i32, d_out: i32, uniform: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_RemapDimensionsTransform_new_with(&mut inner, d_in, d_out, uniform as i32)
        })?;
        Ok(Self { inner })
    }
}

define_vector_transform_impl!(
    FaissNormalizationTransform,
    faiss_NormalizationTransform_free
);

impl FaissNormalizationTransform {
    pub fn new_with(d: i32, norm: f32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_NormalizationTransform_new_with(&mut inner, d, norm) })?;
        Ok(Self { inner })
    }

    pub fn norm(&self) -> f32 {
        unsafe { sys::faiss_NormalizationTransform_norm(self.inner) }
    }
}

define_vector_transform_impl!(FaissCenteringTransform, faiss_CenteringTransform_free);

impl FaissCenteringTransform {
    pub fn new_with(d: i32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_CenteringTransform_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }
}
