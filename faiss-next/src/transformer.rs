use std::ptr::null_mut;

use crate::{
    error::Result,
    index::{FaissIndex, IndexInner},
    macros::faiss_rc,
    prelude::Index,
};
use faiss_next_sys as sys;

pub trait VectorTransformInner {
    fn inner(&self) -> *mut sys::FaissVectorTransform;
}

pub trait VectorTransform: VectorTransformInner {
    fn is_trained(&self) -> bool {
        unsafe { sys::faiss_VectorTransform_is_trained(self.inner()) != 0 }
    }

    fn d_in(&self) -> usize {
        unsafe { sys::faiss_VectorTransform_d_in(self.inner()) as usize }
    }
    fn d_out(&self) -> usize {
        unsafe { sys::faiss_VectorTransform_d_out(self.inner()) as usize }
    }

    fn train(&mut self, x: &[f32]) {
        let n = x.len() / self.d_in();
        unsafe {
            sys::faiss_VectorTransform_train(self.inner(), n as sys::idx_t, x.as_ptr());
        }
    }

    fn apply(&self, x: &[f32]) -> Result<Vec<&[f32]>> {
        let n = x.len() / self.d_in();
        let ret =
            unsafe { sys::faiss_VectorTransform_apply(self.inner(), n as sys::idx_t, x.as_ptr()) };
        let ret = unsafe { std::slice::from_raw_parts(ret, n * self.d_out()) };
        Ok(ret.chunks(self.d_out()).collect())
    }
    fn apply_noalloc(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let n = x.len() / self.d_in();
        let mut ret = vec![0.0; n * self.d_out()];
        unsafe {
            sys::faiss_VectorTransform_apply_noalloc(
                self.inner(),
                n as sys::idx_t,
                x.as_ptr(),
                ret.as_mut_ptr(),
            )
        };
        ret.chunks(self.d_out()).map(|x| x.to_vec()).collect()
    }

    fn reverse_transform(&self, y: &[f32]) -> Vec<Vec<f32>> {
        let n = y.len() / self.d_out();
        let mut x = vec![0.0; n * self.d_in()];
        unsafe {
            sys::faiss_VectorTransform_reverse_transform(
                self.inner(),
                n as sys::idx_t,
                y.as_ptr(),
                x.as_mut_ptr(),
            );
        };
        x.chunks(self.d_in()).map(|x| x.to_vec()).collect()
    }
}

pub trait LinearTransform: VectorTransform {
    fn transform_transpose(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() / self.d_in();
        let mut y = vec![0.0; n * self.d_out()];
        unsafe {
            sys::faiss_LinearTransform_transform_transpose(
                self.inner(),
                n as sys::idx_t,
                x.as_ptr(),
                y.as_mut_ptr(),
            )
        };
        y
    }

    fn set_is_orthonormal(&mut self) {
        unsafe { sys::faiss_LinearTransform_set_is_orthonormal(self.inner()) }
    }

    fn is_orthonormal(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_is_orthonormal(self.inner()) != 0 }
    }

    fn have_bias(&self) -> bool {
        unsafe { sys::faiss_LinearTransform_have_bias(self.inner()) != 0 }
    }
}

macro_rules! define_linear_vector_transform {
    ($klass: ident, $drop: ident) => {
        pub struct $klass {
            inner: *mut sys::$klass,
        }

        impl Drop for $klass {
            fn drop(&mut self) {
                unsafe {
                    sys::$drop(self.inner);
                }
            }
        }

        impl VectorTransformInner for $klass {
            fn inner(&self) -> *mut sys::FaissVectorTransform {
                self.inner
            }
        }
        impl VectorTransform for $klass {}
        impl LinearTransform for $klass {}
        unsafe impl Send for $klass {}
        unsafe impl Sync for $klass {}
    };
}

define_linear_vector_transform!(FaissRandomRotationMatrix, faiss_RandomRotationMatrix_free);

impl FaissRandomRotationMatrix {
    pub fn new(d_in: usize, d_out: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_RandomRotationMatrix_new_with(&mut inner, d_in as i32, d_out as i32)
        })?;
        Ok(Self { inner })
    }
}

define_linear_vector_transform!(FaissPCAMatrix, faiss_PCAMatrix_free);

impl FaissPCAMatrix {
    pub fn new(d_in: usize, d_out: usize, eigen_power: f32, random_rotation: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_PCAMatrix_new_with(
                &mut inner,
                d_in as i32,
                d_out as i32,
                eigen_power,
                random_rotation as i32,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn eigen_power(&self) -> f32 {
        unsafe { sys::faiss_PCAMatrix_eigen_power(self.inner()) }
    }

    pub fn random_rotation(&self) -> bool {
        unsafe { sys::faiss_PCAMatrix_random_rotation(self.inner()) != 0 }
    }
}

define_linear_vector_transform!(FaissITQMatrix, faiss_ITQMatrix_free);

impl FaissITQMatrix {
    pub fn new(d: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_ITQMatrix_new_with(&mut inner, d as i32) })?;
        Ok(Self { inner })
    }
}

define_linear_vector_transform!(FaissOPQMatrix, faiss_OPQMatrix_free);

impl FaissOPQMatrix {
    pub fn new(d_in: usize, d_out: usize, n_ions: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_OPQMatrix_new_with(&mut inner, d_in as i32, d_out as i32, n_ions as i32)
        })?;
        Ok(Self { inner })
    }
    pub fn verbose(&self) -> bool {
        unsafe { sys::faiss_OPQMatrix_verbose(self.inner()) != 0 }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        unsafe { sys::faiss_OPQMatrix_set_verbose(self.inner(), verbose as i32) }
    }

    pub fn niter(&self) -> i32 {
        unsafe { sys::faiss_OPQMatrix_niter(self.inner()) }
    }

    pub fn set_niter(&mut self, niter: i32) {
        unsafe { sys::faiss_OPQMatrix_set_niter(self.inner(), niter) }
    }

    pub fn niter_pq(&self) -> i32 {
        unsafe { sys::faiss_OPQMatrix_niter_pq(self.inner()) }
    }
    pub fn set_niter_pq(&mut self, niter_pq: i32) {
        unsafe { sys::faiss_OPQMatrix_set_niter_pq(self.inner(), niter_pq) }
    }
}

define_linear_vector_transform!(FaissITQTransform, faiss_ITQTransform_free);

impl FaissITQTransform {
    pub fn new(d_in: usize, d_out: usize, do_pca: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_ITQTransform_new_with(&mut inner, d_in as i32, d_out as i32, do_pca as i32)
        })?;
        Ok(Self { inner })
    }
    pub fn do_pca(self) -> bool {
        unsafe { sys::faiss_ITQTransform_do_pca(self.inner) != 0 }
    }
}

define_linear_vector_transform!(
    FaissRemapDimensionsTransform,
    faiss_RemapDimensionsTransform_free
);

impl FaissRemapDimensionsTransform {
    pub fn new(d_in: usize, d_out: usize, uniform: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_RemapDimensionsTransform_new_with(
                &mut inner,
                d_in as i32,
                d_out as i32,
                uniform as i32,
            )
        })?;
        Ok(Self { inner })
    }
}

define_linear_vector_transform!(
    FaissNormalizationTransform,
    faiss_NormalizationTransform_free
);

impl FaissNormalizationTransform {
    pub fn new(d: usize, norm: f32) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_NormalizationTransform_new_with(&mut inner, d as i32, norm) })?;
        Ok(Self { inner })
    }

    pub fn norm(&self) -> f32 {
        unsafe { sys::faiss_NormalizationTransform_norm(self.inner()) }
    }
}

define_linear_vector_transform!(FaissCenteringTransform, faiss_CenteringTransform_free);

impl FaissCenteringTransform {
    pub fn new(d: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_CenteringTransform_new_with(&mut inner, d as i32) })?;
        Ok(Self { inner })
    }
}

pub struct FaissIndexPreTransform {
    inner: *mut sys::FaissIndexPreTransform,
    #[allow(unused)]
    index: FaissIndex,
    #[allow(unused)]
    transform: Box<dyn VectorTransform>,
}

impl Drop for FaissIndexPreTransform {
    fn drop(&mut self) {
        unsafe { sys::faiss_IndexPreTransform_free(self.inner) }
    }
}

impl FaissIndexPreTransform {
    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexPreTransform_own_fields(self.inner) != 0 }
    }

    pub fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexPreTransform_set_own_fields(self.inner, own_fields as i32) }
    }

    pub fn new(transform: impl VectorTransform + 'static, index: FaissIndex) -> Result<Self> {
        let transform = Box::new(transform);
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_IndexPreTransform_new_with_transform(
                &mut inner,
                transform.inner(),
                index.inner(),
            )
        })?;
        Ok(Self {
            inner,
            index,
            transform,
        })
    }
}

impl IndexInner for FaissIndexPreTransform {
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }
}

impl Index for FaissIndexPreTransform {}
