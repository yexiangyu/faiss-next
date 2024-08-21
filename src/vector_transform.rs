#![allow(non_snake_case)]

use std::slice::from_raw_parts;

use faiss_next_sys as ffi;

use crate::macros::*;

pub trait FaissVectorTransformTrait {
    fn inner(&self) -> *mut ffi::FaissVectorTransform;

    fn is_trained(&self) -> bool {
        unsafe { ffi::faiss_VectorTransform_is_trained(self.inner()) > 0 }
    }

    fn d_in(&self) -> i32 {
        unsafe { ffi::faiss_VectorTransform_d_in(self.inner()) }
    }

    fn d_out(&self) -> i32 {
        unsafe { ffi::faiss_VectorTransform_d_out(self.inner()) }
    }

    fn train(&self, x: impl AsRef<[f32]>) -> crate::error::Result<()> {
        let n = x.as_ref().len() as i64;
        crate::error::faiss_rc(unsafe {
            ffi::faiss_VectorTransform_train(self.inner(), n, x.as_ref().as_ptr())
        })
    }

    fn apply(&self, x: impl AsRef<[f32]>) -> &[f32] {
        let n = x.as_ref().len() as i64;
        let ptr = unsafe { ffi::faiss_VectorTransform_apply(self.inner(), n, x.as_ref().as_ptr()) };
        unsafe { from_raw_parts(ptr, self.d_out() as usize) }
    }

    fn apply_noalloc(&self, x: impl AsRef<[f32]>, mut y: impl AsMut<[f32]>) {
        let n = x.as_ref().len() as i64;
        assert_eq!(y.as_mut().len(), self.d_out() as usize);
        unsafe {
            ffi::faiss_VectorTransform_apply_noalloc(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                y.as_mut().as_mut_ptr(),
            )
        };
    }

    fn reverse_transform(&self, x: impl AsRef<[f32]>, mut y: impl AsMut<[f32]>) {
        let n = x.as_ref().len() as i64;
        unsafe {
            ffi::faiss_VectorTransform_reverse_transform(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                y.as_mut().as_mut_ptr(),
            )
        };
    }

    fn set_is_orthonormal(&self) {
        unsafe { ffi::faiss_LinearTransform_set_is_orthonormal(self.inner()) }
    }

    fn is_orthonomal(&self) -> bool {
        unsafe { ffi::faiss_LinearTransform_is_orthonormal(self.inner()) > 0 }
    }

    fn have_bias(&self) -> bool {
        unsafe { ffi::faiss_LinearTransform_have_bias(self.inner()) > 0 }
    }
}

pub struct FaissRandomRotationMatrix {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(FaissRandomRotationMatrix, faiss_RandomRotationMatrix_free);
impl_faiss_new!(
    FaissRandomRotationMatrix,
    new,
    FaissRandomRotationMatrix,
    faiss_RandomRotationMatrix_new_with,
    d_int,
    i32,
    d_out,
    i32
);

pub struct FaissPCAMatrix {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(FaissPCAMatrix, faiss_PCAMatrix_free);
impl_faiss_new!(
    FaissPCAMatrix,
    new,
    FaissPCAMatrix,
    faiss_PCAMatrix_new_with,
    d_in,
    i32,
    d_out,
    i32,
    eigen_power,
    f32,
    random_rotation,
    i32
);
impl_faiss_getter!(
    FaissPCAMatrix,
    eigen_power,
    faiss_PCAMatrix_eigen_power,
    f32
);
impl_faiss_getter!(
    FaissPCAMatrix,
    random_rotation,
    faiss_PCAMatrix_random_rotation,
    i32
);

pub struct FaissITQMatrix {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(FaissITQMatrix, faiss_ITQMatrix_free);
impl_faiss_new!(
    FaissITQMatrix,
    new,
    FaissITQMatrix,
    faiss_ITQMatrix_new_with,
    d,
    i32
);

pub struct FaissITQTransform {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(FaissITQTransform, faiss_ITQTransform_free);
impl_faiss_new!(
    FaissITQTransform,
    new,
    FaissITQTransform,
    faiss_ITQTransform_new_with,
    d_in,
    i32,
    d_out,
    i32,
    do_pca,
    i32
);
impl_faiss_getter!(FaissITQTransform, do_pca, faiss_ITQTransform_do_pca, i32);

pub struct FaissOPQMatrix {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(FaissOPQMatrix, faiss_OPQMatrix_free);
impl_faiss_new!(
    FaissOPQMatrix,
    new,
    FaissOPQMatrix,
    faiss_OPQMatrix_new_with,
    d,
    i32,
    M,
    i32,
    d_out,
    i32
);
impl_faiss_getter!(FaissOPQMatrix, verbose, faiss_OPQMatrix_verbose, i32);
impl_faiss_setter!(
    FaissOPQMatrix,
    set_verbose,
    faiss_OPQMatrix_set_verbose,
    verbose,
    i32
);
impl_faiss_getter!(FaissOPQMatrix, niter, faiss_OPQMatrix_niter, i32);
impl_faiss_setter!(
    FaissOPQMatrix,
    set_niter,
    faiss_OPQMatrix_set_niter,
    niter,
    i32
);
impl_faiss_getter!(FaissOPQMatrix, niter_pq, faiss_OPQMatrix_niter_pq, i32);
impl_faiss_setter!(
    FaissOPQMatrix,
    set_niter_pq,
    faiss_OPQMatrix_set_niter_pq,
    niter_pq,
    i32
);

pub struct FaissRemapDimensionsTransform {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(
    FaissRemapDimensionsTransform,
    faiss_RemapDimensionsTransform_free
);
impl_faiss_new!(
    FaissRemapDimensionsTransform,
    new,
    FaissRemapDimensionsTransform,
    faiss_RemapDimensionsTransform_new_with,
    d_in,
    i32,
    d_out,
    i32,
    uniform,
    i32
);

pub struct FaissNormalizationTransform {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(
    FaissNormalizationTransform,
    faiss_NormalizationTransform_free
);
impl_faiss_new!(
    FaissNormalizationTransform,
    new,
    FaissNormalizationTransform,
    faiss_NormalizationTransform_new_with,
    d,
    i32,
    norm,
    f32
);
impl_faiss_getter!(
    FaissNormalizationTransform,
    norm,
    faiss_NormalizationTransform_norm,
    f32
);

pub struct FaissCenteringTransform {
    pub inner: *mut ffi::FaissVectorTransform,
}
impl_faiss_drop!(FaissCenteringTransform, faiss_CenteringTransform_free);
impl_faiss_new!(
    FaissCenteringTransform,
    new,
    FaissCenteringTransform,
    faiss_CenteringTransform_new_with,
    d,
    i32
);

macro_rules! impl_vector_transorm {
    ($klass:ident) => {
        impl FaissVectorTransformTrait for $klass {
            fn inner(&self) -> *mut ffi::FaissVectorTransform {
                self.inner as *mut _
            }
        }
    };
}

impl_vector_transorm!(FaissRandomRotationMatrix);
impl_vector_transorm!(FaissPCAMatrix);
impl_vector_transorm!(FaissITQMatrix);
impl_vector_transorm!(FaissITQTransform);
impl_vector_transorm!(FaissOPQMatrix);
impl_vector_transorm!(FaissRemapDimensionsTransform);
impl_vector_transorm!(FaissNormalizationTransform);
impl_vector_transorm!(FaissCenteringTransform);
