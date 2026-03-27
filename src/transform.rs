use std::ptr;

use faiss_next_sys::{
    self, FaissCenteringTransform, FaissITQMatrix, FaissITQTransform, FaissLinearTransform,
    FaissNormalizationTransform, FaissOPQMatrix, FaissPCAMatrix, FaissRandomRotationMatrix,
    FaissRemapDimensionsTransform, FaissVectorTransform,
};

use crate::error::{check_return_code, Result};

pub trait VectorTransform {
    fn inner_ptr(&self) -> *mut FaissVectorTransform;

    fn is_trained(&self) -> bool {
        unsafe { faiss_next_sys::faiss_VectorTransform_is_trained(self.inner_ptr()) != 0 }
    }

    fn d_in(&self) -> u32 {
        unsafe { faiss_next_sys::faiss_VectorTransform_d_in(self.inner_ptr()) as u32 }
    }

    fn d_out(&self) -> u32 {
        unsafe { faiss_next_sys::faiss_VectorTransform_d_out(self.inner_ptr()) as u32 }
    }

    fn train(&mut self, n: usize, x: &[f32]) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_VectorTransform_train(self.inner_ptr(), n as i64, x.as_ptr())
        })
    }

    fn apply(&self, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        let d_out = self.d_out() as usize;
        let mut xt = vec![0.0f32; n * d_out];
        unsafe {
            faiss_next_sys::faiss_VectorTransform_apply_noalloc(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                xt.as_mut_ptr(),
            )
        }
        Ok(xt)
    }

    fn apply_noalloc(&self, n: usize, x: &[f32], xt: &mut [f32]) {
        unsafe {
            faiss_next_sys::faiss_VectorTransform_apply_noalloc(
                self.inner_ptr(),
                n as i64,
                x.as_ptr(),
                xt.as_mut_ptr(),
            )
        }
    }

    fn reverse_transform(&self, n: usize, xt: &[f32], x: &mut [f32]) {
        unsafe {
            faiss_next_sys::faiss_VectorTransform_reverse_transform(
                self.inner_ptr(),
                n as i64,
                xt.as_ptr(),
                x.as_mut_ptr(),
            )
        }
    }
}

pub trait LinearTransform: VectorTransform {
    fn inner_linear_ptr(&self) -> *mut FaissLinearTransform;

    fn transform_transpose(&self, n: usize, y: &[f32], x: &mut [f32]) {
        unsafe {
            faiss_next_sys::faiss_LinearTransform_transform_transpose(
                self.inner_linear_ptr(),
                n as i64,
                y.as_ptr(),
                x.as_mut_ptr(),
            )
        }
    }

    fn set_is_orthonormal(&mut self) {
        unsafe { faiss_next_sys::faiss_LinearTransform_set_is_orthonormal(self.inner_linear_ptr()) }
    }

    fn have_bias(&self) -> bool {
        unsafe { faiss_next_sys::faiss_LinearTransform_have_bias(self.inner_linear_ptr()) != 0 }
    }

    fn is_orthonormal(&self) -> bool {
        unsafe {
            faiss_next_sys::faiss_LinearTransform_is_orthonormal(self.inner_linear_ptr()) != 0
        }
    }
}

pub struct PcaMatrix {
    ptr: *mut FaissPCAMatrix,
}

impl PcaMatrix {
    pub fn new(d_in: u32, d_out: u32, eigen_power: f32, random_rotation: bool) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissPCAMatrix = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_PCAMatrix_new_with(
                &mut ptr,
                d_in as i32,
                d_out as i32,
                eigen_power,
                random_rotation as i32,
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn eigen_power(&self) -> f32 {
        unsafe { faiss_next_sys::faiss_PCAMatrix_eigen_power(self.ptr) }
    }

    pub fn random_rotation(&self) -> bool {
        unsafe { faiss_next_sys::faiss_PCAMatrix_random_rotation(self.ptr) != 0 }
    }
}

impl VectorTransform for PcaMatrix {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl LinearTransform for PcaMatrix {
    fn inner_linear_ptr(&self) -> *mut FaissLinearTransform {
        self.ptr as *mut FaissLinearTransform
    }
}

impl Drop for PcaMatrix {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_PCAMatrix_free(self.ptr);
            }
        }
    }
}

pub struct RandomRotationMatrix {
    ptr: *mut FaissRandomRotationMatrix,
}

impl RandomRotationMatrix {
    pub fn new(d_in: u32, d_out: u32) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissRandomRotationMatrix = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_RandomRotationMatrix_new_with(
                &mut ptr,
                d_in as i32,
                d_out as i32,
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl VectorTransform for RandomRotationMatrix {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl LinearTransform for RandomRotationMatrix {
    fn inner_linear_ptr(&self) -> *mut FaissLinearTransform {
        self.ptr as *mut FaissLinearTransform
    }
}

impl Drop for RandomRotationMatrix {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_RandomRotationMatrix_free(self.ptr);
            }
        }
    }
}

pub struct ItqMatrix {
    ptr: *mut FaissITQMatrix,
}

impl ItqMatrix {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissITQMatrix = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_ITQMatrix_new_with(&mut ptr, d as i32))?;
            Ok(Self { ptr })
        }
    }
}

impl VectorTransform for ItqMatrix {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl LinearTransform for ItqMatrix {
    fn inner_linear_ptr(&self) -> *mut FaissLinearTransform {
        self.ptr as *mut FaissLinearTransform
    }
}

impl Drop for ItqMatrix {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_ITQMatrix_free(self.ptr);
            }
        }
    }
}

pub struct ItqTransform {
    ptr: *mut FaissITQTransform,
}

impl ItqTransform {
    pub fn new(d_in: u32, d_out: u32, do_pca: bool) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissITQTransform = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_ITQTransform_new_with(
                &mut ptr,
                d_in as i32,
                d_out as i32,
                do_pca as i32,
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn do_pca(&self) -> bool {
        unsafe { faiss_next_sys::faiss_ITQTransform_do_pca(self.ptr) != 0 }
    }
}

impl VectorTransform for ItqTransform {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl Drop for ItqTransform {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_ITQTransform_free(self.ptr);
            }
        }
    }
}

pub struct OpqMatrix {
    ptr: *mut FaissOPQMatrix,
}

impl OpqMatrix {
    pub fn new(d_in: u32, d_out: u32, m: u32) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissOPQMatrix = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_OPQMatrix_new_with(
                &mut ptr,
                d_in as i32,
                d_out as i32,
                m as i32,
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn verbose(&self) -> bool {
        unsafe { faiss_next_sys::faiss_OPQMatrix_verbose(self.ptr) != 0 }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        unsafe { faiss_next_sys::faiss_OPQMatrix_set_verbose(self.ptr, verbose as i32) }
    }

    pub fn niter(&self) -> i32 {
        unsafe { faiss_next_sys::faiss_OPQMatrix_niter(self.ptr) }
    }

    pub fn set_niter(&mut self, niter: i32) {
        unsafe { faiss_next_sys::faiss_OPQMatrix_set_niter(self.ptr, niter) }
    }

    pub fn niter_pq(&self) -> i32 {
        unsafe { faiss_next_sys::faiss_OPQMatrix_niter_pq(self.ptr) }
    }

    pub fn set_niter_pq(&mut self, niter: i32) {
        unsafe { faiss_next_sys::faiss_OPQMatrix_set_niter_pq(self.ptr, niter) }
    }
}

impl VectorTransform for OpqMatrix {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl LinearTransform for OpqMatrix {
    fn inner_linear_ptr(&self) -> *mut FaissLinearTransform {
        self.ptr as *mut FaissLinearTransform
    }
}

impl Drop for OpqMatrix {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_OPQMatrix_free(self.ptr);
            }
        }
    }
}

pub struct NormalizationTransform {
    ptr: *mut FaissNormalizationTransform,
}

impl NormalizationTransform {
    pub fn new(d: u32, norm: f32) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissNormalizationTransform = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_NormalizationTransform_new_with(
                &mut ptr, d as i32, norm,
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn norm(&self) -> f32 {
        unsafe { faiss_next_sys::faiss_NormalizationTransform_norm(self.ptr) }
    }
}

impl VectorTransform for NormalizationTransform {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl Drop for NormalizationTransform {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_NormalizationTransform_free(self.ptr);
            }
        }
    }
}

pub struct CenteringTransform {
    ptr: *mut FaissCenteringTransform,
}

impl CenteringTransform {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissCenteringTransform = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_CenteringTransform_new_with(
                &mut ptr, d as i32,
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl VectorTransform for CenteringTransform {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl Drop for CenteringTransform {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_CenteringTransform_free(self.ptr);
            }
        }
    }
}

pub struct RemapDimensionsTransform {
    ptr: *mut FaissRemapDimensionsTransform,
}

impl RemapDimensionsTransform {
    pub fn new(d_in: u32, d_out: u32, uniform: bool) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissRemapDimensionsTransform = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_RemapDimensionsTransform_new_with(
                &mut ptr,
                d_in as i32,
                d_out as i32,
                uniform as i32,
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl VectorTransform for RemapDimensionsTransform {
    fn inner_ptr(&self) -> *mut FaissVectorTransform {
        self.ptr as *mut FaissVectorTransform
    }
}

impl Drop for RemapDimensionsTransform {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_RemapDimensionsTransform_free(self.ptr);
            }
        }
    }
}
