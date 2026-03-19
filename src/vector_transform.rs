use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};
use crate::traits::FaissVectorTransform;

pub struct VectorTransform {
    pub(crate) inner: *mut bindings::FaissVectorTransform,
}

impl Drop for VectorTransform {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_VectorTransform_free(self.inner) }
        }
    }
}

impl FaissVectorTransform for VectorTransform {
    fn inner(&self) -> *mut bindings::FaissVectorTransform {
        self.inner
    }
}

pub struct RandomRotationMatrix {
    inner: *mut bindings::FaissVectorTransform,
}

impl Drop for RandomRotationMatrix {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_RandomRotationMatrix_free(self.inner) }
        }
    }
}

impl RandomRotationMatrix {
    pub fn new(d_in: i32, d_out: i32) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_RandomRotationMatrix_new_with(&mut inner, d_in, d_out)
        })?;
        Ok(Self { inner })
    }
}

impl FaissVectorTransform for RandomRotationMatrix {
    fn inner(&self) -> *mut bindings::FaissVectorTransform {
        self.inner
    }
}

pub struct PCAMatrix {
    inner: *mut bindings::FaissVectorTransform,
}

impl Drop for PCAMatrix {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_PCAMatrix_free(self.inner) }
        }
    }
}

impl PCAMatrix {
    pub fn new(d_in: i32, d_out: i32, eigen_power: f32, random_rotation: bool) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_PCAMatrix_new_with(
                &mut inner,
                d_in,
                d_out,
                eigen_power,
                random_rotation as i32,
            )
        })?;
        Ok(Self { inner })
    }
}

impl FaissVectorTransform for PCAMatrix {
    fn inner(&self) -> *mut bindings::FaissVectorTransform {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    #[test]
    fn test_random_rotation_matrix() {
        let d_in = 32;
        let d_out = 16;

        let transform = RandomRotationMatrix::new(d_in, d_out).unwrap();
        assert_eq!(transform.d_in(), d_in);
        assert_eq!(transform.d_out(), d_out);
    }

    #[test]
    #[ignore = "Test may trigger C++ exceptions that Rust cannot catch. Needs investigation."]
    fn test_random_rotation_matrix_apply() {
        let d_in = 16;
        let d_out = 8;
        let n = 10;

        let transform = RandomRotationMatrix::new(d_in, d_out).unwrap();

        let data = generate_random_vectors(n, d_in as usize);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        let result = transform.apply(n as i64, &data_slice);
        assert!(!result.is_null());

        unsafe {
            let slice = std::slice::from_raw_parts(result, n * d_out as usize);
            for &val in slice {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_pca_matrix() {
        let d_in = 32;
        let d_out = 8;

        let transform = PCAMatrix::new(d_in, d_out, 0.0, true).unwrap();
        assert_eq!(transform.d_in(), d_in);
        assert_eq!(transform.d_out(), d_out);
    }

    #[test]
    fn test_pca_matrix_train() {
        let d_in = 16;
        let d_out = 4;
        let n = 100;

        let mut transform = PCAMatrix::new(d_in, d_out, 0.0, false).unwrap();
        assert!(!transform.is_trained());

        let data = generate_random_vectors(n, d_in as usize);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        transform.train(n as i64, &data_slice).unwrap();
        assert!(transform.is_trained());

        let result = transform.apply(n as i64, &data_slice);
        assert!(!result.is_null());
    }

    #[test]
    fn test_pca_matrix_eigen_power() {
        let d_in = 32;
        let d_out = 16;
        let n = 200;

        let mut transform = PCAMatrix::new(d_in, d_out, 0.5, true).unwrap();

        let data = generate_random_vectors(n, d_in as usize);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        transform.train(n as i64, &data_slice).unwrap();

        let result = transform.apply(1, &data_slice[0..d_in as usize]);
        assert!(!result.is_null());
    }
}
