use faiss_next_sys::{self, FaissDistanceComputer};

use crate::error::{check_return_code, Result};

#[allow(dead_code)]
pub struct DistanceComputer {
    ptr: *mut FaissDistanceComputer,
}

impl DistanceComputer {
    #[allow(dead_code)]
    pub(crate) fn new(ptr: *mut FaissDistanceComputer) -> Self {
        Self { ptr }
    }

    pub fn set_query(&mut self, x: &[f32]) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_DistanceComputer_set_query(self.ptr, x.as_ptr())
        })
    }

    pub fn symmetric_dis(&self, i: i64, j: i64) -> Result<f32> {
        let mut dis = 0.0f32;
        check_return_code(unsafe {
            faiss_next_sys::faiss_DistanceComputer_symmetric_dis(self.ptr, i, j, &mut dis)
        })?;
        Ok(dis)
    }

    pub fn vector_to_query_dis(&self, i: i64) -> Result<f32> {
        let mut dis = 0.0f32;
        check_return_code(unsafe {
            faiss_next_sys::faiss_DistanceComputer_vector_to_query_dis(self.ptr, i, &mut dis)
        })?;
        Ok(dis)
    }
}

impl Drop for DistanceComputer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_DistanceComputer_free(self.ptr);
            }
        }
    }
}
