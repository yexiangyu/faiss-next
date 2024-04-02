use crate::error::Result;
use crate::macros::rc;
use faiss_next_sys as sys;

pub trait DistanceComputerTrait {
    fn ptr(&self) -> *mut sys::FaissDistanceComputer;

    fn set_query(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        rc!({ sys::faiss_DistanceComputer_set_query(self.ptr(), x.as_ref().as_ptr()) })
    }

    fn vector_to_query_dis(&self, i: i64) -> Result<f32> {
        let mut r = 0.0f32;
        rc!({ sys::faiss_DistanceComputer_vector_to_query_dis(self.ptr(), i, &mut r) })?;
        Ok(r)
    }

    fn symmetric_dis(&self, i: i64, j: i64) -> Result<f32> {
        let mut r = 0.0f32;
        rc!({ sys::faiss_DistanceComputer_symmetric_dis(self.ptr(), i, j, &mut r) })?;
        Ok(r)
    }
}
