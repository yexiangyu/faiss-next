#[cxx::bridge]
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/cpp/distance_computer.hpp");
        unsafe fn distance_computer_free(ptr: *mut i32);
        unsafe fn distance_computer_set_query(ptr: *mut i32, query: *const f32);
        unsafe fn distance_computer_compute(ptr: *mut i32, i: i64) -> f32;
        unsafe fn distance_computer_symmetric_dis(ptr: *mut i32, i: i64, j: i64) -> f32;
        unsafe fn flat_codes_distance_computer_distance_to_code(
            ptr: *mut i32,
            code: *const u8,
        ) -> f32;
    }
}

pub type DistanceComputerPtr = *mut i32;

pub trait DistanceComputerTrait {
    fn ptr(&self) -> DistanceComputerPtr;
    fn set_query(&mut self, query: &[f32]) {
        unsafe {
            ffi::distance_computer_set_query(self.ptr(), query.as_ptr());
        }
    }
    fn compute(&self, i: i64) -> f32 {
        unsafe { ffi::distance_computer_compute(self.ptr(), i) }
    }

    fn symmetric_dis(&self, i: i64, j: i64) -> f32 {
        unsafe { ffi::distance_computer_symmetric_dis(self.ptr(), i, j) }
    }
}

pub trait FlatCodesDistanceComputerTrait: DistanceComputerTrait {
    fn distance_to_code(&self, code: &[u8]) -> f32 {
        unsafe { ffi::flat_codes_distance_computer_distance_to_code(self.ptr(), code.as_ptr()) }
    }
}
