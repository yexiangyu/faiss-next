use crate::bindings;

pub trait FaissIndex {
    fn inner(&self) -> *mut bindings::FaissIndex;

    fn d(&self) -> i32 {
        unsafe { bindings::faiss_Index_d(self.inner()) }
    }

    fn ntotal(&self) -> i64 {
        unsafe { bindings::faiss_Index_ntotal(self.inner()) }
    }

    fn is_trained(&self) -> bool {
        unsafe { bindings::faiss_Index_is_trained(self.inner()) != 0 }
    }

    fn metric_type(&self) -> bindings::FaissMetricType {
        unsafe { bindings::faiss_Index_metric_type(self.inner()) }
    }

    fn train(&mut self, n: i64, x: &[f32]) -> crate::error::Result<()>;

    fn add(&mut self, n: i64, x: &[f32]) -> crate::error::Result<()>;

    fn add_with_ids(&mut self, n: i64, x: &[f32], ids: &[i64]) -> crate::error::Result<()>;

    fn search(
        &self,
        n: i64,
        x: &[f32],
        k: i64,
        distances: &mut [f32],
        labels: &mut [i64],
    ) -> crate::error::Result<()>;

    fn range_search(
        &self,
        n: i64,
        x: &[f32],
        radius: f32,
        result: *mut bindings::FaissRangeSearchResult,
    ) -> crate::error::Result<()>;

    fn reset(&mut self) -> crate::error::Result<()>;

    fn reconstruct(&self, key: i64, recons: &mut [f32]) -> crate::error::Result<()>;
}

pub trait FaissIndexBinary {
    fn inner(&self) -> *mut bindings::FaissIndexBinary;

    fn d(&self) -> i32 {
        unsafe { bindings::faiss_IndexBinary_d(self.inner()) }
    }

    fn ntotal(&self) -> i64 {
        unsafe { bindings::faiss_IndexBinary_ntotal(self.inner()) }
    }

    fn is_trained(&self) -> bool {
        unsafe { bindings::faiss_IndexBinary_is_trained(self.inner()) != 0 }
    }

    fn metric_type(&self) -> bindings::FaissMetricType {
        unsafe { bindings::faiss_IndexBinary_metric_type(self.inner()) }
    }

    fn train(&mut self, n: i64, x: &[u8]) -> crate::error::Result<()>;

    fn add(&mut self, n: i64, x: &[u8]) -> crate::error::Result<()>;

    fn search(
        &self,
        n: i64,
        x: &[u8],
        k: i64,
        distances: &mut [i32],
        labels: &mut [i64],
    ) -> crate::error::Result<()>;

    fn reset(&mut self) -> crate::error::Result<()>;
}

pub trait FaissIVFIndex: FaissIndex {
    fn nlist(&self) -> usize;

    fn nprobe(&self) -> usize;

    fn set_nprobe(&mut self, nprobe: usize);

    fn quantizer(&self) -> *mut bindings::FaissIndex;
}

pub trait FaissVectorTransform {
    fn inner(&self) -> *mut bindings::FaissVectorTransform;

    fn is_trained(&self) -> bool {
        unsafe { bindings::faiss_VectorTransform_is_trained(self.inner()) != 0 }
    }

    fn d_in(&self) -> i32 {
        unsafe { bindings::faiss_VectorTransform_d_in(self.inner()) }
    }

    fn d_out(&self) -> i32 {
        unsafe { bindings::faiss_VectorTransform_d_out(self.inner()) }
    }

    fn train(&mut self, n: i64, x: &[f32]) -> crate::error::Result<()> {
        crate::error::check_return_code(unsafe {
            bindings::faiss_VectorTransform_train(self.inner(), n, x.as_ptr())
        })
    }

    fn apply(&self, n: i64, x: &[f32]) -> *mut f32 {
        unsafe { bindings::faiss_VectorTransform_apply(self.inner(), n, x.as_ptr()) }
    }
}
