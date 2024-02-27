use crate::error::{Error, Result};
use crate::index::{FaissMetricType, Index, IndexPtr};
use crate::macros::{define_index_impl, faiss_rc};
use faiss_next_sys as sys;
use std::marker::PhantomData;
use std::ptr::null_mut;

// TODO: Test
define_index_impl!(FaissIndexIVFFlat, faiss_IndexIVFFlat_free);

pub struct FaissQuantizer<'a> {
    pub inner: *mut sys::FaissIndex,
    pub index: std::marker::PhantomData<&'a dyn IndexPtr>,
}

impl Index for FaissQuantizer<'_> {}
impl IndexPtr for FaissQuantizer<'_> {
    fn ptr(&self) -> *const sys::FaissIndex {
        self.inner as *const _
    }

    fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_ptr(self) -> *mut sys::FaissIndex {
        self.inner
    }
}

#[derive(Clone, Debug, Copy)]
#[repr(i8)]
pub enum FaissQuantizerType {
    UseQuantizerAsIndex = 0, //use the quantizer as index in a kmeans training
    PassTrainingSet = 1,     //pass the training set to the quantizer
    TrainingOnIndex = 2,     //train the quantizer on the full dataset
}

impl FaissIndexIVFFlat {
    pub fn downcast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexIVFFlat_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }

    pub fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVFFlat_nlist(self.ptr()) }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVFFlat_nprobe(self.ptr()) }
    }

    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { sys::faiss_IndexIVFFlat_set_nprobe(self.mut_ptr(), nprobe) }
    }

    pub fn quantizer(&self) -> FaissQuantizer {
        let inner = unsafe { sys::faiss_IndexIVFFlat_quantizer(self.ptr()) };
        FaissQuantizer {
            inner,
            index: PhantomData,
        }
    }

    pub fn quantizer_trains_alone(&self) -> FaissQuantizerType {
        match unsafe { sys::faiss_IndexIVFFlat_quantizer_trains_alone(self.ptr()) } {
            0 => FaissQuantizerType::UseQuantizerAsIndex,
            1 => FaissQuantizerType::PassTrainingSet,
            2 => FaissQuantizerType::TrainingOnIndex,
            _ => unimplemented!(),
        }
    }

    //TODO: wtf?
    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexIVFFlat_own_fields(self.ptr()) != 0 }
    }

    //TODO: wtf?
    pub fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexIVFFlat_set_own_fields(self.mut_ptr(), own_fields as i32) }
    }

    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexIVFFlat_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(quantizer: &mut impl IndexPtr, d: usize, nlist: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexIVFFlat_new_with(&mut inner, quantizer.mut_ptr(), d, nlist) })?;
        Ok(Self { inner })
    }

    pub fn new_with_metric(
        quantizer: &mut impl IndexPtr,
        d: usize,
        nlist: usize,
        metric: FaissMetricType,
    ) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_IndexIVFFlat_new_with_metric(
                &mut inner,
                quantizer.mut_ptr(),
                d,
                nlist,
                metric,
            )
        })?;
        Ok(Self { inner })
    }

    //TODO: wtf?
    pub fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let x = x.as_ref();
        let xids = xids.as_ref();
        let precomputed_idx = precomputed_idx.as_ref();
        let n = x.len() as i64 / self.d();
        faiss_rc!({
            sys::faiss_IndexIVFFlat_add_core(
                self.mut_ptr(),
                n,
                x.as_ptr(),
                xids.as_ptr(),
                precomputed_idx.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn update_vectors(&mut self, v: impl AsRef<[f32]>) -> Result<Vec<i64>> {
        let v = v.as_ref();
        let nv = v.len() as i64 / self.d();
        let mut ret = vec![0; nv as usize];
        faiss_rc!({
            sys::faiss_IndexIVFFlat_update_vectors(
                self.mut_ptr(),
                nv as i32,
                ret.as_mut_ptr(),
                v.as_ptr(),
            )
        })?;
        Ok(ret)
    }
}
