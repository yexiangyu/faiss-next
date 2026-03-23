use std::ffi::CString;
use std::ptr;

use faiss_next_sys;

use crate::error::{check_return_code, Result};
use crate::index::{Index, IndexImpl};
use crate::metric::MetricType;

pub fn index_factory(d: u32, description: &str, metric: MetricType) -> Result<IndexImpl> {
    let c_description = CString::new(description)?;
    let mut inner = ptr::null_mut();

    unsafe {
        check_return_code(faiss_next_sys::faiss_index_factory(
            &mut inner,
            d as i32,
            c_description.as_ptr(),
            metric.as_native(),
        ))?;

        IndexImpl::from_raw(inner)
    }
}

pub struct IndexBuilder {
    d: u32,
    description: String,
    metric: MetricType,
    nprobe: Option<usize>,
    verbose: bool,
}

impl IndexBuilder {
    pub fn new(d: u32) -> Self {
        Self {
            d,
            description: "Flat".to_string(),
            metric: MetricType::L2,
            nprobe: None,
            verbose: false,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn flat(self) -> Self {
        self.description("Flat")
    }

    pub fn ivf_flat(self, nlist: usize) -> Self {
        self.description(format!("IVF{},Flat", nlist))
    }

    pub fn ivf_pq(self, nlist: usize, m: usize, nbits: usize) -> Self {
        self.description(format!("IVF{},PQ{}x{}", nlist, m, nbits))
    }

    pub fn pq(self, m: usize, nbits: usize) -> Self {
        self.description(format!("PQ{}x{}", m, nbits))
    }

    pub fn hnsw(self, m: usize) -> Self {
        self.description(format!("HNSW{}", m))
    }

    pub fn lsh(self, nbits: usize) -> Self {
        self.description(format!("LSH{}", nbits))
    }

    pub fn metric(mut self, metric: MetricType) -> Self {
        self.metric = metric;
        self
    }

    pub fn l2(self) -> Self {
        self.metric(MetricType::L2)
    }

    pub fn ip(self) -> Self {
        self.metric(MetricType::InnerProduct)
    }

    pub fn nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = Some(nprobe);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn build(self) -> Result<IndexImpl> {
        let mut index = index_factory(self.d, &self.description, self.metric)?;

        if self.verbose {
            index.set_verbose(true);
        }

        Ok(index)
    }
}
