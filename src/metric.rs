use faiss_next_sys as sys;

#[derive(Debug, smart_default::SmartDefault, Clone, Copy)]
#[repr(u32)]
pub enum MetricType {
    #[doc = "< maximum inner product search"]
    InnerProduct = 0,
    #[doc = "< squared L2 search"]
    L2 = 1,
    #[doc = "< L1 (aka cityblock)"]
    #[default]
    L1 = 2,
    #[doc = "< infinity distance"]
    Linf = 3,
    #[doc = "< L_p distance, p is given by metric_arg"]
    Lp = 4,
    #[doc = " some additional metrics defined in scipy.spatial.distance"]
    Canberra = 20,
    #[doc = " some additional metrics defined in scipy.spatial.distance"]
    BrayCurtis = 21,
    #[doc = " some additional metrics defined in scipy.spatial.distance"]
    JensenShannon = 22,
}

impl From<sys::FaissMetricType> for MetricType {
    fn from(value: sys::FaissMetricType) -> Self {
        match value {
            sys::FaissMetricType::METRIC_INNER_PRODUCT => Self::InnerProduct,
            sys::FaissMetricType::METRIC_L2 => Self::L2,
            sys::FaissMetricType::METRIC_L1 => Self::L1,
            sys::FaissMetricType::METRIC_Linf => Self::Linf,
            sys::FaissMetricType::METRIC_Lp => Self::Lp,
            sys::FaissMetricType::METRIC_Canberra => Self::Canberra,
            sys::FaissMetricType::METRIC_BrayCurtis => Self::BrayCurtis,
            sys::FaissMetricType::METRIC_JensenShannon => Self::JensenShannon,
            _ => panic!("metric type not supported!"),
        }
    }
}

impl From<MetricType> for sys::FaissMetricType {
    fn from(value: MetricType) -> Self {
        match value {
            MetricType::InnerProduct => sys::FaissMetricType::METRIC_INNER_PRODUCT,
            MetricType::L2 => sys::FaissMetricType::METRIC_L2,
            MetricType::L1 => sys::FaissMetricType::METRIC_L1,
            MetricType::Linf => sys::FaissMetricType::METRIC_Linf,
            MetricType::Lp => sys::FaissMetricType::METRIC_Lp,
            MetricType::Canberra => sys::FaissMetricType::METRIC_Canberra,
            MetricType::BrayCurtis => sys::FaissMetricType::METRIC_BrayCurtis,
            MetricType::JensenShannon => sys::FaissMetricType::METRIC_JensenShannon,
        }
    }
}

// pub use sys::FaissMetricType as MetricType;
