use faiss_next_sys::FaissMetricType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum MetricType {
    InnerProduct = 0,
    #[default]
    L2 = 1,
    L1 = 2,
    Linf = 3,
    Lp = 4,
    Canberra = 20,
    BrayCurtis = 21,
    JensenShannon = 22,
}

impl MetricType {
    pub fn as_native(self) -> FaissMetricType {
        match self {
            MetricType::InnerProduct => FaissMetricType::METRIC_INNER_PRODUCT,
            MetricType::L2 => FaissMetricType::METRIC_L2,
            MetricType::L1 => FaissMetricType::METRIC_L1,
            MetricType::Linf => FaissMetricType::METRIC_Linf,
            MetricType::Lp => FaissMetricType::METRIC_Lp,
            MetricType::Canberra => FaissMetricType::METRIC_Canberra,
            MetricType::BrayCurtis => FaissMetricType::METRIC_BrayCurtis,
            MetricType::JensenShannon => FaissMetricType::METRIC_JensenShannon,
        }
    }

    pub fn from_native(mt: FaissMetricType) -> Self {
        match mt {
            FaissMetricType::METRIC_INNER_PRODUCT => MetricType::InnerProduct,
            FaissMetricType::METRIC_L2 => MetricType::L2,
            FaissMetricType::METRIC_L1 => MetricType::L1,
            FaissMetricType::METRIC_Linf => MetricType::Linf,
            FaissMetricType::METRIC_Lp => MetricType::Lp,
            FaissMetricType::METRIC_Canberra => MetricType::Canberra,
            FaissMetricType::METRIC_BrayCurtis => MetricType::BrayCurtis,
            FaissMetricType::METRIC_JensenShannon => MetricType::JensenShannon,
            _ => MetricType::L2,
        }
    }
}

impl From<FaissMetricType> for MetricType {
    fn from(mt: FaissMetricType) -> Self {
        Self::from_native(mt)
    }
}

impl From<MetricType> for FaissMetricType {
    fn from(mt: MetricType) -> Self {
        mt.as_native()
    }
}
