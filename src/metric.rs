use num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(IntoPrimitive, TryFromPrimitive, Debug, Clone, Copy)]
#[repr(i32)]
pub enum MetricType {
    InnerProduct = 0,
    L2 = 1,
    L1,
    Linf,
    Lp,
    Canberra = 20,
    BrayCurtis,
    JensenShannon,
    Jaccard,
}
