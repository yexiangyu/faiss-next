#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to convert {0} to metric type")]
    MetricFromNumber(i32),
}

pub type Result<T> = std::result::Result<T, Error>;
