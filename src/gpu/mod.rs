#[cfg(feature = "cuda")]
mod index;
#[cfg(feature = "cuda")]
mod resources;

#[cfg(feature = "cuda")]
pub use index::GpuIndexImpl;
#[cfg(feature = "cuda")]
pub use resources::GpuResources;
