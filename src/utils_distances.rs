#![allow(non_snake_case)]
use faiss_next_sys as ffi;

/// Compute the square root of L2 distances (wrapper around L2 squared)
pub fn faiss_pairwise_L2sqrt(
    d: i64,
    nq: i64,
    xq: impl AsRef<[f32]>,
    nb: i64,
    xb: impl AsRef<[f32]>,
    mut dis: impl AsMut<[f32]>,
) {
    let xq_len = xq.as_ref().len();
    let xb_len = xb.as_ref().len();
    let dis_len = dis.as_mut().len();
    
    // Verify we have the right amount of data for the inputs
    assert_eq!(xq_len as i64, nq * d);
    assert_eq!(xb_len as i64, nb * d);
    assert_eq!(dis_len as i64, nq * nb);
    
    unsafe {
        ffi::faiss_pairwise_L2sqr_with_defaults(
            d,
            nq,
            xq.as_ref().as_ptr(),
            nb,
            xb.as_ref().as_ptr(),
            dis.as_mut().as_mut_ptr(),
        )
    }
    
    // Take the square root of all distances
    for dist in dis.as_mut() {
        *dist = dist.sqrt();
    }
}

/// Compute pairwise distances between sets of vectors
pub fn faiss_pairwise_L2sqr(
    d: i64,
    nq: i64,
    xq: impl AsRef<[f32]>,
    nb: i64,
    xb: impl AsRef<[f32]>,
    mut dis: impl AsMut<[f32]>,
    ldq: i64,
    ldb: i64,
    ldd: i64,
) {
    let xq_len = xq.as_ref().len();
    let xb_len = xb.as_ref().len();
    let dis_len = dis.as_mut().len();
    
    // Verify we have the right amount of data for the inputs
    assert_eq!(xq_len as i64, if ldq == -1 { nq * d } else { nq * ldq });
    assert_eq!(xb_len as i64, if ldb == -1 { nb * d } else { nb * ldb });
    assert_eq!(dis_len as i64, if ldd == -1 { nq * nb } else { nq * ldd });
    
    unsafe {
        ffi::faiss_pairwise_L2sqr(
            d,
            nq,
            xq.as_ref().as_ptr(),
            nb,
            xb.as_ref().as_ptr(),
            dis.as_mut().as_mut_ptr(),
            ldq,
            ldb,
            ldd,
        )
    }
}

/// Compute pairwise distances between sets of vectors with default parameters
pub fn faiss_pairwise_L2sqr_with_defaults(
    d: i64,
    nq: i64,
    xq: impl AsRef<[f32]>,
    nb: i64,
    xb: impl AsRef<[f32]>,
    mut dis: impl AsMut<[f32]>,
) {
    let xq_len = xq.as_ref().len();
    let xb_len = xb.as_ref().len();
    let dis_len = dis.as_mut().len();
    
    // Verify we have the right amount of data for the inputs
    assert_eq!(xq_len as i64, nq * d);
    assert_eq!(xb_len as i64, nb * d);
    assert_eq!(dis_len as i64, nq * nb);
    
    unsafe {
        ffi::faiss_pairwise_L2sqr_with_defaults(
            d,
            nq,
            xq.as_ref().as_ptr(),
            nb,
            xb.as_ref().as_ptr(),
            dis.as_mut().as_mut_ptr(),
        )
    }
}

/// Compute the inner products between nx vectors x and one y
pub fn faiss_fvec_inner_products_ny(
    mut ip: impl AsMut<[f32]>,
    x: impl AsRef<[f32]>,
    y: impl AsRef<[f32]>,
    d: usize,
    ny: usize,
) {
    let x_len = x.as_ref().len();
    let ip_len = ip.as_mut().len();
    
    // The input vectors x should be ny * d in size
    assert_eq!(x_len, ny * d);
    assert_eq!(y.as_ref().len(), d);
    assert_eq!(ip_len, ny);
    
    unsafe {
        ffi::faiss_fvec_inner_products_ny(
            ip.as_mut().as_mut_ptr(),
            x.as_ref().as_ptr(),
            y.as_ref().as_ptr(),
            d,
            ny,
        )
    }
}

/// Compute ny square L2 distance between x and a set of contiguous y vectors
pub fn faiss_fvec_L2sqr_ny(
    mut dis: impl AsMut<[f32]>,
    x: impl AsRef<[f32]>,
    y: impl AsRef<[f32]>,
    d: usize,
    ny: usize,
) {
    let x_len = x.as_ref().len();
    let y_len = y.as_ref().len();
    let dis_len = dis.as_mut().len();
    
    assert_eq!(x_len, d);
    assert_eq!(y_len, ny * d);
    assert_eq!(dis_len, ny);
    
    unsafe {
        ffi::faiss_fvec_L2sqr_ny(
            dis.as_mut().as_mut_ptr(),
            x.as_ref().as_ptr(),
            y.as_ref().as_ptr(),
            d,
            ny,
        )
    }
}

/// Compute the squared L2 norm of a vector
pub fn faiss_fvec_norm_L2sqr(x: impl AsRef<[f32]>, d: usize) -> f32 {
    let x_len = x.as_ref().len();
    assert_eq!(x_len, d);
    
    unsafe { ffi::faiss_fvec_norm_L2sqr(x.as_ref().as_ptr(), d) }
}

/// Compute the L2 norms for a set of vectors
pub fn faiss_fvec_norms_L2(mut norms: impl AsMut<[f32]>, x: impl AsRef<[f32]>, d: usize, nx: usize) {
    let x_len = x.as_ref().len();
    let norms_len = norms.as_mut().len();
    
    assert_eq!(x_len, nx * d);
    assert_eq!(norms_len, nx);
    
    unsafe {
        ffi::faiss_fvec_norms_L2(
            norms.as_mut().as_mut_ptr(),
            x.as_ref().as_ptr(),
            d,
            nx,
        )
    }
}

/// Compute the squared L2 norms for a set of vectors
pub fn faiss_fvec_norms_L2sqr(mut norms: impl AsMut<[f32]>, x: impl AsRef<[f32]>, d: usize, nx: usize) {
    let x_len = x.as_ref().len();
    let norms_len = norms.as_mut().len();
    
    assert_eq!(x_len, nx * d);
    assert_eq!(norms_len, nx);
    
    unsafe {
        ffi::faiss_fvec_norms_L2sqr(
            norms.as_mut().as_mut_ptr(),
            x.as_ref().as_ptr(),
            d,
            nx,
        )
    }
}

/// L2-renormalize a set of vectors. Nothing done if the vector is 0-normed
pub fn faiss_fvec_renorm_L2(d: usize, nx: usize, mut x: impl AsMut<[f32]>) {
    let x_len = x.as_mut().len();
    assert_eq!(x_len, nx * d);
    
    unsafe {
        ffi::faiss_fvec_renorm_L2(
            d,
            nx,
            x.as_mut().as_mut_ptr(),
        )
    }
}

// Setter/Getter functions for distance computation parameters

/// Setter of threshold value on nx above which we switch to BLAS to compute distances
pub fn faiss_set_distance_compute_blas_threshold(value: i32) {
    unsafe {
        ffi::faiss_set_distance_compute_blas_threshold(value)
    }
}

/// Getter of threshold value on nx above which we switch to BLAS to compute distances
pub fn faiss_get_distance_compute_blas_threshold() -> i32 {
    unsafe { ffi::faiss_get_distance_compute_blas_threshold() }
}

/// Setter of block sizes value for BLAS query distance computations
pub fn faiss_set_distance_compute_blas_query_bs(value: i32) {
    unsafe {
        ffi::faiss_set_distance_compute_blas_query_bs(value)
    }
}

/// Getter of block sizes value for BLAS query distance computations
pub fn faiss_get_distance_compute_blas_query_bs() -> i32 {
    unsafe { ffi::faiss_get_distance_compute_blas_query_bs() }
}

/// Setter of block sizes value for BLAS database distance computations
pub fn faiss_set_distance_compute_blas_database_bs(value: i32) {
    unsafe {
        ffi::faiss_set_distance_compute_blas_database_bs(value)
    }
}

/// Getter of block sizes value for BLAS database distance computations
pub fn faiss_get_distance_compute_blas_database_bs() -> i32 {
    unsafe { ffi::faiss_get_distance_compute_blas_database_bs() }
}

/// Setter of number of results we switch to a reservoir to collect results rather than a heap
pub fn faiss_set_distance_compute_min_k_reservoir(value: i32) {
    unsafe {
        ffi::faiss_set_distance_compute_min_k_reservoir(value)
    }
}

/// Getter of number of results we switch to a reservoir to collect results rather than a heap
pub fn faiss_get_distance_compute_min_k_reservoir() -> i32 {
    unsafe { ffi::faiss_get_distance_compute_min_k_reservoir() }
}