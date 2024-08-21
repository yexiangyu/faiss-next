#![allow(non_snake_case)]
use faiss_next_sys as ffi;

pub fn faiss_pairewise_L2sqrt(
    d: i64,
    xq: impl AsRef<[f32]>,
    xb: impl AsRef<[f32]>,
    mut distance: impl AsMut<[f32]>,
) {
    assert_eq!(xq.as_ref().len() as i64 % d, 0);
    assert_eq!(xb.as_ref().len() as i64 % d, 0);
    let nq = xq.as_ref().len() as i64 / d as i64;
    let nb = xb.as_ref().len() as i64 / d as i64;
    assert_eq!(distance.as_mut().len() as i64, nb * nq);
    unsafe {
        ffi::faiss_pairwise_L2sqr_with_defaults(
            d,
            nq,
            xq.as_ref().as_ptr(),
            nb,
            xb.as_ref().as_ptr(),
            distance.as_mut().as_mut_ptr(),
        )
    }
}

// TODO: complte "faiss/c_api/utils/distances.h" wrapping
