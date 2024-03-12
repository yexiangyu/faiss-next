macro_rules! bridge_cpp {
    ($i: expr) => {{
        let rs = format!("src/{}.rs", $i);
        let cpp = format!("src/cpp/{}.cpp", $i);
        let header = format!("src/cpp/{}.hpp", $i);
        cxx_build::bridge(&rs).file(&cpp).std("c++17").compile($i);
        println!("cargo:rerun-if-changed={}", cpp);
        println!("cargo:rerun-if-changed={}", rs);
        println!("cargo:rerun-if-changed={}", header);
    }};
}

fn main() {
    println!("cargo:rustc-link-search=/Users/yexiangyu/faiss/lib");
    println!("cargo:rustc-link-lib=faiss");

    bridge_cpp!("index_factory");
    bridge_cpp!("index");
    bridge_cpp!("index_binary");
    bridge_cpp!("id_selector");
    bridge_cpp!("distance_computer");
    bridge_cpp!("aux_index_structures");
}
