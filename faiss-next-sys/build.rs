#![allow(unused)]

use cmake::Config;
use std::{
    env::{set_var, var},
    path::PathBuf,
};

use bindgen::Builder;
use cfg_if::cfg_if;

fn main() {
    println!("cargo:rustc-link-lib=static=faiss_c");
    println!("cargo:rustc-link-lib=static=faiss");

    cfg_if! {
        if #[cfg(feature = "system")]
        {
            faiss_system_setup();
        }
        else
        {
            #[cfg(target_os = "macos")]
            {
                faiss_cmake_macos();
            }

            #[cfg(target_os = "linux")]
            {
                faiss_cmake_linux();
            }

            #[cfg(target_os = "windows")]
            {
                faiss_cmake_windows();
            }
        }


    }

    #[cfg(feature = "bindgen")]
    {
        faiss_bindgen();
    }
}

fn faiss_system_setup() {
    println!("cargo:rerun-if-env-changed=FAISS_DIR");

    let mut dir = var("FAISS_DIR").ok();

    #[cfg(target_os = "windows")]
    {
        if dir.is_none() {
            set_var("FAISS_DIR", "c:\\tools\\faiss");
            dir = Some("c:\\tools\\faiss".to_string());
        }
    }

    if let Some(dir) = dir {
        let dir = PathBuf::from(dir);
        println!("cargo:rustc-link-search={}", dir.join("lib").display());
    }
}

fn faiss_cmake_macos() {
    let dst = Config::new("faiss")
        .define("CMAKE_C_COMPILER", "/opt/homebrew/opt/llvm/bin/clang")
        .define("CMAKE_ASM_COMPILER", "/opt/homebrew/opt/llvm/bin/clang")
        .define("CMAKE_CXX_COMPILER", "/opt/homebrew/opt/llvm/bin/clang++")
        .define("FAISS_ENABLE_GPU", "OFF")
        .define("FAISS_ENABLE_PYTHON", "OFF")
        .define("FAISS_ENABLE_C_API", "ON")
        .define("BUILD_TESTING", "OFF")
        .build();

    let c_api = dst.join("build").join("c_api").join("libfaiss_c.a");

    std::fs::copy(c_api, dst.join("lib").join("libfaiss_c.a"))
        .expect("failed to copy libfaiss_c.a");

    println!("cargo:rustc-link-search={}", dst.join("lib").display());
}

fn faiss_cmake_windows() {
    todo!()
}

fn faiss_cmake_linux() {
    let mut builder = Config::new("faiss");

    builder
        .define("FAISS_ENABLE_PYTHON", "OFF")
        .define("FAISS_ENABLE_C_API", "ON")
        .define("BUILD_TESTING", "OFF");

    cfg_if! {
        if #[cfg(feature = "gpu")]
        {
            builder.define("FAISS_ENABLE_GPU", "ON");
        }
        else
        {
            builder.define("FAISS_ENABLE_GPU", "OFF");
        }
    }

    let dst = builder.build();

    let c_api = dst.join("build").join("c_api").join("libfaiss_c.a");

    std::fs::copy(c_api, dst.join("lib").join("libfaiss_c.a"))
        .expect("failed to copy libfaiss_c.a");

    println!("cargo:rustc-link-search={}", dst.join("lib").display());
}

fn faiss_bindgen() {
    println!("cargo:rerun-if-changed=faiss.h");

    let mut builder = Builder::default()
        .header("faiss.h")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .layout_tests(false)
        .allowlist_function("faiss_.*")
        .allowlist_type("idx_t|Faiss.*")
        .opaque_type("FILE")
        .wrap_unsafe_ops(true);

    #[cfg(feature = "system")]
    {
        if let Ok(faiss_dir) = var("FAISS_DIR") {
            builder = builder.clang_arg(format!("-I{}/include", faiss_dir));
        }
    }

    #[cfg(not(feature = "system"))]
    {
        builder = builder.clang_arg(format!("-I./"));
    }

    let os_name = var("CARGO_CFG_TARGET_OS").expect("no in cargo");
    let output_dir = PathBuf::from("src").join(os_name);
    std::fs::create_dir_all(&output_dir).expect("failed to create output_dir");

    let mut output = output_dir.join("bindings.rs");

    #[cfg(feature = "gpu")]
    {
        output = output_dir.join("bindings_gpu.rs");
        builder = builder
            .clang_arg("-DFAISS_USE_GPU")
            .clang_arg(format!("-I{}", get_cuda_include_dir()));
    }

    builder
        .generate()
        .expect("unable to generate bindings")
        .write_to_file(&mut output)
        .expect("unable to write bindings");
}

fn get_cuda_include_dir() -> String {
    let mut inc_dir = "".to_string();

    cfg_if! {
        if #[cfg(target_os = "linux")]
        {
            inc_dir = "/usr/local/cuda/targets/x86_64-linux/include/".to_string();
        }
        else
        {
        }
    }

    if inc_dir.is_empty() {
        panic!("could not find cuda include dir");
    }

    inc_dir
}
