// build.rs for fusekit_runtime_native.
//
// Two responsibilities:
//   1. macOS linker config so pyo3 extension-module symbols resolve at
//      load time (Py_None etc. come from the host CPython).
//   2. Compile our small `metal/test_kernels.metal` into a metallib
//      so smoke tests can dispatch a known kernel against buffers
//      shared with mlx-side mx.arrays.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    if cfg!(target_os = "macos") {
        // Python extension-module convention: undefined symbols
        // resolve from the host interpreter at dlopen time.
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
    compile_test_metallib();
}

fn compile_test_metallib() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest_dir.join("metal").join("test_kernels.metal");
    println!("cargo:rerun-if-changed={}", src.display());

    let metal_bin = find_xcrun_tool("metal");
    let metallib_bin = find_xcrun_tool("metallib");

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let air = out_dir.join("test_kernels.air");
    let lib = out_dir.join("test_kernels.metallib");

    let status = Command::new(&metal_bin)
        .args(["-c", "-O3"])
        .arg(&src)
        .arg("-o")
        .arg(&air)
        .status()
        .expect("invoke metal compiler");
    assert!(status.success(), "metal compile failed");

    let status = Command::new(&metallib_bin)
        .arg(&air)
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("invoke metallib linker");
    assert!(status.success(), "metallib link failed");

    println!("cargo:rustc-env=FUSEKIT_TEST_METALLIB={}", lib.display());
}

fn find_xcrun_tool(name: &str) -> PathBuf {
    let out = Command::new("xcrun")
        .args(["--sdk", "macosx", "--find", name])
        .output()
        .unwrap_or_else(|_| panic!("xcrun --find {name}"));
    assert!(out.status.success(), "{name} not found via xcrun");
    PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
