//! Capture and load an mlx-lm model's forward graph.
//!
//! End-to-end flow:
//!   1. Embed cpython-3.11 (oMLX-bundled) via pyo3 — no subprocess.
//!   2. Call `mlx_lm.load → model.forward → mx.export_function(callback, ...)`
//!      with a Rust closure as the callback. Each primitive's PyDict
//!      lands straight in a [`RawGraph`].
//!   3. [`lower`] converts the [`RawGraph`] into the typed [`Graph`] IR.
//!
//! Tensor identity in the records is mlx's short alphabetic name
//! (`"A"`, `"B"`, ...). We dedupe by that name: the same name across
//! records refers to the same logical tensor.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::graph::{Attrs, Dtype, Graph, ScalarValue, TensorId};
use crate::paths;

/// Capture the model's forward graph by embedding cpython-3.11 in-process
/// (no subprocess, no JSON spill). Calls `mlx_lm.load → model.forward
/// → mx.export_function(callback, ...)` and lets a Rust closure unpack
/// each primitive's PyDict straight into a [`RawGraph`]. Keeps the same
/// `RawRecord` shape the JSON path produced so [`lower`] doesn't have
/// to know which capture method ran.
pub fn capture(model_dir: &Path) -> Result<RawGraph> {
    capture_with_seq(model_dir, 8)
}

pub fn capture_with_seq(model_dir: &Path, seq_len: u32) -> Result<RawGraph> {
    use pyo3::prelude::*;
    use pyo3::types::{
        PyAnyMethods, PyCFunction, PyDict, PyDictMethods, PyList, PyListMethods, PyTuple,
    };
    use std::sync::{Arc, Mutex};

    // Capture must always see raw mlx-lm. During `patch` we may emit
    // the base artifact before capturing larger prefill variants; the
    // installed .pth hook would otherwise wrap the model and pollute
    // the exported graph.
    let old_disable_hook = std::env::var_os("FUSEKIT_DISABLE_HOOK");
    std::env::set_var("FUSEKIT_DISABLE_HOOK", "1");

    // The bundled cpython-3.11 was built with `/install/...` as its
    // placeholder install prefix. PYTHONHOME has to be steered at the
    // real bundle root before init or stdlib import fails.
    if std::env::var_os("PYTHONHOME").is_none() {
        let app = paths::mlx_app_root()?;
        let home = app.join("Contents/Python/cpython-3.11");
        std::env::set_var("PYTHONHOME", &home);
    }
    pyo3::prepare_freethreaded_python();

    let site_packages = paths::mlx_site_packages()?;
    let model_dir_str = model_dir
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 model_dir: {}", model_dir.display()))?
        .to_owned();
    let allow_extra_shared_kv = gemma4_has_extra_shared_kv_weights(model_dir);

    let raw = Python::with_gil(|py| -> Result<RawGraph> {
        // Inject the oMLX site-packages so `import mlx_lm` resolves.
        let sys = py.import_bound("sys").map_err(map_pyerr)?;
        let path: Bound<PyList> = sys
            .getattr("path")
            .map_err(map_pyerr)?
            .downcast_into()
            .map_err(|_| anyhow!("sys.path is not a list"))?;
        path.insert(0, site_packages.to_str().unwrap())
            .map_err(map_pyerr)?;

        // Load the model. Gemma4 E*B exports can include K/V tensors for
        // layers that the config marks as KV-shared. The model semantics are
        // the shared-KV config; the extra tensors are harmless checkpoint
        // baggage, so capture uses non-strict loading only for that verified
        // config/weight shape.
        let model = if allow_extra_shared_kv {
            let pathlib = py.import_bound("pathlib").map_err(map_pyerr)?;
            let model_path = pathlib
                .getattr("Path")
                .map_err(map_pyerr)?
                .call1((model_dir_str.as_str(),))
                .map_err(map_pyerr)?;
            let utils = py.import_bound("mlx_lm.utils").map_err(map_pyerr)?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("strict", false).map_err(map_pyerr)?;
            let load_result = utils
                .getattr("load_model")
                .map_err(map_pyerr)?
                .call((model_path,), Some(&kwargs))
                .map_err(map_pyerr)?;
            load_result.get_item(0).map_err(map_pyerr)?
        } else {
            let mlx_lm = py.import_bound("mlx_lm").map_err(map_pyerr)?;
            let load_result = mlx_lm
                .getattr("load")
                .map_err(map_pyerr)?
                .call1((model_dir_str.as_str(),))
                .map_err(map_pyerr)?;
            load_result.get_item(0).map_err(map_pyerr)?
        };

        // Multi-token prefill dummy. Decode-time patterns
        // (`SdpaDecode`, `RopeKvCacheStoreDecode`, …) gate on the
        // structural shape of attention + RoPE + cache I/O, which
        // mlx-lm only emits when the prompt is at least a couple of
        // tokens long. L=1 would collapse SDPA scores into a 1×1
        // matrix that the matcher doesn't recognise as decode.
        let mx = py.import_bound("mlx.core").map_err(map_pyerr)?;
        let int32 = mx.getattr("int32").map_err(map_pyerr)?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", int32).map_err(map_pyerr)?;
        let shape = (1i64, seq_len.max(1) as i64);
        let dummy = mx
            .getattr("zeros")
            .map_err(map_pyerr)?
            .call((shape,), Some(&kwargs))
            .map_err(map_pyerr)?;

        // Trace WITHOUT a KV cache. The patterns we match
        // (`RmsNormQkv`, `SdpaDecode`, `RopeKvCacheStoreDecode`, …)
        // were designed to recognize the cache-less compute pattern
        // and emit kernels that READ/WRITE the cache via runtime
        // uniform buffers and pointer slots — so cache state lives
        // entirely in the runtime, never in the captured graph.
        let forward = model.clone();

        // Records accumulator. PyCFunction::new_closure_bound requires
        // `Send`, hence Arc<Mutex<…>> over the naive Rc<RefCell<…>>.
        let records: Arc<Mutex<Vec<RawRecord>>> = Arc::new(Mutex::new(Vec::new()));
        // Trace-time constants come through as `(trace_name, mx.array)`.
        // We grab each array's DLPack data pointer here so we can match
        // it back to a model parameter (which we walk separately
        // below). This lets task_graph.json's `external` bindings
        // carry the actual Python attribute path (e.g.
        // `model.layers.3.self_attn.q_proj.weight`) instead of a
        // bare TensorId.
        let const_ptrs: Arc<Mutex<Vec<(String, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let scalar_consts: Arc<Mutex<Vec<(String, ScalarValue)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let records_cb = records.clone();
        let const_ptrs_cb = const_ptrs.clone();
        let scalar_consts_cb = scalar_consts.clone();

        let cb = PyCFunction::new_closure_bound(
            py,
            None,
            None,
            move |args: &Bound<PyTuple>, _kwargs: Option<&Bound<PyDict>>| -> PyResult<Py<PyAny>> {
                let py = args.py();
                let info: Bound<PyDict> = args.get_item(0)?.downcast_into()?;
                let kind: String = info
                    .get_item("type")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing `type`"))?
                    .extract()?;
                if kind == "constants" {
                    if let Some(val) = info.get_item("constants")? {
                        let lst: Bound<PyList> = val.downcast_into()?;
                        for item in lst.iter() {
                            let tup: Bound<PyTuple> = item.downcast_into()?;
                            if tup.len() < 2 {
                                continue;
                            }
                            let name: String = tup.get_item(0)?.extract()?;
                            let arr = tup.get_item(1)?;
                            if let Ok(ptr) = array_data_ptr(&arr) {
                                const_ptrs_cb.lock().unwrap().push((name.clone(), ptr));
                            }
                            // Capture scalar values too. Without this
                            // the runtime can't reproduce mlx-internal
                            // scalar constants (axis indices,
                            // broadcast dims, scale factors, …) —
                            // they'd come through `task_graph.json`
                            // as bare TensorIds with no live source.
                            if let Ok(shape) =
                                arr.getattr("shape").and_then(|s| s.extract::<Vec<i64>>())
                            {
                                if shape.is_empty() {
                                    if let Some(v) = extract_scalar_value(&arr) {
                                        scalar_consts_cb.lock().unwrap().push((name, v));
                                    }
                                }
                            }
                        }
                    }
                }
                let rec = build_raw_record(&info)?;
                records_cb.lock().unwrap().push(rec);
                Ok(py.None())
            },
        )
        .map_err(map_pyerr)?;

        mx.getattr("export_function")
            .map_err(map_pyerr)?
            .call1((cb, forward.clone(), dummy))
            .map_err(map_pyerr)?;

        // Walk model parameters via mlx.utils.tree_flatten so each
        // external trace constant resolves to a real Python attribute
        // path. We also capture each parameter's DLPack data pointer
        // and join the two tables to produce `trace_name → path`.
        let (params, ptr_to_path) =
            collect_model_params(py, &model).unwrap_or_else(|_| (Vec::new(), HashMap::new()));
        let const_ptrs = std::sync::Arc::try_unwrap(const_ptrs)
            .map_err(|_| anyhow!("const_ptrs Arc still has outstanding clones"))?
            .into_inner()
            .map_err(|_| anyhow!("const_ptrs mutex poisoned"))?;
        let mut name_to_path: HashMap<String, String> = HashMap::new();
        for (name, ptr) in const_ptrs {
            if let Some(path) = ptr_to_path.get(&ptr) {
                name_to_path.insert(name, path.clone());
            }
        }
        let scalar_consts = std::sync::Arc::try_unwrap(scalar_consts)
            .map_err(|_| anyhow!("scalar_consts Arc still has outstanding clones"))?
            .into_inner()
            .map_err(|_| anyhow!("scalar_consts mutex poisoned"))?;
        let name_to_scalar: HashMap<String, ScalarValue> = scalar_consts.into_iter().collect();

        let records = std::sync::Arc::try_unwrap(records)
            .map_err(|_| anyhow!("records Arc still has outstanding clones"))?
            .into_inner()
            .map_err(|_| anyhow!("records mutex poisoned"))?;
        Ok(RawGraph {
            records,
            params,
            name_to_path,
            name_to_scalar,
            model_dir: model_dir_str.clone(),
            config_summary: serde_json::Value::Null,
        })
    });
    match old_disable_hook {
        Some(value) => std::env::set_var("FUSEKIT_DISABLE_HOOK", value),
        None => std::env::remove_var("FUSEKIT_DISABLE_HOOK"),
    }
    raw
}

fn gemma4_has_extra_shared_kv_weights(model_dir: &Path) -> bool {
    let Ok(config_text) = std::fs::read_to_string(model_dir.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_text) else {
        return false;
    };
    let text_config = config
        .get("text_config")
        .unwrap_or(&serde_json::Value::Null);
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let text_model_type = text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if model_type != "gemma4" && text_model_type != "gemma4_text" {
        return false;
    }
    let Some(num_layers) = text_config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
    else {
        return false;
    };
    let Some(shared_layers) = text_config
        .get("num_kv_shared_layers")
        .and_then(|v| v.as_u64())
    else {
        return false;
    };
    if num_layers == 0 || shared_layers == 0 || shared_layers > num_layers {
        return false;
    }
    let first_shared = num_layers - shared_layers;
    let Ok(index_text) = std::fs::read_to_string(model_dir.join("model.safetensors.index.json"))
    else {
        return false;
    };
    let Ok(index) = serde_json::from_str::<serde_json::Value>(&index_text) else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) else {
        return false;
    };
    let layer_marker = format!(".layers.{first_shared}.self_attn.");
    weight_map.keys().any(|key| {
        key.contains(&layer_marker)
            && (key.contains(".k_proj.") || key.contains(".v_proj.") || key.contains(".k_norm."))
    })
}

fn extract_scalar_value(arr: &pyo3::Bound<'_, pyo3::PyAny>) -> Option<ScalarValue> {
    use pyo3::types::PyAnyMethods;
    let v = arr.call_method0("item").ok()?;
    if let Ok(b) = v.extract::<bool>() {
        // bool first because Python booleans are ints — checking
        // them after would misclassify.
        return Some(ScalarValue::Bool(b));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Some(ScalarValue::Int(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Some(ScalarValue::Float(f));
    }
    None
}

/// Extract the underlying data pointer from an `mx.array`'s DLPack
/// capsule. The DLPack ABI puts a `DLManagedTensor` at the capsule
/// pointer, and the first field of its inner `DLTensor` is the
/// `data` pointer — so we just dereference the capsule pointer as a
/// `*const u64`. Returns 0 on any error so the caller can treat the
/// extraction as best-effort.
fn dlpack_data_ptr(arr: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<u64> {
    use pyo3::prelude::*;
    use pyo3::types::PyCapsule;

    let cap_any = arr.call_method0("__dlpack__")?;
    let cap: Bound<PyCapsule> = cap_any.downcast_into().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("__dlpack__ did not return a capsule")
    })?;
    let raw = cap.pointer();
    if raw.is_null() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dlpack capsule pointer was null",
        ));
    }
    // SAFETY: DLPack guarantees `*raw == DLManagedTensor`, whose
    // first field is a `DLTensor`, whose first field is `void* data`.
    // Reading 8 bytes at the capsule pointer is well-defined.
    let data_ptr = unsafe { *(raw as *const u64) };
    Ok(data_ptr)
}

/// Like [`dlpack_data_ptr`] but tolerates mlx dtypes that DLPack
/// refuses to export (notably `bfloat16` — mlx routes it through
/// numpy interop and numpy can't represent it). For those we
/// reinterpret the array as `uint16` (same width as bf16); mlx's
/// `view` is a zero-copy aliasing op, so the underlying data pointer
/// is preserved and DLPack succeeds on the uint16 view. Anything
/// else (`int8`, future narrow types) falls through to the same
/// width-based fallback path: float8/uint8 → uint8, etc.
fn array_data_ptr(arr: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<u64> {
    use pyo3::prelude::*;
    if let Ok(ptr) = dlpack_data_ptr(arr) {
        return Ok(ptr);
    }
    let py = arr.py();
    let mx_core = py.import_bound("mlx.core")?;
    // Pick a same-width DLPack-friendly dtype based on item size. mlx
    // currently only blocks bf16 on this path but the lookup keeps the
    // fallback usable for any future narrow type that mlx adds.
    let itemsize: i64 = arr.getattr("itemsize")?.extract()?;
    let alias_name = match itemsize {
        1 => "uint8",
        2 => "uint16",
        4 => "uint32",
        8 => "uint64",
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array_data_ptr: unsupported itemsize {itemsize} for DLPack fallback"
            )))
        }
    };
    let alias_dtype = mx_core.getattr(alias_name)?;
    let viewed = arr.call_method1("view", (alias_dtype,))?;
    dlpack_data_ptr(&viewed)
}

fn map_pyerr(e: pyo3::PyErr) -> anyhow::Error {
    anyhow!("python: {e}")
}

/// Walk the model with `mlx.utils.tree_flatten` and collect each
/// parameter's dotted path, shape, dtype, and DLPack data pointer.
/// Returns the parameter list plus an inverse `data_ptr → path` map.
/// The data pointer is what the trace-capture callback joins against
/// to resolve a constant's `trace_name → path`.
fn collect_model_params(
    py: pyo3::Python<'_>,
    model: &pyo3::Bound<'_, pyo3::PyAny>,
) -> pyo3::PyResult<(Vec<RawParam>, HashMap<u64, String>)> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyList, PyTuple};

    let tree = py.import_bound("mlx.utils")?;
    let flatten = tree.getattr("tree_flatten")?;
    let params_dict = model.call_method0("parameters")?;
    let flat = flatten.call1((params_dict,))?;
    let lst: Bound<PyList> = flat.downcast_into().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("tree_flatten did not return a list")
    })?;

    let mut params = Vec::with_capacity(lst.len());
    let mut ptr_to_path: HashMap<u64, String> = HashMap::new();
    for entry in lst.iter() {
        let tup: Bound<PyTuple> = entry.downcast_into()?;
        if tup.len() < 2 {
            continue;
        }
        let path: String = tup.get_item(0)?.extract()?;
        let arr = tup.get_item(1)?;
        let shape: Vec<i32> = arr
            .getattr("shape")?
            .extract::<Vec<i64>>()?
            .into_iter()
            .map(|x| x as i32)
            .collect();
        let dtype = format!("{}", arr.getattr("dtype")?);
        let dtype = dtype.rsplit('.').next().unwrap_or(&dtype).to_string();
        if let Ok(ptr) = array_data_ptr(&arr) {
            ptr_to_path.insert(ptr, path.clone());
        }
        params.push(RawParam { path, shape, dtype });
    }
    Ok((params, ptr_to_path))
}

/// Translate one `mx.export_function` callback dict into a [`RawRecord`].
/// Mirrors the JSON shape produced by the legacy `dump_graph.py` driver
/// so [`lower`] doesn't need to branch on capture mode.
fn build_raw_record(info: &pyo3::Bound<'_, pyo3::types::PyDict>) -> pyo3::PyResult<RawRecord> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDictMethods, PyList, PyListMethods, PyTuple};

    let kind: String = info
        .get_item("type")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing `type`"))?
        .extract()?;

    let mut rec = RawRecord {
        kind: kind.clone(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        constants: Vec::new(),
        name: None,
        arguments: Vec::new(),
    };

    let pull_tensor_list = |key: &str| -> PyResult<Vec<RawTensor>> {
        let Some(val) = info.get_item(key)? else {
            return Ok(vec![]);
        };
        let lst: Bound<PyList> = val.downcast_into()?;
        let mut out = Vec::with_capacity(lst.len());
        for item in lst.iter() {
            let tup: Bound<PyTuple> = item.downcast_into()?;
            if tup.len() < 3 {
                continue;
            }
            let name: String = tup.get_item(0)?.extract()?;
            let shape: Vec<i32> = tup
                .get_item(1)?
                .extract::<Vec<i64>>()?
                .into_iter()
                .map(|x| x as i32)
                .collect();
            // mlx ships dtype as an mx.Dtype object; its repr looks like
            // `mlx.core.float32`. We only need the short suffix.
            let dtype = format!("{}", tup.get_item(2)?);
            let dtype = dtype.rsplit('.').next().unwrap_or(&dtype).to_string();
            out.push(RawTensor { name, shape, dtype });
        }
        Ok(out)
    };

    match kind.as_str() {
        "inputs" => {
            rec.inputs = pull_tensor_list("inputs")?;
        }
        "outputs" => {
            rec.outputs = pull_tensor_list("outputs")?;
        }
        "constants" => {
            rec.constants = pull_tensor_list("constants")?;
        }
        "primitive" => {
            rec.inputs = pull_tensor_list("inputs")?;
            rec.outputs = pull_tensor_list("outputs")?;
            rec.name = Some(
                info.get_item("name")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err("primitive missing `name`")
                    })?
                    .extract()?,
            );
            if let Some(args) = info.get_item("arguments")? {
                let lst: Bound<PyList> = args.downcast_into()?;
                for arg in lst.iter() {
                    rec.arguments.push(py_to_json(&arg)?);
                }
            }
        }
        "keyword_inputs" => {
            // mlx-lm models don't take kwargs at forward — keep an empty
            // record so the lowerer sees the same shape it always did.
        }
        _ => {}
    }
    Ok(rec)
}

/// Convert a generic Python value into our JSON representation. Same set
/// of primitives the old `_arg_to_json` Python helper accepted; we
/// re-flatten everything to leaf scalars / strings / nested arrays.
fn py_to_json(v: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<serde_json::Value> {
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyBool, PyFloat, PyInt, PyList, PyString, PyTuple};

    if v.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = v.downcast::<PyBool>() {
        return Ok(serde_json::Value::Bool(b.extract()?));
    }
    if let Ok(i) = v.downcast::<PyInt>() {
        let n: i64 = i.extract()?;
        return Ok(serde_json::Value::Number(n.into()));
    }
    if let Ok(f) = v.downcast::<PyFloat>() {
        let n: f64 = f.extract()?;
        return Ok(serde_json::json!(n));
    }
    if let Ok(s) = v.downcast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }
    if let Ok(lst) = v.downcast::<PyList>() {
        let mut out = Vec::with_capacity(lst.len());
        for item in lst.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(out));
    }
    if let Ok(tup) = v.downcast::<PyTuple>() {
        let mut out = Vec::with_capacity(tup.len());
        for item in tup.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(out));
    }
    // Anything else falls back to `repr()`, matching the Python driver's
    // last-resort string conversion.
    Ok(serde_json::Value::String(format!("{v}")))
}

/// Lower a captured RawGraph into our [`Graph`] IR.
pub fn lower(raw: &RawGraph) -> Result<Graph> {
    let mut g = Graph::new();
    // Name (e.g. "A") -> TensorId. Each name appears multiple times across
    // records but always refers to the same tensor.
    let mut tensors: HashMap<String, TensorId> = HashMap::new();
    let mut next_synth_source: u64 = 1;
    let name_to_path = &raw.name_to_path;
    let name_to_scalar = &raw.name_to_scalar;

    let intern = |g: &mut Graph,
                  tensors: &mut HashMap<String, TensorId>,
                  next_synth: &mut u64,
                  name: &str,
                  shape: &[i32],
                  dtype: Dtype|
     -> TensorId {
        if let Some(&id) = tensors.get(name) {
            return id;
        }
        let synth = *next_synth;
        *next_synth += 1;
        let id = g.add_input(shape.to_vec(), dtype, synth);
        if let Some(path) = name_to_path.get(name) {
            g.tensors[id as usize].param_path = Some(path.clone());
        }
        if let Some(sv) = name_to_scalar.get(name) {
            g.tensors[id as usize].scalar_value = Some(sv.clone());
        }
        tensors.insert(name.to_string(), id);
        id
    };

    let mut declared_inputs: Vec<TensorId> = Vec::new();
    let mut declared_outputs: Vec<TensorId> = Vec::new();

    for record in &raw.records {
        match record.kind.as_str() {
            "inputs" => {
                for t in &record.inputs {
                    let dt = Dtype::from_mlx_name(&t.dtype);
                    let id = intern(
                        &mut g,
                        &mut tensors,
                        &mut next_synth_source,
                        &t.name,
                        &t.shape,
                        dt,
                    );
                    declared_inputs.push(id);
                }
            }
            "constants" => {
                for t in &record.constants {
                    let dt = Dtype::from_mlx_name(&t.dtype);
                    intern(
                        &mut g,
                        &mut tensors,
                        &mut next_synth_source,
                        &t.name,
                        &t.shape,
                        dt,
                    );
                }
            }
            "outputs" => {
                for t in &record.outputs {
                    if let Some(&id) = tensors.get(&t.name) {
                        declared_outputs.push(id);
                    }
                }
            }
            "primitive" => {
                let mut input_ids = Vec::with_capacity(record.inputs.len());
                for t in &record.inputs {
                    let dt = Dtype::from_mlx_name(&t.dtype);
                    let id = intern(
                        &mut g,
                        &mut tensors,
                        &mut next_synth_source,
                        &t.name,
                        &t.shape,
                        dt,
                    );
                    input_ids.push(id);
                }

                let outputs_meta: Vec<(Vec<i32>, Dtype, u64)> = record
                    .outputs
                    .iter()
                    .map(|t| {
                        let dt = Dtype::from_mlx_name(&t.dtype);
                        let src = next_synth_source;
                        next_synth_source += 1;
                        (t.shape.clone(), dt, src)
                    })
                    .collect();

                let name = record.name.clone().unwrap_or_else(|| "?".to_string());
                let attrs = parse_attrs(&name, &record.arguments);
                let (_op_id, output_ids) = g.add_op(name, input_ids, outputs_meta, attrs);

                // Register the freshly-allocated output tensors under their
                // names so subsequent records can refer to them.
                for (t, &id) in record.outputs.iter().zip(output_ids.iter()) {
                    tensors.insert(t.name.clone(), id);
                }
            }
            "keyword_inputs" => {
                // mlx-lm models do not currently take kwargs in forward,
                // but the record exists for completeness.
            }
            other => {
                return Err(anyhow!("unknown record type: {other}"));
            }
        }
    }

    if let Some(&last) = declared_outputs.last() {
        g.set_root(last);
    } else if let Some(last) = g.tensors.last() {
        g.set_root(last.id);
    }

    Ok(g)
}

// --- JSON schema ----------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RawGraph {
    pub records: Vec<RawRecord>,
    /// Model parameters captured via `mlx.utils.tree_flatten`. Ordered
    /// the way `tree_flatten` returns them.
    #[serde(default)]
    pub params: Vec<RawParam>,
    /// `trace_name → param_path` map, joined at capture time by
    /// matching constants' DLPack data pointers against the parameter
    /// list. Lets every external trace tensor resolve directly to a
    /// Python attribute path; the runtime reads these out of
    /// `task_graph.json` and looks the live mlx tensor up by
    /// attribute path.
    #[serde(default)]
    pub name_to_path: HashMap<String, String>,
    /// `trace_name → scalar value` for 0-dim constants the trace
    /// captured (axis indices, broadcast dims, scale factors). Kept
    /// alongside the path map so the runtime can materialize each
    /// scalar into its own uniform buffer instead of guessing the
    /// value.
    #[serde(default, skip)]
    pub name_to_scalar: HashMap<String, ScalarValue>,
    #[serde(default)]
    pub model_dir: String,
    #[serde(default)]
    pub config_summary: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawParam {
    pub path: String,
    pub shape: Vec<i32>,
    pub dtype: String,
}

#[derive(Debug, Deserialize)]
pub struct RawRecord {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub inputs: Vec<RawTensor>,
    #[serde(default)]
    pub outputs: Vec<RawTensor>,
    #[serde(default)]
    pub constants: Vec<RawTensor>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct RawTensor {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: String,
}

// --- helpers --------------------------------------------------------------

/// Convert a primitive's raw JSON argument list into a typed `Attrs`.
///
/// Returns `Attrs::Unknown` for ops we either don't carry parameters for
/// or whose parameters are recoverable from output tensor shapes (e.g.
/// Reshape/Flatten/Squeeze). Reaching codegen with `Unknown` means the
/// op falls back to the string `name` for dispatch — which is fine for
/// nullary scalar ops (Add, Multiply, Sigmoid, ...) but a bug for ops
/// where parameters actually matter (RoPE, RMSNorm, AddMM, Arange).
fn parse_attrs(name: &str, args: &[serde_json::Value]) -> Attrs {
    use serde_json::Value;

    let int_array = |v: &Value| -> Option<Vec<i32>> {
        v.as_array()?
            .iter()
            .map(|x| x.as_i64().map(|n| n as i32))
            .collect()
    };

    match name {
        "RMSNorm" => args
            .first()
            .and_then(Value::as_f64)
            .map(|eps| Attrs::RmsNorm { eps: eps as f32 })
            .unwrap_or(Attrs::Unknown),

        "LayerNorm" => args
            .first()
            .and_then(Value::as_f64)
            .map(|eps| Attrs::LayerNorm { eps: eps as f32 })
            .unwrap_or(Attrs::Unknown),

        "Transpose" => args
            .first()
            .and_then(int_array)
            .map(|axes| Attrs::Transpose { axes })
            .unwrap_or(Attrs::Unknown),

        "Reshape" => args
            .first()
            .and_then(int_array)
            .map(|target_shape| Attrs::Reshape { target_shape })
            .unwrap_or(Attrs::Unknown),

        "Broadcast" => args
            .first()
            .and_then(int_array)
            .map(|target_shape| Attrs::Broadcast { target_shape })
            .unwrap_or(Attrs::Unknown),

        "Slice" => {
            if args.len() >= 3 {
                let start = int_array(&args[0]).unwrap_or_default();
                let stop = int_array(&args[1]).unwrap_or_default();
                let stride = int_array(&args[2]).unwrap_or_default();
                Attrs::Slice {
                    start,
                    stop,
                    stride,
                }
            } else {
                Attrs::Unknown
            }
        }

        "Split" => {
            // mlx Split carries the split axis as its first scalar
            // argument on current mlx-lm traces. If it is absent, the
            // emitter can still infer the axis from input/output shapes.
            let axis = args.first().and_then(Value::as_i64).unwrap_or(-1) as i32;
            Attrs::Split { axis }
        }

        "Concatenate" => {
            let axis = args.first().and_then(Value::as_i64).unwrap_or(-1) as i32;
            Attrs::Concatenate { axis }
        }

        "Softmax" => {
            // mlx Softmax always operates along the last axis; the JSON
            // argument is the `precise` bool.
            Attrs::Softmax { axis: -1 }
        }

        // mlx serializes QuantizedMatmul args as [group_size, bits, ?, transpose].
        // We only need group_size and bits for codegen.
        "QuantizedMatmul" => {
            if args.len() >= 2 {
                let group_size = args[0].as_i64().unwrap_or(64) as u32;
                let bits = args[1].as_i64().unwrap_or(4) as u32;
                Attrs::Quantized { bits, group_size }
            } else {
                Attrs::Quantized {
                    bits: 4,
                    group_size: 64,
                }
            }
        }

        // mlx::fast::CustomKernel::state() returns a 13-element tuple:
        //   [name, source, grid_x, grid_y, grid_z,
        //    tg_x, tg_y, tg_z, shape_infos, ensure_row_contiguous,
        //    scalar_args, is_precompiled, shared_memory]
        // We capture name, source, grid, threadgroup, ensure_row_contig
        // and shared_memory — enough to drop the kernel into our
        // metallib verbatim and dispatch it via ICB at the right
        // (grid, threadgroup) shape.
        "CustomKernel" => {
            if args.len() < 13 {
                return Attrs::Unknown;
            }
            let name = args[0].as_str().unwrap_or("").to_string();
            let source = args[1].as_str().unwrap_or("").to_string();
            let gx = args[2].as_i64().unwrap_or(1).max(1) as u32;
            let gy = args[3].as_i64().unwrap_or(1).max(1) as u32;
            let gz = args[4].as_i64().unwrap_or(1).max(1) as u32;
            let tx = args[5].as_i64().unwrap_or(1).max(1) as u32;
            let ty = args[6].as_i64().unwrap_or(1).max(1) as u32;
            let tz = args[7].as_i64().unwrap_or(1).max(1) as u32;
            let ensure_row_contiguous = args[9].as_bool().unwrap_or(true);
            let shared_memory = args[12].as_i64().unwrap_or(0).max(0) as u32;
            Attrs::CustomKernel {
                name,
                source,
                grid: [gx, gy, gz],
                threadgroup: [tx, ty, tz],
                shared_memory,
                ensure_row_contiguous,
            }
        }

        "RoPE" => {
            if args.len() >= 4 {
                let dims = args[0].as_i64().unwrap_or(0) as i32;
                let traditional = args[1].as_bool().unwrap_or(false);
                let base = args[2].as_f64().unwrap_or(10000.0) as f32;
                let scale = args[3].as_f64().unwrap_or(1.0) as f32;
                Attrs::Rope {
                    dims,
                    traditional,
                    base,
                    scale,
                    inline_freqs: None,
                }
            } else {
                Attrs::Unknown
            }
        }

        "ScaledDotProductAttention" => {
            let scale = args
                .first()
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(0.0);
            Attrs::ScaledDotProductAttention { scale }
        }

        "AddMM" => {
            if args.len() >= 2 {
                let alpha = args[0].as_f64().unwrap_or(1.0) as f32;
                let beta = args[1].as_f64().unwrap_or(1.0) as f32;
                Attrs::AddMm { alpha, beta }
            } else {
                Attrs::Unknown
            }
        }

        "Arange" => {
            if args.len() >= 3 {
                let start = args[0].as_f64().unwrap_or(0.0);
                let stop = args[1].as_f64().unwrap_or(0.0);
                let step = args[2].as_f64().unwrap_or(1.0);
                Attrs::Arange { start, stop, step }
            } else {
                Attrs::Unknown
            }
        }

        "Gather" => {
            if args.len() >= 2 {
                let axes = int_array(&args[0]).unwrap_or_default();
                let slice_sizes = int_array(&args[1]).unwrap_or_default();
                Attrs::Gather { axes, slice_sizes }
            } else {
                Attrs::Unknown
            }
        }

        // Shape-only manipulations (output shape already in IR), elementwise
        // ops, attention, etc. — no parameters worth typing yet.
        _ => Attrs::Unknown,
    }
}
