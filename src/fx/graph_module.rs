use std::{
    collections::HashMap,
    fmt::{self, Error},
    ops::Deref,
    slice,
};

use pyo3::{
    types::{PyBool, PyDict},
    AsPyPointer, FromPyObject, IntoPy, Py, PyAny, PyNativeType, PyObject, PyResult, PyTypeInfo,
    Python, ToPyObject,
};

use crate::fx::graph::Graph;

/// A wrapper for PyTorch's [`GraphModule`][graphmodule] class.
///
/// The constructor method of this returns a shared reference `&GraphModule`
/// instead of an owned value. The return value is GIL-bound owning
/// reference into Python's heap.
///
/// [graphmodule]: https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule
#[repr(transparent)]
pub struct GraphModule(PyAny);

unsafe impl PyNativeType for GraphModule {}

impl ToPyObject for GraphModule {
    #[inline]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl AsRef<PyAny> for GraphModule {
    #[inline]
    fn as_ref(&self) -> &PyAny {
        &self.0
    }
}

impl Deref for GraphModule {
    type Target = PyAny;

    #[inline]
    fn deref(&self) -> &PyAny {
        &self.0
    }
}

impl AsPyPointer for GraphModule {
    #[inline]
    fn as_ptr(&self) -> *mut pyo3::ffi::PyObject {
        self.0.as_ptr()
    }
}

impl IntoPy<Py<GraphModule>> for &'_ GraphModule {
    #[inline]
    fn into_py(self, py: Python<'_>) -> Py<GraphModule> {
        unsafe { Py::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl From<&'_ GraphModule> for Py<GraphModule> {
    #[inline]
    fn from(value: &'_ GraphModule) -> Self {
        unsafe { Py::from_borrowed_ptr(value.py(), value.as_ptr()) }
    }
}

impl<'a> From<&'a GraphModule> for &'a PyAny {
    #[inline]
    fn from(value: &'a GraphModule) -> Self {
        unsafe { &*(value as *const GraphModule as *const PyAny) }
    }
}

unsafe impl PyTypeInfo for GraphModule {
    type AsRefTarget = Self;

    const NAME: &'static str = "GraphModule";
    const MODULE: Option<&'static str> = Some("torch.fx");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        PyAny::type_object_raw(py)
    }
}

impl<'py> FromPyObject<'py> for &'py GraphModule {
    #[inline]
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        ob.downcast().map_err(Into::into)
    }
}

impl GraphModule {
    /// Create new instance of `GraphModule` PyTorch class with PyTorch native constructor
    /// but `class_name` is not given (so that it remains as the default
    /// value `'GraphModule'`).
    ///
    /// If new instance is created succesfully, returns `Ok` with a shared reference
    /// to the newly created instance in it. Otherwise, returns `Err` with a `PyErr`
    /// in it. The `PyErr` will explain the cause of the failure.
    pub fn new<'py>(py: Python<'py>, nn: &GraphModule, graph: &Graph) -> PyResult<&'py Self> {
        let gm_module = py.import("torch.fx")?.getattr("GraphModule")?;
        let gm = gm_module.getattr("__new__")?.call1((gm_module,))?;
        gm_module.getattr("__init__")?.call1((gm, nn, graph))?;
        Ok(gm.downcast()?)
    }

    /// Create new instane of `GraphModule` PyTorch class with PyTorch native constructor
    /// but `class_name` is not given (so that it remains as the default
    /// value `'GraphModule'`)
    /// and `root` is a newly created `torch.nn.Module` by `torch.nn.Module()`.
    ///
    /// If new instance is created succesfully, returns `Ok` with a shared reference
    /// to the newly created instance in it. Otherwise, returns `Err` with a `PyErr`
    /// in it. The `PyErr` will explain the cause of the failure.
    pub fn new_with_empty_gm<'py>(py: Python<'py>, graph: &Graph) -> PyResult<&'py Self> {
        let nn_module = py.import("torch.nn")?.getattr("Module")?;
        let nn = nn_module.getattr("__new__")?.call1((nn_module,))?;
        nn_module.getattr("__init__")?.call1((nn,))?;
        let gm_module = py.import("torch.fx")?.getattr("GraphModule")?;
        let gm = gm_module.getattr("__new__")?.call1((gm_module,))?;
        gm_module.getattr("__init__")?.call1((gm, nn, graph))?;
        Ok(gm.downcast()?)
    }

    /// Collect all parameters as zero‑copy views with lifetimes tied to the GIL.
    ///
    /// Unlike [`extract_parameters`], this returns views which are lifetime‑bound
    /// to the provided `py` token, making it safe to use without pretending a
    /// `'static` lifetime.
    pub fn extract_parameters_view<'py>(
        &'py self,
        _py: Python<'py>,
    ) -> PyResult<HashMap<String, BufferView<'py>>> {
        self.get_parameters_pydict()?
            .into_iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, value_to_view(v)?)))
            .collect()
    }

    /// Collect all buffers as zero‑copy views with lifetimes tied to the GIL.
    ///
    /// Unlike [`extract_buffers`], this returns views which are lifetime‑bound
    /// to the provided `py` token, making it safe to use without pretending a
    /// `'static` lifetime.
    pub fn extract_buffers_view<'py>(
        &'py self,
        _py: Python<'py>,
    ) -> PyResult<HashMap<String, BufferView<'py>>> {
        self.get_buffers_pydict()?
            .into_iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, value_to_view(v)?)))
            .collect()
    }

    /// Iterate parameters as `(name, BufferView)` without materializing a map.
    pub fn iter_parameters_view<'py>(
        &'py self,
        _py: Python<'py>,
    ) -> PyResult<impl Iterator<Item = PyResult<(String, BufferView<'py>)>> + 'py> {
        let dict = self.get_parameters_pydict()?;
        let keys: Vec<String> = dict
            .keys()
            .into_iter()
            .map(|k| k.extract())
            .collect::<PyResult<_>>()?;
        Ok(DictViewIter { dict, keys, idx: 0 })
    }

    /// Iterate buffers as `(name, BufferView)` without materializing a map.
    pub fn iter_buffers_view<'py>(
        &'py self,
        _py: Python<'py>,
    ) -> PyResult<impl Iterator<Item = PyResult<(String, BufferView<'py>)>> + 'py> {
        let dict = self.get_buffers_pydict()?;
        let keys: Vec<String> = dict
            .keys()
            .into_iter()
            .map(|k| k.extract())
            .collect::<PyResult<_>>()?;
        Ok(DictViewIter { dict, keys, idx: 0 })
    }

    /// Retrieve the `graph` attribute of this `GraphModule`.
    ///
    /// If the retrieval is done successfully, returns `Ok` with a shared reference
    /// to the `graph` attribute (`&Graph`) in it.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn graph(&self) -> PyResult<&Graph> {
        Ok(self.getattr("graph")?.downcast()?)
    }

    /// Zero‑copy parameter view whose lifetime is tied to the GIL.
    pub fn get_parameter_view<'py>(
        &'py self,
        _py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<BufferView<'py>>> {
        let found = self.get_parameters_pydict()?.get_item_with_error(name)?;
        found.map(value_to_view).transpose()
    }

    /// Get the number of parameters of this `GraphModule`.
    ///
    /// If a Python error occurs during this procedure,
    /// returns `Err` with a `PyErr` in it.
    /// `PyErr` will explain the error.
    /// Otherwise, returns `Ok` with
    /// the number of parameters of this `GraphModule` in it.
    pub fn count_parameters(&self) -> PyResult<usize> {
        Ok(self.get_parameters_pydict()?.len())
    }

    /// Zero‑copy buffer view whose lifetime is tied to the GIL.
    pub fn get_buffer_view<'py>(
        &'py self,
        _py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<BufferView<'py>>> {
        let found = self.get_buffers_pydict()?.get_item_with_error(name)?;
        found.map(value_to_view).transpose()
    }

    /// Get the number of buffers of this `GraphModule`.
    ///
    /// If a Python error occurs during this procedure,
    /// returns `Err` with a `PyErr` in it.
    /// `PyErr` will explain the error.
    /// Otherwise, returns `Ok` with
    /// the number of parameters of this `GraphModule` in it.
    pub fn count_buffers(&self) -> PyResult<usize> {
        Ok(self.get_buffers_pydict()?.len())
    }

    /// Stringify this `GraphModule`.
    ///
    /// This does the same what `print_readable` instance method of
    /// `GraphModule` PyTorch class does, but `print_output` is given as
    /// `True`.
    ///
    /// If stringifying is done successfully, returns `Ok` with the
    /// resulting string in it. Otherwise, returns `Err` with a
    /// `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn print_readable(&self) -> PyResult<String> {
        let py = self.py();
        self.getattr("print_readable")?
            .call1((PyBool::new(py, false),))?
            .extract()
    }

    #[inline]
    fn get_parameters_pydict(&self) -> PyResult<&PyDict> {
        Ok(self.getattr("_parameters")?.downcast::<PyDict>()?)
    }

    #[inline]
    fn get_buffers_pydict(&self) -> PyResult<&PyDict> {
        Ok(self.getattr("_buffers")?.downcast::<PyDict>()?)
    }
}

impl fmt::Debug for GraphModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.graph().map_err(|_| Error)?)
    }
}

// value_to_slice removed in favor of lifetime‑bound BufferView

/// A zero‑copy, read‑only view into a Python tensor's underlying storage.
///
/// The view is lifetime‑bound to the GIL token used to produce it, ensuring the
/// underlying Python object outlives any slice produced from this view.
pub struct BufferView<'py> {
    owner: Py<PyAny>,
    slice: &'py [u8],
}

impl<'py> Deref for BufferView<'py> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        // Touch owner so the field counts as read and to
        // emphasize the lifetime tie to the Python object.
        let _ = &self.owner;
        self.slice
    }
}

#[inline]
fn value_to_view<'py>(v: &'py PyAny) -> PyResult<BufferView<'py>> {
    let offset = v.getattr("storage_offset")?.call0()?.extract::<usize>()?
        * v.getattr("element_size")?.call0()?.extract::<usize>()?;
    let storage = v.getattr("untyped_storage")?.call0()?;
    let ptr = storage.getattr("data_ptr")?.call0()?.extract::<usize>()? + offset;
    let len = storage.getattr("nbytes")?.call0()?.extract::<usize>()? - offset;
    let py = v.py();
    let owner: Py<PyAny> = v.into_py(py);
    let slice: &'py [u8] = unsafe { slice::from_raw_parts(ptr as *const u8, len) };
    Ok(BufferView { owner, slice })
}

struct DictViewIter<'py> {
    dict: &'py PyDict,
    keys: Vec<String>,
    idx: usize,
}

impl<'py> Iterator for DictViewIter<'py> {
    type Item = PyResult<(String, BufferView<'py>)>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < self.keys.len() {
            let key = self.keys[self.idx].clone();
            self.idx += 1;
            match self.dict.get_item(&key) {
                Some(v) => match value_to_view(v) {
                    Ok(view) => return Some(Ok((key, view))),
                    Err(e) => return Some(Err(e)),
                },
                None => continue,
            }
        }
        None
    }
}
