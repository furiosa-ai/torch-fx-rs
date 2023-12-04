use std::{collections::HashMap, fmt, ops::Deref, slice};

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

    /// Collect all parameters of this `GraphModule`.
    ///
    /// Make a `HashMap` which maps the parameter name
    /// to a slice representing the underlying storage of the parameter value.
    ///
    /// If this process is done successfully, returns `Ok` with the `HashMap` in it.
    /// Otherwise, return `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn extract_parameters(&self) -> PyResult<HashMap<String, &[u8]>> {
        self.get_parameters_pydict()?
            .into_iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, value_to_slice(v)?)))
            .collect()
    }

    /// Collect all buffers of this `GraphModule`.
    ///
    /// Make a `HashMap` which maps the buffer name
    /// to a slice representing the underlying storage of the buffer value.
    ///
    /// If this process is done successfully, returns `Ok` with the `HashMap` in it.
    /// Otherwise, return `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn extract_buffers(&self) -> PyResult<HashMap<String, &[u8]>> {
        self.get_buffers_pydict()?
            .into_iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, value_to_slice(v)?)))
            .collect()
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

    /// Get the underlying storage of the parameter value named as the value of `name`,
    /// for this `GraphModule`.
    ///
    /// If there is no parameter named as the value of `name`, returns `Ok(None)`.
    /// If there exists such parameter, returns `Ok(Some)` with a slice representing
    /// the underlying storage of the parameter value.
    /// If this process fails, returns `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn get_parameter(&self, name: &str) -> PyResult<Option<&[u8]>> {
        let found = self.get_parameters_pydict()?.get_item_with_error(name)?;
        found.map(value_to_slice).transpose()
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

    /// Get the underlying storage of the buffer value named as the value of `name`,
    /// for this `GraphModule`.
    ///
    /// If there is no buffer named as the value of `name`, returns `Ok(None)`.
    /// If there exists such buffer, returns `Ok(Some)` with a slice representing
    /// the underlying storage of the buffer value.
    /// If this process fails, returns `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn get_buffer(&self, name: &str) -> PyResult<Option<&[u8]>> {
        let found = self.get_buffers_pydict()?.get_item_with_error(name)?;
        found.map(value_to_slice).transpose()
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
        write!(f, "{:?}", self.graph().unwrap())
    }
}

#[inline]
fn value_to_slice(v: &PyAny) -> PyResult<&'static [u8]> {
    let offset = v.getattr("storage_offset")?.call0()?.extract::<usize>()?
        * v.getattr("element_size")?.call0()?.extract::<usize>()?;
    let storage = v.getattr("untyped_storage")?.call0()?;
    let ptr = storage.getattr("data_ptr")?.call0()?.extract::<usize>()? + offset;
    let len = storage.getattr("nbytes")?.call0()?.extract::<usize>()? - offset;
    Ok(unsafe { slice::from_raw_parts(ptr as *const u8, len) })
}
