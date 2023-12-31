use std::{
    collections::HashMap,
    fmt::{self, Error},
    ops::Deref,
    sync::Arc,
};

use indexmap::IndexMap;
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyTypeError},
    types::{PyDict, PyIterator, PyList, PyTuple},
    AsPyPointer, FromPyObject, IntoPy, Py, PyAny, PyNativeType, PyObject, PyResult, PyTypeInfo,
    Python, ToPyObject,
};

use crate::fx::{
    custom_fn::{CustomFn, FunctionWrapper},
    node::Node,
    Argument, Op, Target,
};

/// A wrapper for PyTorch's [`Graph`][graph] class.
///
/// The constructor method of this returns a shared reference `&Graph`
/// instead of an owned value. The return value is GIL-bound owning
/// reference into Python's heap.
///
/// [graph]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph
#[repr(transparent)]
pub struct Graph(PyAny);

unsafe impl PyNativeType for Graph {}

impl ToPyObject for Graph {
    #[inline]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl AsRef<PyAny> for Graph {
    #[inline]
    fn as_ref(&self) -> &PyAny {
        &self.0
    }
}

impl Deref for Graph {
    type Target = PyAny;

    #[inline]
    fn deref(&self) -> &PyAny {
        &self.0
    }
}

impl AsPyPointer for Graph {
    #[inline]
    fn as_ptr(&self) -> *mut pyo3::ffi::PyObject {
        self.0.as_ptr()
    }
}

impl IntoPy<Py<Graph>> for &'_ Graph {
    #[inline]
    fn into_py(self, py: Python<'_>) -> Py<Graph> {
        unsafe { Py::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl From<&'_ Graph> for Py<Graph> {
    #[inline]
    fn from(value: &'_ Graph) -> Self {
        unsafe { Py::from_borrowed_ptr(value.py(), value.as_ptr()) }
    }
}

impl<'a> From<&'a Graph> for &'a PyAny {
    #[inline]
    fn from(value: &'a Graph) -> Self {
        unsafe { &*(value as *const Graph as *const PyAny) }
    }
}

unsafe impl PyTypeInfo for Graph {
    type AsRefTarget = Self;

    const NAME: &'static str = "Graph";
    const MODULE: Option<&'static str> = Some("torch.fx");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        PyAny::type_object_raw(py)
    }
}

impl<'py> FromPyObject<'py> for &'py Graph {
    #[inline]
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        ob.downcast().map_err(Into::into)
    }
}

impl Graph {
    /// Create new instance of `Graph` PyTorch class with PyTorch native constructor.
    ///
    /// If new instance is created successfully, returns `Ok` with a shared reference
    /// to the newly created instance in it. Otherwise, returns `Err` with a `PyErr`
    /// in it. The `PyErr` will explain the cause of the failure.
    pub fn new(py: Python<'_>) -> PyResult<&Self> {
        let module = py.import("torch.fx")?.getattr("Graph")?;
        let origin = module.getattr("__new__")?.call1((module,))?;
        module.getattr("__init__")?.call1((origin,))?;
        Ok(origin.downcast()?)
    }

    /// Retrieve all the [`Node`s][nodes] of this `Graph` as a Python iterator.
    ///
    /// If the retrieval is done successfully, returns `Ok` with a shared reference
    /// to a Python iterator for it in it. Otherwise, returns `Err` with a `PyErr`
    /// in it. The `PyErr` will explain the cause of the failure.
    ///
    /// [nodes]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.nodes
    pub fn nodes_iterator(&self) -> PyResult<&PyIterator> {
        self.getattr("nodes")?.iter()
    }

    /// An interface for [`eliminate_dead_code`][eliminate_dead_code] instance
    /// method of `Graph` PyTorch class.
    ///
    /// If the method call is done successfully, returns `Ok(())`. Otherwise,
    /// returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of
    /// the failure.
    ///
    /// [eliminate_dead_code]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.eliminate_dead_code
    pub fn eliminate_dead_code(&self) -> PyResult<()> {
        self.getattr("eliminate_dead_code")?.call0()?;
        Ok(())
    }

    /// An interface for [`lint`][lint] instance method of `Graph` PyTorch
    /// class.
    ///
    /// If the
    /// method call is done successfully, returns `Ok(())`. Otherwise, returns
    /// `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the
    /// failure.
    ///
    /// [lint]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint
    pub fn lint(&self) -> PyResult<()> {
        self.getattr("lint")?.call0()?;
        Ok(())
    }

    /// An interface for [`create_node`][create_node] instance method of
    /// `Graph` PyTorch class, but `type_expr` is not given (`None`).
    /// Also, if `meta` is given, the newly created `Node` will have an
    /// attribute `meta`, whose value will be the given argument `meta`.
    ///
    /// If the method call is done successfully, returns
    /// `Ok` with a shared reference to the newly created `Node` in it.
    /// Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [create_node]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.create_node
    pub fn create_node<S: AsRef<str>>(
        &self,
        op: Op,
        target: Target,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
        name: S,
        meta: Option<HashMap<String, PyObject>>,
    ) -> PyResult<&Node> {
        let py = self.py();
        let method_args = PyTuple::new(py, &[op.into_py(py), target.into_py(py)]);
        let method_kwargs = {
            let method_kwargs = PyDict::new(py);
            method_kwargs.set_item(
                "args",
                PyTuple::new(
                    py,
                    args.into_iter()
                        .map(|arg| self.argument_into_py(py, arg))
                        .collect::<PyResult<Vec<_>>>()?,
                ),
            )?;
            method_kwargs.set_item(
                "kwargs",
                kwargs
                    .into_iter()
                    .map(|(key, arg)| Ok((key, self.argument_into_py(py, arg)?)))
                    .collect::<PyResult<HashMap<_, _>>>()?,
            )?;
            method_kwargs.set_item("name", name.as_ref())?;
            Some(method_kwargs)
        };
        let node = self.call_method("create_node", method_args, method_kwargs)?;
        if let Some(meta) = meta {
            node.setattr("meta", meta)?;
        }
        Ok(node.downcast()?)
    }

    /// Create and insert a placeholder `Node` into this `Graph`.
    /// A placeholder represents a function input.
    /// `name` is the name for the input value.
    ///
    /// This does the same what [`placeholder`][placeholder] instance method of
    /// `Graph` PyTorch class does, but `type_expr` is `None` and `default_value`
    /// is `inspect.Signature.empty`.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [placeholder]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.placeholder
    pub fn placeholder<S: AsRef<str>>(&self, name: S) -> PyResult<&Node> {
        self.create_node(
            Op::Placeholder,
            Target::Str(name.as_ref().to_string()),
            None,
            None,
            name,
            None,
        )
    }

    /// Create and insert an output `Node` into this `Graph`.
    /// `args` is the value that should be returned by this output node.
    /// `args` has to be `Argument::NodeTuple`
    ///
    /// This does the same what [`output`][output] instance method of
    /// `Graph` PyTorch class does, but `type_expr` is `None` and
    /// the newly created `Node` has a name 'output'.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [output]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.output
    pub fn output(&self, args: Argument) -> PyResult<&Node> {
        if !matches!(args, Argument::NodeTuple(_)) {
            return Err(PyTypeError::new_err("output arg must be a tuple of nodes"));
        }

        let name = "output";
        self.create_node(
            Op::Output,
            Target::Str(name.to_string()),
            vec![args],
            None,
            name,
            None,
        )
    }

    /// Create and insert a call_function `Node` into this `Graph`.
    /// call_function `Node` represents a call to a Python callable,
    /// specified by `custom_fn`.
    ///
    /// This does the same what [`call_function`][call_function] instance method
    /// of `Graph` PyTorch class does, but the name of `the_function` parameter
    /// is changed into `custom_fn`, `type_expr` is not given (`None`), and
    /// the `name` for the name of this node is given.
    ///
    /// `custom_fn` must be a `CustomFn`, a python callable which calls
    /// a Rust function actually.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [call_function]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_function
    pub fn call_custom_function<S: AsRef<str>>(
        &self,
        name: S,
        custom_fn: CustomFn,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
    ) -> PyResult<&Node> {
        self.create_node(
            Op::CallFunction,
            Target::CustomFn(custom_fn),
            args,
            kwargs,
            name,
            None,
        )
    }

    /// Create and insert a call_function `Node` into this `Graph`.
    /// call_function `Node` represents a call to a Python callable,
    /// specified by `the_function`.
    ///
    /// This does the same what [`call_function`][call_function] instance method
    /// of `Graph` PyTorch class does, but `type_expr` is not given (`None`) and
    /// the `name` for the name of this node is given.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [call_function]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_function
    pub fn call_python_function<S: AsRef<str>>(
        &self,
        name: S,
        the_function: Py<PyAny>,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
    ) -> PyResult<&Node> {
        self.create_node(
            Op::CallFunction,
            Target::Callable(the_function),
            args,
            kwargs,
            name,
            None,
        )
    }

    pub fn call_arg_method<S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        name: S1,
        method_name: S2,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
    ) -> PyResult<&Node> {
        self.create_node(
            Op::CallMethod,
            Target::Str(method_name.as_ref().to_string()),
            args,
            kwargs,
            name,
            None,
        )
    }

    pub fn call_module<S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        name: S1,
        module_name: S2,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
    ) -> PyResult<&Node> {
        let owning_module = self.getattr("owning_module")?;
        if owning_module.is_true()?
            && owning_module
                .call_method1("get_submodule", (module_name.as_ref(),))?
                .is_none()
        {
            self.py().import("warnings")?.getattr("warn")?.call1((
                "Attempted to insert a call_module Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule",
            ))?;
        }
        self.create_node(
            Op::CallModule,
            Target::Str(module_name.as_ref().to_string()),
            args,
            kwargs,
            name,
            None,
        )
    }

    pub fn fetch_attr<S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        name: S1,
        qualified_name: S2,
    ) -> PyResult<&Node> {
        let py = self.py();
        let qname: PyObject = qualified_name.as_ref().into_py(self.py());
        let owning_module = self.getattr("owning_module")?;
        if owning_module.is_true()? {
            let qname = qname.as_ref(self.py());
            let (module_path, pname) = {
                let tuple: &PyTuple = qname.call_method1("rpartition", (".",))?.downcast()?;
                (tuple.get_item(0)?, tuple.get_item(2)?)
            };
            let get_attr_reference_exists =
                match owning_module.call_method1("get_submodule", (module_path,)) {
                    // TODO
                    Ok(submod) => {
                        let pname_str = pname.extract::<String>()?;
                        if !(submod.hasattr(pname_str.as_str())?) {
                            false
                        } else {
                            let res = submod.getattr(pname_str.as_str())?;
                            let nn = py.import("torch.nn")?;
                            !(!(res.is_instance(nn.getattr("Module")?)?)
                                && !(res.is_instance(nn.getattr("Parameter")?)?)
                                && !(submod.getattr("_buffers")?.contains(pname)?))
                        }
                    }
                    Err(e) => {
                        if e.is_instance_of::<PyAttributeError>(py) {
                            py.import("warnings")?.getattr("warn")?.call1((format!(
                                "Failed to fetch module {}",
                                module_path.extract::<String>()?
                            ),))?;
                            false
                        } else {
                            return Err(e);
                        }
                    }
                };
            if get_attr_reference_exists {
                let kwargs = PyDict::new(py);
                kwargs.set_item("stacklevel", 2)?;
                py.import("warnings")?.getattr("warn")?.call(
                    ("Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer",),
                    Some(kwargs),
                )?;
            }
        }
        self.create_node(
            Op::GetAttr,
            Target::Str(qname.extract::<String>(py)?),
            None,
            None,
            name,
            None,
        )
    }

    /// Copy a `Node` from another `Graph` into this `Graph`(`self`).
    /// `node` is the node to copy into `self`.
    /// `mapper` needs to transform arguments
    /// from the graph of `node` to the graph of self.
    ///
    /// This does the same what [`node_copy`][node_copy] instance method of `Graph`
    /// PyTorch class does.
    ///
    /// If the copying and insertion of the `Node` is done successfuly,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [node_copy]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.node_copy
    pub fn node_copy(
        &self,
        node: &Node,
        mapper: Option<&HashMap<String, String>>,
    ) -> PyResult<&Node> {
        let graph: Py<Graph> = self.into();
        let f: FunctionWrapper = if let Some(mapper) = mapper {
            let mapper = mapper.clone();
            Arc::new(move |args, _| {
                let name = args.get_item(0)?.downcast::<Node>()?.name()?;
                let mapped = mapper.get(&name).ok_or(PyRuntimeError::new_err(format!(
                    "Failed to get mapped arg from mapper {name:?}"
                )))?;
                let graph = graph.as_ref(args.py());
                let mapped = graph.lookup_node(mapped.clone())?.unwrap();
                Ok(mapped.into())
            })
        } else {
            Arc::new(|args, _| Ok(args.get_item(0)?.into()))
        };
        let f = CustomFn::new("f", f);
        Ok(self.call_method1("node_copy", (node, f))?.downcast()?)
    }

    pub fn erase_node(&self, node: &Node) -> PyResult<()> {
        self.call_method1("erase_node", (node,)).map(|_| ())
    }

    pub fn erase_node_by_name<S: AsRef<str>>(&self, name: S) -> PyResult<()> {
        let node = self
            .lookup_node(name.as_ref())?
            .ok_or(PyRuntimeError::new_err(format!(
                "no such node: \"{}\"",
                name.as_ref()
            )))?;
        self.call_method1("erase_node", (node,)).map(|_| ())
    }

    /// Retrieve the names of argument `Node`s of the `Node` named
    /// as the value of `node_name` in this `Graph`.
    ///
    /// If this graph doesn't have a `Node` named as the value of
    /// `node_name`, returns `Ok(None)`.
    /// If this graph have a `Node` named as the value of `node_name`,
    /// returns `Ok(Some)` with a `Vec` of names of argument `Node`s of the
    /// `Node`, in the `Some`.
    /// If something fails while looking into this `Graph`, returns `Err`
    /// with a `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn flatten_node_args<S: AsRef<str>>(&self, node_name: S) -> PyResult<Option<Vec<String>>> {
        let named_nodes = self.extract_named_nodes()?;
        match named_nodes.get(node_name.as_ref()) {
            Some(node) => Ok(Some(node.downcast::<Node>()?.flatten_node_args()?)),
            None => Ok(None),
        }
    }

    /// Retrieve the names of user `Node`s of the `Node` named
    /// as the value of `node_name` in this `Graph`.
    ///
    /// If this graph doesn't have a `Node` named as the value of
    /// `node_name`, returns `Ok(None)`.
    /// If this graph have a `Node` named as the value of `node_name`,
    /// returns `Ok(Some)` with a `Vec` of names of user `Node`s of the
    /// `Node`, in the `Some`.
    /// If something fails while looking into this `Graph`, returns `Err`
    /// with a `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn users<S: AsRef<str>>(&self, node_name: S) -> PyResult<Option<Vec<String>>> {
        let named_nodes = self.extract_named_nodes()?;
        match named_nodes.get(node_name.as_ref()) {
            Some(node) => {
                let user_keys = node.getattr("users")?.getattr("keys")?.call0()?;
                Ok(Some(
                    user_keys
                        .iter()?
                        .map(|r| r?.getattr("name")?.extract())
                        .collect::<PyResult<Vec<String>>>()?,
                ))
            }
            None => Ok(None),
        }
    }

    /// Stringify this `Graph`.
    ///
    /// This does the same what `__str__` instance method of `Graph`
    /// PyTorch class.
    ///
    /// If stringifying is done successfully, returns `Ok` with the
    /// resulting string in it. Otherwise, returns `Err` with a
    /// `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn graph_to_string(&self, py: Python<'_>) -> PyResult<String> {
        let builtin_str_fn = py.import("builtins")?.getattr("str")?;
        builtin_str_fn.call1((self,))?.extract()
    }

    /// Collect all named `Node`s of this `Graph`.
    ///
    /// Make an `IndexMap` which maps each `Node`'s name to a shared reference
    /// of the `Node` itself, for every `Node` in `self`.
    ///
    /// If this process is done successfully, returns `Ok` with the `IndexMap` in it.
    /// Otherwise, return `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn extract_named_nodes(&self) -> PyResult<IndexMap<String, &Node>> {
        self.nodes_iterator()?
            .map(|r| {
                let r = r?;
                let name: String = r.getattr("name")?.extract()?;
                Ok((name, r.downcast()?))
            })
            .collect()
    }

    /// Lookup a `Node` by its name(`name`) in this `Graph`.
    ///
    /// If there is no `Node` with a name named as the value of `name`,
    /// `Ok(None)` is returned.
    /// If there exists such `Node` in this `Graph`,
    /// `Ok(Some)` with a shared reference to the `Node` is returned.
    /// If this process fails, returns `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn lookup_node<S: AsRef<str>>(&self, name: S) -> PyResult<Option<&Node>> {
        let mut nodes = self.nodes_iterator()?;
        nodes
            .find_map(|node| {
                (|| {
                    let node = node?;
                    let n: String = node.getattr("name")?.extract()?;
                    Ok(if n == name.as_ref() {
                        Some(node.downcast::<Node>()?)
                    } else {
                        None
                    })
                })()
                .transpose()
            })
            .transpose()
    }

    fn argument_into_py(&self, py: Python<'_>, arg: Argument) -> PyResult<PyObject> {
        match arg {
            Argument::Node(node_name) => {
                let node = self.lookup_node(node_name)?.unwrap();
                Ok(node.into_py(py))
            }
            Argument::NodeTuple(node_names) => Ok(PyTuple::new(py, {
                let named_nodes = self.extract_named_nodes()?;
                node_names
                    .into_iter()
                    .map(move |name| *named_nodes.get(&name).unwrap())
            })
            .into_py(py)),
            Argument::NodeList(node_names) => Ok(PyList::new(py, {
                let named_nodes = self.extract_named_nodes()?;
                node_names
                    .into_iter()
                    .map(move |name| *named_nodes.get(&name).unwrap())
            })
            .into_py(py)),
            Argument::OptionalNodeTuple(node_names) => Ok(PyTuple::new(py, {
                let named_nodes = self.extract_named_nodes()?;
                node_names.into_iter().map(move |name| {
                    name.map_or_else(
                        || py.None(),
                        |name| named_nodes.get(&name).unwrap().into_py(py),
                    )
                })
            })
            .into_py(py)),
            Argument::OptionalNodeList(node_names) => Ok(PyList::new(py, {
                let named_nodes = self.extract_named_nodes()?;
                node_names.into_iter().map(move |name| {
                    name.map_or_else(
                        || py.None(),
                        |name| named_nodes.get(&name).unwrap().into_py(py),
                    )
                })
            })
            .into_py(py)),
            Argument::NoneTuple(len) => Ok(PyTuple::new(py, vec![(); len]).into()),
            Argument::NoneList(len) => Ok(PyList::new(py, vec![(); len]).into()),
            Argument::Bool(value) => Ok(value.into_py(py)),
            Argument::Int(value) => Ok(value.into_py(py)),
            Argument::Float(value) => Ok(value.into_py(py)),
            Argument::VecBool(values) => Ok(PyList::new(py, values).into()),
            Argument::VecInt(values) => Ok(PyList::new(py, values).into()),
            Argument::VecFloat(values) => Ok(PyList::new(py, values).into()),
            Argument::Dtype(value) => Ok(value.into_py(py)),
            Argument::Layout(value) => Ok(value.into_py(py)),
            Argument::Device(value) => Ok(value.into_py(py)),
            Argument::MemoryFormat(value) => Ok(value.into_py(py)),
            Argument::Value(ob) => Ok(ob.into_py(py)),
            Argument::EmptyList => Ok(PyList::empty(py).into_py(py)),
            Argument::None => Ok(py.None()),
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("origin", &PyObject::from(self))
            .field(
                "named_nodes",
                &self.extract_named_nodes().map_err(|_| Error)?,
            )
            .finish()
    }
}
