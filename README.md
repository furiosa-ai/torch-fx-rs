# `torch-fx-rs`

Rust APIs to handle PyTorch graph modules and graphs

## Where to use

This API can help writing a Python module in Rust **using [PyO3](https://pyo3.rs/v0.20.0/)**, in case **the module needs to handle PyTorch graph modules or graphs**.

## APIs

### `pub struct GraphModule`

```rust
#[repr(transparent)]
pub struct GraphModule(_);
```

A wrapper for PyTorch's [`GraphModule`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule) class.

The constructor method of this returns a shared reference [`&GraphModule`](#pub-struct-graphmodule) instead of an owned value. The return value is GIL-bound owning reference into Python's heap.

#### Methods

*   ```rust
    pub fn new<'py>(
        py: Python<'py>,
        nn: &GraphModule,
        graph: &Graph
    ) -> PyResult<&'py Self>
    ```

    Create new instance of [`GraphModule`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule) PyTorch class with PyTorch [native constructor](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.__init__) but `class_name` is not given (so that it remains as the default value `'GraphModule'`).

    If new instance is created succesfully, returns `Ok` with a shared reference to the newly created instance in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn new_with_empty_gm<'py>(
        py: Python<'py>,
        graph: &Graph
    ) -> PyResult<&'py Self>
    ```

    Create new instane of [`GraphModule`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule) PyTorch class with PyTorch [native constructor](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.__init__) but `class_name` is not given (so that it remains as the default value `'GraphModule'`) and `root` is a newly created [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) by `torch.nn.Module()`.

    If new instance is created succesfully, returns `Ok` with a shared reference to the newly created instance in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn extract_parameters_view<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<HashMap<String, BufferView<'py>>>
    ```

    Collect all parameters of this [`GraphModule`](#pub-struct-graphmodule) as zero-copy views.

    Returns a `HashMap` mapping parameter names to `BufferView<'py>` values, whose lifetimes are tied to the provided GIL token. No data is copied.

*   ```rust
    pub fn extract_buffers_view<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<HashMap<String, BufferView<'py>>>
    ```

    Collect all buffers of this [`GraphModule`](#pub-struct-graphmodule) as zero-copy views.

    Returns a `HashMap` mapping buffer names to `BufferView<'py>` values, lifetime-bound to `py`. No data is copied.

*   ```rust
    pub fn graph(&self) -> PyResult<&Graph>
    ```

    Retrieve the `graph` attribute of this [`GraphModule`](#pub-struct-graphmodule).

    If the retrieval is done successfully, returns `Ok` with a shared reference to the `graph` attribute (`&Graph`) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn get_parameter_view<'py>(
        &self,
        py: Python<'py>,
        name: &str
    ) -> PyResult<Option<BufferView<'py>>>
    ```

    Get a zero-copy view into the underlying storage of the parameter named `name`. Since `BufferView` dereferences to `[u8]`, you can use methods like `view.len()` or pass `&*view` where a slice is expected.
    
    Returns `Ok(None)` if absent, or `Ok(Some(BufferView<'py>))` if present. No data is copied.

*   ```rust
    pub fn count_parameters(&self) -> PyResult<usize>
    ```

    Get the number of parameters of this [`GraphModule`](#pub-struct-graphmodule).

    If a Python error occurs during this procedure, returns `Err` with a `PyErr` in it. `PyErr` will explain the error. Otherwise, returns `Ok` with the number of parameters of this [`GraphModule`](#pub-struct-graphmodule) in it.

*   ```rust
    pub fn get_buffer_view<'py>(
        &self,
        py: Python<'py>,
        name: &str
    ) -> PyResult<Option<BufferView<'py>>>
    ```

    Get a zero-copy view into the underlying storage of the buffer named `name`. Since `BufferView` dereferences to `[u8]`, you can use methods like `view.len()` or pass `&*view` where a slice is expected.
    
    Returns `Ok(None)` if absent, or `Ok(Some(BufferView<'py>))` if present. No data is copied.

#### Zero-Copy Views: Safety & Usage

`BufferView<'py>` provides read-only, zero-copy access to a tensor's underlying storage.

- Lifetime: the view is tied to the GIL token (`Python<'py>`) used to create it. Do not store it beyond the GIL scope. If you need to persist the data, copy it via `&*view` into an owned buffer.
- Read-only: do not mutate the underlying tensor while a view is held. Treat the view as immutable bytes.
- Strided tensors: views represent the storage range backing the tensor, not logical shape. Use separate metadata (e.g., `TensorMeta`) to reason about shape/stride.

Example:

```rust
pyo3::prepare_freethreaded_python();
Python::with_gil(|py| {
    let gm = GraphModule::new_with_empty_gm(py, Graph::new(py).unwrap()).unwrap();
    // Assume gm has a parameter "w"
    if let Some(view) = gm.get_parameter_view(py, "w").unwrap() {
        // Safe to read within the GIL scope
        let len = view.len();
        let first = view[0];
        // If you need to keep data, copy it out
        let owned: Vec<u8> = view.to_vec();
        drop((len, first, owned));
    }
});
```

*   ```rust
    pub fn iter_parameters_view<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<impl Iterator<Item = PyResult<(String, BufferView<'py>)>> + 'py>
    ```

    Iterate parameters lazily as `(name, BufferView)`, avoiding intermediate `HashMap` allocation. Each item is a `PyResult` to surface Python interop errors during iteration.

*   ```rust
    pub fn iter_buffers_view<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<impl Iterator<Item = PyResult<(String, BufferView<'py>)>> + 'py>
    ```

    Iterate buffers lazily as `(name, BufferView)` with zero-copy semantics.

*   ```rust
    pub fn count_buffers(&self) -> PyResult<usize>
    ```

    Get the number of buffers of this [`GraphModule`](#pub-struct-graphmodule).

    If a Python error occurs during this procedure, returns `Err` with a `PyErr` in it. `PyErr` will explain the error. Otherwise, returns `Ok` with the number of parameters of this [`GraphModule`](#pub-struct-graphmodule) in it.

*   ```rust
    pub fn print_readable(&self) -> PyResult<String>
    ```

    Stringify this [`GraphModule`](#pub-struct-graphmodule).

    This does the same what [`print_readable`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.print_readable) instance method of [`GraphModule`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule) PyTorch class does, but `print_output` is given as `True`.

    If stringifying is done successfully, returns `Ok` with the resulting string in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

### `pub struct Graph`

```rust
#[repr(transparent)]
pub struct Graph(_);
```

A wrapper for PyTorch's [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) class.

The constructor method of this returns a shared reference [`&Graph`](#pub-struct-graph) instead of an owned value. The return value is GIL-bound owning reference into Python's heap.

#### Methods

*   ```rust
    pub fn new(py: Python<'_>) -> PyResult<&Self>
    ```

    Create new instance of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class with PyTorch native constructor.
    
    If new instance is created successfully, returns `Ok` with a shared reference to the newly created instance in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn nodes_iterator(&self) -> PyResult<&PyIterator>
    ```

    Retrieve all the [`Node`s](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.nodes) of this [`Graph`](#pub-struct-graph) as a Python iterator.
    
    If the retrieval is done successfully, returns `Ok` with a shared reference
    to a Python iterator for it in it. Otherwise, returns `Err` with a `PyErr`
    in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn eliminate_dead_code(&self) -> PyResult<()>
    ```

    An interface for [`eliminate_dead_code`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.eliminate_dead_code) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class.
    
    If the method call is done successfully, returns `Ok(())`. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn lint(&self) -> PyResult<()>
    ```

    An interface for [`lint`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class.
    
    If the method call is done successfully, returns `Ok(())`. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn create_node<S: AsRef<str>>(
        &self,
        op: Op,
        target: Target,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
        name: S,
        meta: Option<HashMap<String, PyObject>>,
    ) -> PyResult<&Node>
    ```

    An interface for [`create_node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.create_node) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class, but `type_expr` is not given (`None`). Also, if `meta` is given, the newly created [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) will have an attribute `meta`, whose value will be the given argument `meta`.
    
    If the method call is done successfully, returns `Ok` with a shared reference to the newly created [`Node`](#pub-struct-node) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn placeholder<S: AsRef<str>>(
        &self,
        name: S
    ) -> PyResult<&Node>
    ```

    Create and insert a placeholder [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) into this [`Graph`](#pub-struct-graph). A placeholder represents a function input. `name` is the name for the input value.
    
    This does the same what [`placeholder`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.placeholder) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class does, but `type_expr` is `None` and `default_value` is `inspect.Signature.empty`.
    
    If the creation and insertion of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) is done successfully, returns `Ok` with a shared reference to the newly created [`Node`](#pub-struct-node) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn output(
        &self,
        args: Argument
    ) -> PyResult<&Node>
    ```

    Create and insert an output [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) into this [`Graph`](#pub-struct-graph). `args` is the value that should be returned by this output node. `args` has to be [`Argument::NodeTuple`](#pub-enum-argument).
    
    This does the same what [`output`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.output) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class does, but `type_expr` is `None` and the newly created [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) has a name 'output'.
    
    If the creation and insertion of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) is done successfully, returns `Ok` with a shared reference to the newly created [`Node`](#pub-struct-node) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn call_custom_function<S: AsRef<str>>(
        &self,
        name: S,
        custom_fn: CustomFn,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
    ) -> PyResult<&Node>
    ```

    Create and insert a call_function [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) into this [`Graph`](#pub-struct-graph). call_function [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) represents a call to a Python callable, specified by `custom_fn`.
    
    This does the same what [`call_function`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_function) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class does, but the name of `the_function` parameter is changed into `custom_fn`, `type_expr` is not given (`None`), and the `name` for the name of this node is given.

    `custom_fn` must be a `CustomFn`, a python callable which calls a Rust function actually.
    
    If the creation and insertion of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) is done successfully, returns `Ok` with a shared reference to the newly created [`Node`](#pub-struct-node) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn call_python_function<S: AsRef<str>>(
        &self,
        name: S,
        the_function: Py<PyAny>,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
    ) -> PyResult<&Node>
    ```

    Create and insert a call_function [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) into this [`Graph`](#pub-struct-graph). call_function [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) represents a call to a Python callable, specified by `the_function`.
    
    This does the same what [`call_function`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_function) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class does, but `type_expr` is not given (`None`) and the `name` for the name of this node is given.
    
    If the creation and insertion of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) is done successfully, returns `Ok` with a shared reference to the newly created [`Node`](#pub-struct-node) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn node_copy(
        &self,
        node: &Node,
        mapper: Option<&HashMap<String, String>>,
    ) -> PyResult<&Node>
    ```

    Copy a [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) from another [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) into this [`Graph`](#pub-struct-graph)(`self`). `node` is the node to copy into `self`. `mapper` needs to transform arguments from the graph of `node` to the graph of self.
    
    This does the same what [`node_copy`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.node_copy) instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class does.
    
    If the copying and insertion of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) is done successfuly, returns `Ok` with a shared reference to the newly created [`Node`](#pub-struct-node) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn flatten_node_args<S: AsRef<str>>(
        &self,
        node_name: S
    ) -> PyResult<Option<Vec<String>>>
    ```

    Retrieve the names of argument [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) named as the value of `node_name` in this [`Graph`](#pub-struct-graph).
    
    If this graph doesn't have a [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) named as the value of `node_name`, returns `Ok(None)`. If this graph have a [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) named as the value of `node_name`, returns `Ok(Some)` with a `Vec` of names of argument [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node), in the `Some`. If something fails while looking into this [`Graph`](#pub-struct-graph), returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn flatten_node_args_nodes<S: AsRef<str>>(
        &self,
        node_name: S,
    ) -> PyResult<Option<Vec<&Node>>>
    ```

    Retrieve the argument nodes of `node_name` as borrowed `&Node` references, avoiding string copies. Returns `Ok(None)` if the node does not exist.

*   ```rust
    pub fn users<S: AsRef<str>>(
        &self,
        node_name: S
    ) -> PyResult<Option<Vec<String>>>
    ```

    Retrieve the names of user [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) named as the value of `node_name` in this [`Graph`](#pub-struct-graph).
    
    If this graph doesn't have a [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) named as the value of `node_name`, returns `Ok(None)`. If this graph have a [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) named as the value of `node_name`, returns `Ok(Some)` with a `Vec` of names of user [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s of the [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node), in the `Some`. If something fails while looking into this `Graph`, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn users_nodes<S: AsRef<str>>(
        &self,
        node_name: S
    ) -> PyResult<Option<Vec<&Node>>>
    ```

    Retrieve user nodes of `node_name` as borrowed `&Node` references without copying names. Returns `Ok(None)` if the node does not exist.

*   ```rust
    pub fn graph_to_string(
        &self,
        py: Python<'_>
    ) -> PyResult<String>
    ```

    Stringify this [`Graph`](#pub-struct-graph).
    
    This does the same what `__str__` instance method of [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) PyTorch class.
    
    If stringifying is done successfully, returns `Ok` with the resulting string in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn extract_named_nodes(&self)
        -> PyResult<IndexMap<String, &Node>>
    ```

    Collect all named [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s of this [`Graph`](#pub-struct-graph).
    
    Make an `IndexMap` which maps each [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)'s name to a shared reference of the [`Node`](#pub-struct-node) itself, for every [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) in `self`.
    
    If this process is successful, returns `Ok` with the `IndexMap` in it. Otherwise, return `Err` with a `PyErr` in it. `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn lookup_node<S: AsRef<str>>(
        &self,
        name: S
    ) -> PyResult<Option<&Node>>
    ```

    Lookup a [`Node`](#pub-struct-node) by its name(`name`) in this [`Graph`](#pub-struct-graph).
    
    If there is no [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) with a name named as the value of `name`, `Ok(None)` is returned. If there exists such [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) in this [`Graph`](#pub-struct-graph), `Ok(Some)` with a shared reference to the [`Node`](#pub-struct-node) is returned. If this process fails, returns `Err` with a `PyErr` in it. `PyErr` will explain the cause of the failure.

### `pub struct Node`

```rust
#[repr(transparent)]
pub struct Node(_);
```

A wrapper for PyTorch's [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) class.

This appears as a shared reference [`&Node`](#pub-struct-node) into Python's heap instead of an owned value.

#### Methods

*   <a id="flatten_node_args"></a>
    ```rust
    pub fn flatten_node_args(&self) -> PyResult<Vec<String>>
    ```

    Retrieve the names of argument [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s of this [`Node`](#pub-struct-node). Although a [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) can have multiple arguments and an argument can have one or more [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s, the result will contain all the argument [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node)s' names in a 1-dimensional vector. (This is why this method is named [`flatten_node_args`](#flatten_node_args).)
    
    If the retrieval is done successfully, returns `Ok` with a `Vec` of names of argument nodes. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn args(&self) -> PyResult<Vec<Argument>>
    ```

    Retrieve the arguments of this [`Node`](#pub-struct-node).
    
    If the retrieval is done successfully, returns `Ok` with a `Vec<`[`Argument`](#pub-enum-argument)`>` containing the arguments. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn name(&self) -> PyResult<String>
    ```

    Retrieve the name of this [`Node`](#pub-struct-node).
    
    If the retrieval is done successfully, returns `Ok` with the name in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn op(&self) -> PyResult<Op>
    ```

    Retrieve the opcode of this [`Node`](#pub-struct-node).
    
    If the retrieval is done successfully, returns `Ok` with the opcode in [`Op`](#pub-enum-op) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn target(&self) -> PyResult<Target>
    ```

    Retrieve the target this [`Node`](#pub-struct-node) should call.
    
    If the retrieval is done successfully, returns `Ok` with the target in [`Target`](#pub-enum-target) in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn kwargs(&self)
        -> PyResult<HashMap<String, Argument>>
    ```

    Retrieve the kwargs to be passed to the target of this [`Node`](#pub-struct-node).
    
    If the retrieval is done successfully, returns `Ok` with the kwargs in `HashMap<String, `[`Argument`](#pub-enum-argument)`>` in it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the failure.

*   ```rust
    pub fn meta(&self)
        -> PyResult<HashMap<String, PyObject>>
    ```

    Retrieve the meta of this [`Node`](#pub-struct-node).
    
    If this [`Node`](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) has an attribute `meta`, returns `Ok` with the meta in `HashMap<String, PyObject>` in it. Otherwise, returns `Ok(Default::default())`. This never returns `Err`.

### `pub type FunctionWrapper`

Wrapper for a Rust function. This wraps a function to execute it in Python. Therefore, the function needs to receive 2 arguments, args as `&PyTuple` and kwargs as `Option<&PyDict>`, and return `PyResult<PyObject>`.

### `pub struct CustomFn`

```rust
#[pyclass]
#[derive(Clone)]
pub struct CustomFn {
    pub func_name: String,
    /* private fields */
}
```

An interface for Python callable object which actually executes a Rust function.

#### Fields

* <a id="func_name"></a>`pub func_name: String`
    * Name of the custom function

#### Methods

*   ```rust
    pub fn new<S: AsRef<str>>(
        func_name: S,
        func: FunctionWrapper
    ) -> Self
    ```

    Create a new Python callable object which is named as the value of [`func_name`](#func_name) and actually executes a Rust function wrapped in `func`.

### `pub struct TensorMeta`

```rust
#[derive(Debug, Clone, FromPyObject)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub requires_grad: bool,
    pub stride: Vec<usize>,
    pub memory_format: Option<MemoryFormat>,
    pub is_quantized: bool,
    pub qparams: HashMap<String, PyObject>,
}
```

A structure containing pertinent information about a tensor within a PyTorch program.

([reference](https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py#L12))

### `pub enum Op`

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Op {
    Placeholder,
    CallFunction,
    CallMethod,
    CallModule,
    GetAttr,
    Output,
}
```

A representation of opcodes for [`Node`](#pub-struct-node)s.

### `pub enum Target`

```rust
#[derive(Debug, Clone)]
pub enum Target {
    Str(String),
    TorchOp(String, PyObject),
    BuiltinFn(String, PyObject),
    Callable(PyObject),
    CustomFn(CustomFn),
}
```

A representation of targets for [`Node`](#pub-struct-node)s.

### `pub enum Argument`

```rust
#[derive(Debug, Clone)]
pub enum Argument {
    Node(String),
    NodeList(Vec<String>),
    NodeTuple(Vec<String>),
    OptionalNodeList(Vec<Option<String>>),
    OptionalNodeTuple(Vec<Option<String>>),
    NoneList(usize),
    NoneTuple(usize),
    Bool(bool),
    Int(i64),
    Float(f64),
    VecBool(Vec<bool>),
    VecInt(Vec<i64>),
    VecFloat(Vec<f64>),
    Dtype(Dtype),
    Layout(Layout),
    Device(Device),
    MemoryFormat(MemoryFormat),
    Value(PyObject),
    EmptyList,
    None,
}
```

A representation of arguments for [`Node`](#pub-struct-node)s.

### `pub enum Dtype`

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Dtype {
    Float32,
    Float64,
    Complex64,
    Complex128,
    Float16,
    Bfloat16,
    Uint8,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
}
```

An `enum` which represents the data type of a [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor).

([reference](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype))

### `pub enum MemoryFormat`

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryFormat {
    ContiguousFormat,
    ChannelsLast,
    ChannelsLast3d,
    PreserveFormat,
}
```

An `enum` which represents the memory format on which a [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) is or will be allocated.

([reference](https://pytorch.org/docs/stable/tensor_attributes.html#torch-memory-format))

### `pub enum Device`

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Device {
    Cpu(Option<usize>),
    Cuda(Option<usize>),
    Mps(Option<usize>),
}
```

An `enum` which represents the device on which a [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) is or will be allocated.

([reference](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device))

## Documentation

By executing following, the documentation, by `cargo-docs`, for this crate will open.
```
cargo doc --open
```

[More detailed documentation for `torch.fx`](https://pytorch.org/docs/stable/fx.html) may be needed.
