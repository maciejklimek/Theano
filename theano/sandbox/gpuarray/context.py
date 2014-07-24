from theano import Type, Variable, Constant

try:
    import pygpu
    from pygpu.gpuarray import GpuContext
except ImportError:
    pass

def register_context(ctx, name):
    _contexts[name] = ctx
    if name in _context_cache:
        del _context_cache[name]

def get_context(name=None):
    if name not in _context_cache:
        _context_cache[name] = GpuContextConstant(name)
    return _context_cache[name]

_contexts = {}
_context_cache = {}

class GpuContextType(Type):
    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        return "GpuContextType"

    def filter(self, data, strict=False, allow_downcast=None):
        # might want to add a cast from string to context in
        # non-strict mode later.
        if not isinstance(data, GpuContext):
            raise TypeError("%s exepcted a GpuContext object." % self,
                            data, type(data))
        return data

    @staticmethod
    def values_eq(a, b):
        return a == b

    def make_variable(self, name=None):
        return self.Variable(self, name=name)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def c_declare(self, name, sub, check_input=True):
        return """
        PyGpuContextObject *%(name)s;
        """ % dict(name=name)

    def c_init(self, name, sub):
        return "%s = NULL;" % (name,)

    def c_extract(self, name, sub, check_input=True):
        if check_input:
            res = """
        if (!PyObject_TypeCheck(py_%(name)s, &PyGpuContextType)) {
          PyErr_SetString(PyExc_TypeError, "expected a GpuContext");
          %(fail)s
        }
        """ % dict(name=name, fail=sub['fail'])
        else:
            res = ""
        return res + """
        %(name)s = (PyGpuContextObject *)py_%(name)s
        Py_INCREF(%(name)s);
        """ % dict(name=name)

    def c_cleanup(self, name, sub):
        return "Py_XDECREF(%(name)s); %(name)s = NULL;" % dict(name=name)

    def c_sync(self, name, sub):
                return """
        if (!%(name)s) {
          Py_XDECREF(py_%(name)s);
          Py_INCREF(Py_None);
          py_%(name)s = Py_None;
        } else if ((void *)py_%(name)s != (void *)%(name)s) {
          Py_XDECREF(py_%(name)s);
          py_%(name)s = (PyObject *)%(name)s;
          Py_INCREF(py_%(name)s);
        }
        """ % dict(name=name)

    def c_init_code(self):
        return ['import_pygpu__gpuarray();']

    def c_headers(self):
        return ['<gpuarray_api.h>']

    def c_header_dirs(self):
        return [pygpu.get_include()]

    def c_code_cache_version(self):
        ver = pygpu.gpuarray.api_version()
        return (1, ver[0])


class GpuContextVariable(Variable):
    pass


GpuContextType.Variable = GpuContextVariable


# This is used in pickling
def _make_context_constant(name):
    return GpuContextConstant(name)


class GpuContextConstant(Constant):
    def __init__(self, name):
        try:
            data = _context[name]
        except KeyError:
            raise ValueError("Undefined context name %r" % (name,))
        Constant.__init__(self, type=GpuContextType(), data=data)
        self._name = name

    def signature(self):
        # If two contexts have the same name they should have the same
        # data and it is a bug (or user error) if this is not the case
        return self._name

    def __str__(self):
        return "GpuContextConstant(%r)" % (self._name,)

    def __reduce__(self):
        return _make_context_constant, self._name


GpuContextType.Constant = GpuContextConstant
