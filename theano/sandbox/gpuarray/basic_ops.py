import os

import numpy

import theano
from theano import Op, Apply
from theano import tensor, scalar, config
from theano.gradient import grad_undefined
from theano.scalar import Scalar
from theano.tensor import TensorType
from theano.tensor.basic import Alloc, Join, Split

from theano.gof import graph
from theano.gof.utils import MethodNotDefined
from theano.compat import PY3

try:
    import pygpu
    from pygpu import gpuarray, elemwise
except ImportError:
    pass

from .type import GpuArrayType, gpu_context_type, get_context

# This is a marker to indicate that any context is acceptable
# It should never be registered as a context name
AnyContext = object()

def as_gpuarray_variable(x, context):
    # This is needed to lower the number of useless transfer
    # introduced during optimization.  This speed up optimization and
    # "canonicalize" the graph, so it make easier making some
    # optimization.
    if hasattr(x, '_as_GpuArrayVariable'):
        return x._as_GpuArrayVariable(context)

    x = tensor.as_tensor_variable(x)

    if (x.owner and
        isinstance(x.owner.op, HostFromGpu) and
        context == x.owner.inputs[0].type.context):
        return x.owner.inputs[0]

    return GpuFromHost(context)(x)


def infer_context(*vars):
    "Infer the context to use from the inputs given"

    # We try to infer the closest context first
    # by doing a breadth-first search
    nextv = list(vars)
    while len(nextv) != 0:
        v = nextv.pop(0)
        if isinstance(v.type, GpuArrayType):
            return v.type.context
        if hasattr(v.tag, 'context'):
            return v.tag.context
        if v.owner:
            for i in v.owner.inputs:
                if isinstance(i.type, GpuArrayType):
                    return i.type.context
                if isinstance(i.type, TensorType):
                    nextv.append(i)
    import pdb; pdb.set_trace()
    raise ValueError("couldn't infer context")


class HideC(object):
    def __hide(*args):
        raise MethodNotDefined()

    c_code = __hide
    c_code_cleanup = __hide

    c_headers = __hide
    c_header_dirs = __hide
    c_libraries = __hide
    c_lib_dirs = __hide

    c_support_code = __hide
    c_support_code_apply = __hide

    c_compile_args = __hide
    c_no_compile_args = __hide
    c_init_code = __hide
    c_init_code_apply = __hide

    c_init_code_struct = __hide
    c_support_code_struct = __hide
    c_cleanup_code_struct = __hide

    def c_code_cache_version(self):
        return ()

    def c_code_cache_version_apply(self, node):
        return self.c_code_cache_version()


class Kernel(object):
    """
    This class groups together all the attributes of a gpu kernel.
    """
    def __init__(self, code, params, name, flags,
                 codevar=None, binvar=None, objvar=None):
        self.code = code
        self.params = params
        self.name = name
        self.flags = flags
        if codevar is None:
            codevar = 'kcode_' + name
        self.codevar = codevar
        if binvar is None:
            binvar = 'kbin_' + name
        self.binvar = binvar
        if objvar is None:
            objvar = 'k_' + name
        self.objvar = objvar

    @staticmethod
    def get_flags(*types):
        def get_dtype(t):
            if isinstance(t, (str, unicode)):
                return numpy.dtype(t)
            elif isinstance(t, Type):
                return t.dtype
            elif isinstance(t, Variable):
                return t.type.dtype
            else:
                raise TypeError, "can't get a dtype from %s" % (type(t),)
        dtypes = [get_dtype(t) for t in types]
        flags = dict(cluda=True)
        if any(d == numpy.float64 for d in dtypes):
            flags['have_double'] = True
        if any(d.itemsize < 4 for d in dtypes):
            flags['have_small'] = True
        if any(d.kind == 'c' for d in dtypes):
            flags['have_complex'] = True
        if any(d == numpy.float16 for d in dtypes):
            flags['have_half'] = True
        return flags

    def _get_c_flags(self):
        res = []
        if self.flags.get('cluda', False):
            res.append('GA_USE_CLUDA')
        if self.flags.get('have_double', False):
            res.append('GA_USE_DOUBLE')
        if self.flags.get('have_small', False):
            res.append('GA_USE_SMALL')
        if self.flags.get('have_complex', False):
            res.append('GA_USE_COMPLEX')
        if self.flags.get('have_half', False):
            res.append('GA_USE_SMALL')
        return '|'.join(res)

    def _get_c_types(self):
        def m(t):
            if t == gpuarray.GpuArray:
                return "GA_BUFFER"
            else:
                return str(gpuarray.dtype_to_typecode(t))
        return ', '.join(m(t) for t in self.params)


class GpuKernelBase(object):
    context_type = gpu_context_type

    def gpu_kernels(self, node, name):
        """
        This is the method to override.  This should return an
        iterable of Kernel objects that describe the kernels this op
        will need.
        """
        raise MethodNotDefined('gpu_kernels')

    def c_headers(self):
        try:
            o = super(GpuKernelBase, self).c_headers()
        except MethodNotDefined:
            o = []
        return o + ['gpuarray/types.h']

    def _generate_kernel_bin(self, k, ctx):
        gk = gpuarray.GpuKernel(k.code, k.name, k.params, context=ctx,
                                **k.flags)
        bin = gk._binary
        bcode = ','.join(hex(ord(c)) for c in bin)
        return ("""static const char %(bname)s[] = { %(bcode)s };""" %
                dict(bname=k.binvar, bcode=bcode))

    def _generate_kernel_code(self, k):
        code = '\\n'.join(l for l in k.code.split('\n'))
        code = code.replace('"', '\\"')
        return ("""static const char *%(cname)s = "%(code)s";""" %
                dict(cname=k.codevar, code=code))

    def _generate_kernel_vars(self, k):
        return """GpuKernel %(kname)s;""" % dict(kname=k.objvar)

    def c_support_code_apply(self, node, name):
        kernels = self.gpu_kernels(node, name)
        ctx = get_context(self.get_context(node))
        bins = '\n'.join(self._generate_kernel_bin(k, ctx)
                         for k in kernels)
        codes = '\n'.join(self._generate_kernel_code(k) for k in kernels)
        return '\n'.join([bins, codes])


    def c_support_code_struct(self, node, name):
        kernels = self.gpu_kernels(node, name)
        vars = '\n'.join(self._generate_kernel_vars(k) for k in kernels)
        return vars

    def _generate_zeros(self, k):
        return """memset(&%(v)s, 0, sizeof(%(v)s));""" % dict(v=k.objvar)

    def _generate_kernel_init(self, k, fail, ctx):
        return """{
  int err;
  int types[%(numargs)u] = {%(types)s};
  const char *bcode = %(bvar)s;
  size_t sz = sizeof(%(bvar)s);
  if (GpuKernel_init(&%(ovar)s, %(ctx)s->ops, %(ctx)s->ctx, 1, &bcode, &sz,
                     "%(kname)s", %(numargs)u, types, GA_USE_BINARY)
      != GA_NO_ERROR) {
    if ((err = GpuKernel_init(&%(ovar)s, %(ctx)s->ops, %(ctx)s->ctx, 1,
                              &%(cname)s, NULL, "%(kname)s", %(numargs)u,
                              types, %(flags)s)) != GA_NO_ERROR) {
      PyErr_Format(PyExc_RuntimeError, "GpuKernel_init error %%d: %%s",
                   err, Gpu_error(%(ctx)s->ops, %(ctx)s->ctx, err));
      %(fail)s
    }
  }
}""" % dict(numargs=len(k.params), types=k._get_c_types(), bvar=k.binvar,
            ovar=k.objvar, kname=k.name, cname=k.codevar,
            flags=k._get_c_flags(), fail=fail, ctx=ctx)

    def c_init_code_struct(self, node, name, sub):
        ctx = sub['context']
        kernels = self.gpu_kernels(node, name)
        inits_0 = '\n'.join(self._generate_zeros(k) for k in kernels)
        inits = '\n'.join(self._generate_kernel_init(k, sub['fail'], ctx)
                          for k in kernels)
        return '\n'.join([inits_0, inits])

    def _generate_kernel_cleanup(self, k):
        return """GpuKernel_clear(&%(ovar)s);""" % dict(ovar=k.objvar)

    def c_cleanup_code_struct(self, node, name):
        kernels = self.gpu_kernels(node, name)
        cleanups = '\n'.join(self._generate_kernel_cleanup(k) for k in kernels)
        return cleanups

    def _GpuKernelBase_version(self):
        return (2,0,6)

    GpuKernelBase_version = property(_GpuKernelBase_version)


class HostFromGpu(Op):
    __props__ = ()

    def __str__(self):
        return 'HostFromGpu(gpuarray)'

    def make_node(self, x):
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        return Apply(self, [x],
                     [tensor.TensorType(dtype=x.dtype,
                                        broadcastable=x.broadcastable)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = numpy.asarray(x)

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        GpuArray %(name)s_ga_s;
        GpuArray *%(name)s_ga = NULL;
        int %(name)serr;
        PyArray_Descr *%(name)s_dtype;
        if (!GpuArray_ISONESEGMENT(&%(inp)s->ga)) {
            if (GpuArray_copy(&%(name)s_ga_s, &%(inp)s->ga, GA_C_ORDER) != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Can't make contiguous copy");
                %(fail)s;
            }
            %(name)s_ga = &%(name)s_ga_s;
        } else {
            %(name)s_ga = &%(inp)s->ga;
        }
        %(name)s_dtype = typecode_to_dtype(%(name)s_ga->typecode);
        Py_XDECREF(%(out)s);
        // PyArray_Empty below steals a reference to the dtype we pass it
        // so we need an extra one to spare.
        Py_INCREF(%(name)s_dtype);
        %(out)s = (PyArrayObject *)PyArray_Empty(%(inp)s->ga.nd,
                                (npy_intp *)%(inp)s->ga.dimensions,
                                %(name)s_dtype,
                                (%(inp)s->ga.flags & GA_F_CONTIGUOUS) &&
                                !(%(inp)s->ga.flags & GA_C_CONTIGUOUS));
        if (%(out)s == NULL) {
            if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
            %(fail)s
        }
        %(name)serr = GpuArray_read(PyArray_DATA(%(out)s),
                                    PyArray_NBYTES(%(out)s),
                                    %(name)s_ga);
        if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
        if (%(name)serr != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Could not read device data.");
            %(fail)s
        }
        """ % {'name': name, 'fail': sub['fail'], 'inp': inputs[0],
               'out': outputs[0]}

    def c_code_cache_version(self):
        return (1,)

    def grad(self, inputs, grads):
        gz, = grads
        return [GpuFromHost(inputs[0].type.context)(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, tensor.TensorType):
            return [GpuFromHost(inputs[0].type.context)(ev)]
        else:
            return [ev]

    def infer_shape(self, node, xshp):
        return xshp

host_from_gpu = HostFromGpu()


class GpuFromHost(Op):
    __props__ = ('context',)

    context_type = gpu_context_type

    def __init__(self, context):
        self.context = context

    def __str__(self):
        return 'GpuFromHost<%s>' % (self.context,)

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype,
                                              context=self.context)()])

    def get_context(self, node):
        return self.context

    def perform(self, node, inp, out, ctx):
        x, = inp
        z, = out
        z[0] = gpuarray.array(x, context=ctx)

    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(as_gpuarray_variable(gz, self.context))]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, GpuArrayType):
            return [host_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_fromhostdata(PyArray_DATA(%(inp)s),
                                     get_typecode((PyObject *)PyArray_DESCR(%(inp)s)),
                                     PyArray_NDIM(%(inp)s),
                                     (size_t *)PyArray_DIMS(%(inp)s),
                                     (ssize_t *)PyArray_STRIDES(%(inp)s),
                                     %(ctx)s,
                                     Py_None);
        if (%(out)s == NULL) {
            %(fail)s
        }
        """ % {'name': name, 'inp': inputs[0], 'ctx': sub['context'],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (4, 1)


class GpuFromGpu(Op):
    __props__ = ('context',)

    context_type = gpu_context_type

    def __init__(self, context):
        self.context = context

    def make_node(self, x):
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype,
                                              context=self.context)()])

    def get_context(self, node):
        return self.context

    def perform(self, node, inp, out, ctx):
        x, = inp
        z, = out
        z[0] = x.transfer(ctx)

    def grad(self, inputs, grads):
        gz, = grads
        return [GpuFromGpu(inputs[0].type.context)(gz)]

    def R_op(self, inputs, eval_points):
        return self.grad(inputs, eval_points)

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_transfer(%(inp)s, %(ctx)s, 0);
        if (%(out)s == NULL) {
            %(fail)s
        }
        """ % {'inp': inputs[0], 'ctx': sub['context'],
               'out': outputs[0], 'fail': sub['fail']}

    def c_code_cache_version(self):
        return (0,)


class GpuFromCuda(Op):
    __props__ = ('context',)

    view_map = {0: [0]}

    context_type = gpu_context_type

    def __init__(self, context):
        self.context = context

    def __str__(self):
        return 'GpuFromCuda<%s>' % (self.context,)

    def make_node(self, x):
        from theano.sandbox.cuda import CudaNdarrayType
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
                                              dtype=x.dtype,
                                              context=self.context)()])

    def get_context(self, node):
        return self.context

    def perform(self, node, inp, out, ctx):
        x, = inp
        z, = out
        z[0] = gpuarray.array(numpy.asarray(x), context=ctx)

    def grad(self, inputs, grads):
        gz, = grads
        return [cuda_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, GpuArrayType):
            return [cuda_from_gpu(ev)]
        else:
            return ev

    def infer_shape(self, node, xshp):
        return xshp

    def c_headers(self):
        return ['<cuda_ndarray.cuh>', '<gpuarray/extension.h>',
                '<gpuarray/types.h>', '<cuda.h>']

    def c_header_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def c_lib_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_libraries(self):
        return ['cudart', 'cublas', 'cuda']

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx);
        gpudata *(*cuda_make_buf)(void *c, CUdeviceptr p, size_t sz);
        """

    def c_init_code(self):
        return ['cuda_get_ctx = (CUcontext (*)(void *))gpuarray_get_extension("cuda_get_ctx");',
                'cuda_make_buf = (gpudata *(*)(void *, CUdeviceptr, size_t))gpuarray_get_extension("cuda_make_buf");']

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        int %(name)serr;
        gpudata *%(name)sdata;
        CUcontext %(name)scur;
        size_t *%(name)sdims;
        ssize_t *%(name)sstr;

        cuCtxGetCurrent(&%(name)scur);
        if (%(name)scur != cuda_get_ctx(%(ctx)s->ctx)) {
            PyErr_SetString(PyExc_ValueError, "Ambient cuda context is not the same as output context.");
            %(fail)s
        }
        %(name)sdims = (size_t *)calloc(%(inp)s->nd, sizeof(size_t));
        if (%(name)sdims == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Can't allocate dimensions.");
            %(fail)s
        }
        %(name)sstr = (ssize_t *)calloc(%(inp)s->nd, sizeof(ssize_t));
        if (%(name)sstr == NULL) {
            free(%(name)sdims);
            PyErr_SetString(PyExc_MemoryError, "Can't allocate strides.");
            %(fail)s
        }

        for (unsigned int i = 0; i < %(inp)s->nd; i++) {
            %(name)sdims[i] = (size_t)CudaNdarray_HOST_DIMS(%(inp)s)[i];
            %(name)sstr[i] = (ssize_t)CudaNdarray_HOST_STRIDES(%(inp)s)[i]*4;
        }

        %(name)sdata = cuda_make_buf(%(ctx)s->ctx,
                                     (CUdeviceptr)%(inp)s->devdata,
                                     ((size_t)%(inp)s->data_allocated)*4);
        if (%(name)sdata == NULL) {
            Py_DECREF(%(out)s);
            free(%(name)sdims);
            free(%(name)sstr);
            PyErr_SetString(PyExc_MemoryError, "Could not allocate gpudata structure.");
            %(fail)s
        }
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_fromgpudata(%(name)sdata, 0, GA_FLOAT, %(inp)s->nd,
                                    %(name)sdims, %(name)sstr,
                                    %(ctx)s, 1,
                                    (PyObject *)%(inp)s,
                                    (PyObject *)&PyGpuArrayType);
        %(ctx)s->ops->buffer_release(%(name)sdata);
        free(%(name)sdims);
        free(%(name)sstr);
        if (%(out)s == NULL) {
            %(fail)s
        }
        """ % dict(name=name, inp=inputs[0], out=outputs[0],
                   ctx=sub['context'], fail=sub['fail'])

    def c_code_cache_version(self):
        return (5, 1)


class CudaFromGpu(Op):
    __props__ = ()
    view_map = {0: [0]}

    def __str__(self):
        return 'CudaFromGpu'

    def make_node(self, x):
        from theano.sandbox.cuda import CudaNdarrayType
        if not isinstance(x.type, GpuArrayType):
            raise TypeError(x)
        if x.type.dtype != 'float32':
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable)()])

    def perform(self, node, inp, out):
        from theano.sandbox.cuda import filter as cuda_filter
        x, = inp
        z, = out
        z[0] = cuda_filter(theano._asarray(x, dtype='float32'),
                           tuple([0] * x.ndim), 0, z[0])

    def grad(self, inputs, grads):
        gz, = grads
        return [GpuFromCuda(inputs[0].type.context)(gz)]

    def R_op(self, inputs, eval_points):
        from theano.sandbox.cuda import CudaNdarrayType
        ev, = eval_points
        if (isinstance(ev, CudaNdarrayType)):
            return [gpu_from_cuda(ev)]
        else:
            return [ev]

    def infer_shape(self, node, shp):
        return shp

    def c_headers(self):
        return ['<cuda_ndarray.cuh>', '<gpuarray/extension.h>', '<cuda.h>']

    def c_header_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def c_lib_dirs(self):
        import cuda_ndarray
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'lib'))
        return ret

    def c_libraries(self):
        return ['cudart', 'cublas', 'cuda']

    def c_support_code(self):
        return """
        CUcontext (*cuda_get_ctx)(void *ctx);
        CUdeviceptr (*cuda_get_ptr)(gpudata *g);
        """

    def c_init_code(self):
        return ['cuda_get_ctx = (CUcontext (*)(void *ctx))gpuarray_get_extension("cuda_get_ctx");',
                'cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))gpuarray_get_extension("cuda_get_ptr");']

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        int %(name)serr = 0, %(name)si;
        CUcontext %(name)scur;

        cuCtxGetCurrent(&%(name)scur);
        if (%(name)scur != cuda_get_ctx(%(inp)s->context->ctx)) {
            PyErr_SetString(PyExc_ValueError, "Ambiant context is not the same as input context.");
            %(fail)s
        }

        if (GpuArray_sync(&%(inp)s->ga) != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Could not sync GpuArray");
            %(fail)s
        }
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray *)CudaNdarray_new_nd(%(inp)s->ga.nd);
        if (!%(out)s) {
            %(fail)s
        }
        for (%(name)si = 0; %(name)si < %(inp)s->ga.nd; %(name)si++) {
            CudaNdarray_set_dim(%(out)s, %(name)si, %(inp)s->ga.dimensions[%(name)si]);
            CudaNdarray_set_stride(%(out)s, %(name)si, %(inp)s->ga.strides[%(name)si]/4);
        }
        %(name)serr = CudaNdarray_set_device_data(%(out)s,
          (float *)(((char *)cuda_get_ptr(%(inp)s->ga.data))+%(inp)s->ga.offset),
                                          (PyObject *)%(inp)s);
        if (%(name)serr) {
           %(fail)s
        }
        """ % {'name': name, 'inp': inputs[0], 'out': outputs[0],
               'fail': sub['fail']}

    def c_code_cache_version(self):
        return (3,)

cuda_from_gpu = CudaFromGpu()


class GpuAlloc(HideC, Alloc):
    __props__ = ('memset_0', 'context') + Alloc.__props__

    context_type = gpu_context_type

    def __init__(self, context, memset_0=False):
        """memset_0 is only an optimized version. True, it mean the
        value is always 0, so the c code call memset as it is faster.
        """
        self.context = context
        self.memset_0 = memset_0

    def __str__(self):
        #Hide the memset parameter when not used to prevent confusion.
        if self.memset_0:
            s = "%s<%s>{memset_0=%s}" % (self.__class__.__name__,
                                         self.context, self.memset_0)
        else:
            s = "%s<%s>" % (self.__class__.__name__, self.context)
        return s

    def make_node(self, value, *shape):
        res = Alloc.make_node(self, value, *shape)
        value = as_gpuarray_variable(value, context=self.context)
        otype = GpuArrayType(dtype=res.outputs[0].dtype,
                             broadcastable=res.outputs[0].broadcastable,
                             context=self.context)
        return Apply(self, [value] + res.inputs[1:], [otype()])

    def get_context(self, node):
        return self.context

    def c_headers(self):
        return ['<numpy_compat.h>']

    def perform(self, node, inputs, outs, ctx):
        out, = outs
        v = inputs[0]
        sh = tuple(map(int, inputs[1:]))
        if out[0] is None or out[0].shape != sh:
            if v.size == 1 and numpy.asarray(v)[0].item() == 0:
                out[0] = gpuarray.zeros(sh, dtype=v.dtype, context=ctx)
            else:
                out[0] = gpuarray.empty(sh, dtype=v.dtype, context=ctx)
                out[0][...] = v
        else:
            out[0][...] = v
        if config.gpuarray.sync:
            out[0].sync()

    def c_code(self, node, name, inp, out, sub):
        vv = inp[0]
        ndim = len(inp[1:])
        zz, = out

        memset_0 = int(self.memset_0)
        code = """
        int i;
        size_t %(name)s_shape[%(ndim)s];
        """ % dict(name=name, ndim=ndim)

        for i, shp_i in enumerate(inp[1:]):
            code += """
        %(name)s_shape[%(i)s] = ((dtype_%(shp_i)s *)PyArray_DATA(%(shp_i)s))[0];
        """ % dict(name=name, i=i, shp_i=shp_i)

        code += """
        int need_new_out = (NULL == %(zz)s || %(zz)s->ga.nd != %(ndim)s);

        if (!need_new_out)
            for (i = 0; i < %(ndim)s; i++)
                need_new_out |= %(zz)s->ga.dimensions[i] != %(name)s_shape[i];

        if (need_new_out && (%(memset_0)s)) {
            //pygpu_zeros can be faster then empty followed by memset.
            Py_XDECREF(%(zz)s);
            %(zz)s = pygpu_zeros(%(ndim)s, %(name)s_shape,
                                 %(vv)s->ga.typecode, GA_C_ORDER,
                                 %(ctx)s, Py_None);
            if (!%(zz)s) {
                %(fail)s
            }
        } else {
            if (need_new_out) {
                Py_XDECREF(%(zz)s);
                %(zz)s = pygpu_empty(%(ndim)s, %(name)s_shape,
                                     %(vv)s->ga.typecode, GA_C_ORDER,
                                     %(ctx)s, Py_None);
                if (!%(zz)s) {
                    %(fail)s
                }
            }
            if (%(memset_0)s && GpuArray_ISONESEGMENT(&%(zz)s->ga))
            {
                int err = GpuArray_memset(&%(zz)s->ga, 0);
                if (err != GA_NO_ERROR)
                {
                    PyErr_Format(PyExc_MemoryError,
                                 "GpuAlloc: Error memsetting %%d"
                                 " element of device memory to 0.",
                                 PyGpuArray_SIZE(%(zz)s));
                    %(fail)s;
                }
            }
            else if (GpuArray_setarray(&%(zz)s->ga, &%(vv)s->ga) !=
                     GA_NO_ERROR) {
                PyErr_SetString(PyExc_ValueError, "setarray failed");
                %(fail)s
            }
        }
        """ % dict(name=name, ndim=ndim, zz=zz, vv=vv, fail=sub['fail'],
                   memset_0=memset_0, ctx=sub['context'])

        if config.gpuarray.sync:
            code += "GpuArray_sync(&%(zz)s->ga);" % dict(zz=zz)

        return code

    def c_code_cache_version(self):
        return (2,)

    def do_constant_folding(self, node):
        for client in node.outputs[0].clients:
            if client[0] == 'output':
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            elif (#The following ops work inplace of their input id 0.
                  client[1] == 0 and
                  isinstance(client[0].op, (
                    #Ops that will work inplace on the Alloc. So if they
                    #get constant_folded, they would copy the
                    #constant and this is less efficients.

                    #Not doing the constant folding could also lower
                    #the peak memory usage, as we the "constant" won't
                    #always exists.
                      #theano.tensor.subtensor.AdvancedIncSubtensor,
                      theano.sandbox.gpuarray.subtensor.GpuIncSubtensor,
                      theano.sandbox.gpuarray.subtensor.GpuAdvancedIncSubtensor1,
                      theano.sandbox.gpuarray.subtensor.GpuAdvancedIncSubtensor1_dev20,
                      theano.sandbox.gpuarray.blas.GpuGemm,
                      theano.sandbox.gpuarray.blas.GpuGemv,
                      theano.sandbox.gpuarray.blas.GpuGer,
                  ))):
                return False
            #If the clients is a transfer, we don't want to fold. We
            #let the moving opt finish before deciding what to do.
            elif isinstance(client[0].op, HostFromGpu):
                return False
        return True


class GpuContiguous(Op):
    """
    Always return a c contiguous output. Copy the input only if it is
    not already c contiguous.
    """
    view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def grad(self, inputs, dout):
        x, = inputs
        dout, = dout
        dout = as_gpuarray_variable(dout, context=infer_context(x))

        return [dout]

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, input):
        input = as_gpuarray_variable(input, context=infer_context(input))
        return Apply(self, [input], [input.type()])

    def c_headers(self):
        return ['<numpy_compat.h>']

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        z, = out
        fail = sub['fail']
        str = """
        {
            if (GpuArray_IS_C_CONTIGUOUS(&(%(input)s->ga))){
                Py_XDECREF(%(z)s);
                %(z)s = %(input)s;
                Py_INCREF(%(z)s);

            } else if ((NULL == %(z)s)""" % locals()
        for i in xrange(len(node.inputs[0].type.broadcastable)):
            str += "\n|| (PyGpuArray_DIMS(%(input)s)[%(i)s] != PyGpuArray_DIMS(%(z)s)[%(i)s])" % locals()
        str += """
                || !GpuArray_IS_C_CONTIGUOUS(&(%(z)s->ga)))
            {
                Py_XDECREF(%(z)s);
                %(z)s = pygpu_copy(%(input)s, GA_C_ORDER);
                if (!%(z)s)
                {
                    %(fail)s;
                }
            }else if(pygpu_move(%(z)s, %(input)s) == -1) {
                %(fail)s;
            }
        }
        """ % locals()
        return str

gpu_contiguous = GpuContiguous()


class GpuReshape(HideC, tensor.Reshape):
    """
    Implement Reshape on the gpu.
    """
    # __hash__, __eq__, __str__ come from tensor.Reshape
    def make_node(self, x, shp):
        ctx = infer_context(x)
        x = as_gpuarray_variable(x, context=ctx)
        res = host_from_gpu(x).reshape(shp, ndim=self.ndim)
        otype = GpuArrayType(dtype=res.dtype,
                             broadcastable=res.broadcastable,
                             context=ctx)
        return Apply(self, [x, shp], [otype()])

    def perform(self, node, inp, out_):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to GpuReshape.perform'
                             ' has incorrect length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)

        if shp.prod() != x.size:
            # We need to do check here to raise the same error as NumPy.
            # We should make pygpu do the same.
            ss = 1
            nb_m1 = 0
            for i in shp:
                if i == -1:
                    nb_m1 += 1
                else:
                    ss *= i
            if nb_m1 > 1:
                raise ValueError("Only one -1 is accepted in the new shape")
            elif nb_m1 == 1:
                if (x.size % ss) != 0:
                    raise ValueError("When using -1 in new shape, the computed new shape must be an multiple of the original shape.")
            else:
                raise ValueError("total size of new array must be unchanged")
        out[0] = x.reshape(tuple(shp))


class GpuJoin(HideC, Join):
    __props__ = ('context',)

    context_type = gpu_context_type

    def __init__(self, context):
        self.context = context

    def make_node(self, axis, *tensors):
        node = Join.make_node(self, axis, *tensors)

        def agv(v):
            return as_gpuarray_variable(v, self.context)

        return Apply(self, [node.inputs[0]] + map(agv, tensors),
                     [GpuArrayType(broadcastable=node.outputs[0].broadcastable,
                                   dtype=node.outputs[0].dtype,
                                   context=self.context)()])

    def get_context(self, node):
        return self.context

    def perform(self, node, axis_and_tensors, out_, ctx):
        out, = out_
        axis = int(axis_and_tensors[0])
        tensors = axis_and_tensors[1:]
        out[0] = pygpu.concatenate(tensors, axis=axis).astype(
            node.outputs[0].dtype, context=ctx)

    def c_code_cache_version(self):
        return (2)

    def c_code(self, node, name, inputs, out_, sub):
        copy_to_list = []
        restype=pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        for i, inp in enumerate(inputs[1:]):
            copy_to_list.append("als[%s] = &%s->ga;" % (i, inp))
        return """
const GpuArray **als = (const GpuArray **)PyMem_Malloc(sizeof(GpuArray *) * %(n)s);
if (als == NULL) {
  PyErr_NoMemory();
  %(fail)s
}
%(copy_inputs_to_list)s
Py_XDECREF(%(out)s);
%(out)s = pygpu_concatenate(als, %(n)s, PyInt_AsLong((PyObject *)%(axis)s),
                            %(restype)s, (PyObject *)&PyGpuArrayType,
                            %(ctx)s);
PyMem_Free(als);
if (%(out)s == NULL)
  %(fail)s
        """ % dict(n=len(inputs[1:]), fail=sub['fail'], out=out_[0],
                   axis=inputs[0], copy_inputs_to_list='\n'.join(copy_to_list),
                   restype=restype, ctx=sub['context'])



class GpuSplit(HideC, Split):
    def make_node(self, x, axis, splits):
        node = Split.make_node(self, x, axis, splits)
        x = as_gpuarray_variable(x, infer_context(x))
        outs = [GpuArrayType(dtype=o.dtype, broadcastable=o.broadcastable,
                             context=x.type.context)()
                for o in node.outputs]
        return Apply(self, [x] + node.inputs[1:], outs)
    # we reuse the perform of the CPU op, which is suitable


class GpuEye(GpuKernelBase, Op):
    __props__ = ('dtype', 'context')

    def __init__(self, dtype=None, context=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype
        self.context = context

    def get_context(self, node):
        return self.context

    def make_node(self, n, m, k):
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        k = tensor.as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0
        otype = GpuArrayType(dtype=self.dtype,
                             broadcastable=(False, False),
                             context=self.context)

        # k != 0 isn't implemented on the GPU yet.
        assert tensor.get_scalar_constant_value(k) == 0
        return Apply(self, [n, m], [otype()])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i])
                for i in xrange(3)]

    def gpu_kernels(self, node, name):
        code = """
KERNEL void k(GLOBAL_MEM %(ctype)s *a, ga_size n, ga_size m) {
    ga_size nb = n < m ? n : m;
    for (ga_size i = LID_0; i < nb; i += LDIM_0) {
        a[i*m + i] = 1;
    }
}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype), name=name)
        return [Kernel(
                code=code, name="k",
                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE],
                flags=Kernel.get_flags(self.dtype),
                objvar='k_eye_'+name,
                )]

    def c_code(self, node, name, inp, out, sub):
        n, m = inp
        z, = out
        fail = sub['fail']
        ctx = sub['context']
        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
        sync = bool(config.gpuarray.sync)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        size_t dims[2] = {0, 0};
        void *args[3];
        int err;

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        Py_CLEAR(%(z)s);

        %(z)s = pygpu_zeros(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            %(ctx)s, Py_None);
        if (%(z)s == NULL) {
            %(fail)s
        }

        args[0] = &%(z)s->ga;
        args[1] = &dims[0];
        args[2] = &dims[1];
        err = GpuKernel_call(&%(kname)s, 0, 1, 256, args);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "gpuarray error: kEye: %%s. n=%%lu, m=%%lu.",
                         GpuKernel_error(&%(kname)s, err),
                         (unsigned long)dims[0], (unsigned long)dims[1]);
            %(fail)s;
        }

        if(%(sync)d)
            GpuArray_sync(&%(z)s->ga);
        """ % locals()

        return s

    def c_code_cache_version(self):
        return (3, 1, self.GpuKernelBase_version)
