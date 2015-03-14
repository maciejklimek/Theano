import theano
from theano import gof
from theano.compile.function_module import orig_function
from theano.compile import SharedVariable, rebuild_collect_shared
from theano.gof import ops_with_inner_function


class Loop(gof.Op):
    """This creates a loop from inputs and outputs lists of variables.

    :param inputs: list of inputs to loop over

    :param outputs: list of output expressions

    :param others: other variables that will be used to compute outputs.
        Shared variables and constants must not be part of this list.
    """
    def __init__(self, inputs, outputs, others=None,
                 input_hints=None, output_hints=None):
        self._i = theano.shared(numpy.asarray(0, dtype='uint64'))
        if others is None:
            others = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i in inputs + outputs + others:
            if not isinstance(i, gof.Variable):
                raise TypeError(
                        'inputs and outputs must be Variable instances', i)

        if any(isinstance(i, SharedVariable) for i in inputs):
            raise ValueError("Inputs can't be shared variables "
                             "for the inner graph")

        if any(isinstance(o, SharedVariable) for o in others):
            raise ValueError("Don't pass shared variables in others, "
                             "they will be handled automatically")
        if any(isinstance(o, gof.Constant) for o in others):
            raise ValueError("Don't pass constants in others, "
                             "they will be handled automatically")

        self.inputs = inputs
        self.others = others
        self.outputs = outputs

        # To correctly support shared variables the inner function
        # must not see them. Otherwise it becomes impossible to
        # compute the gradient.  So we collect them here to hide them
        # later.
        self.shared = [var for var in gof.graph.inputs(outputs)
                       if isinstance(var, SharedVariable)]

        if input_hints is None:
            self.input_hints = [inp.type.clone(
                    broadcastable(False,) + inp.type.broadcastable)()
                           for inp in inputs]
        else:
            # We don't want to reuse the passed-in variable, just its type
            self.input_hints = [inp.clone() for inp in input_hints]
            assert len(self.input_hints) == len(self.inputs)

        if output_hints is None:
            self.output_hints = [out.type.clone(
                    broadcastable=(False,) + out.type.broadcastable)()
                                 for out in outputs]
        else:
            self.output_hints = [out.clone() for out in output_hints]
            assert len(self.output_hints) == len(self.outputs)

    def make_func(self):
        shared_g = [var.type() for var in self.shared]
        inputs_g = [inp[self._i] for inp in self.input_hints]
        outputs_g = [theano.tensor.set_subtensor(out_h[self._i], out)
                     for out_h, out in zip(self.output_hints, outputs)]

        new = rebuild_collect_shared(outputs_g,
                                     inputs=(inputs + others +
                                             self.output_models +
                                             self.shared),
                                     replace=dict(zip(self.shared + inputs,
                                                      shared_g + inputs_g)),
                                     copy_inputs_over=False,
                                     rebuild_strict=True)
        (new_inputs, new_outputs,
         [clone_d, update_d, update_expr, shared_inputs]) = new
        assert len(new_inputs) == (len(inputs) + len(others) +
                                   len(output_models) + len(self.shared))
        assert len(new_outputs) == len(outputs)
        assert shared_inputs == [self._i]

        f_inputs = new_inputs
        f_outputs = new_outputs

        fn = function(
            f_inputs, f_outputs,
            updates=[(self._i, self._i+numpy.asarray(1, dtype='int64'))],
            mode=Mode(linker=VM_Linker(allow_gc=False, use_cloop=True)))

        return fn, f_inputs, f_outputs

    def __eq__(self, other):
        #TODO: recognize a copy
        return self is other

    def __hash__(self):
        #TODO: use internal variables in hash
        return hash(type(self))

    def make_node(self, n_iters, *vars):
        # Check that the number of iterations is a scalar
        assert n_iters.ndim == 0
        assert n_iters.type.dtype == 'int64'

        # First in vars is all the inputs which are iterated
        inputs = vars[:len(self.input_hints)]
        if len(inputs) != len(self.input_hints):
            raise ValueError("Not enough inputs")
        for oi, ii in zip(inputs, self.inputs_hints):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for input, expected %s but got %s"
                                % (ii.type, oi.type))
        # After that is the others
        others = vars[len(self.input_hints):len(self.others)]
        if len(others) != len(self.others):
            raise ValueError("Not enough others")
        for oi, ii in zip(others, self.others):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for other, expected %s but got %s"
                                % (ii.type, oi.type))

        # Finally we have the output buffers
        outputs = vars[len(self.input_hints)+len(self.others):
                           len(self.outputs)]
        if len(outputs) != len(self.output_hints):
            raise ValueError("Not enough outputs")
        for oi, ii in zip(outputs, self.output_hints):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for output, expected %s but got %s"
                                % (ii.type, oi.type))

        if len(vars) > (len(self.inputs_hints) + len(self.others) +
                        len(self.output_hints)):
            raise TypeError("Too many arguments for Loop operation")

        return gof.Apply(self,
                         # tackle on the end of our inputs the list of
                         # shared variables we will need for easy
                         # access.
                         [n_iters] + list(vars) + self.shared,
                         [o.clone() for o in self.output_hints])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        # XXX: maybe do some hocus pocus to share storage_map

        # Although this wouldn't be safe to share for more than one
        # graph we would just have to return a unique thunk from here.
        if not hasattr(self, "fn"):
            self.fn = self.make_func()

        ret = super(Loop, self).make_thunk(node, storage_map,
                                           compute_map, no_recycling)
        return ret

    def perform(self, node, inputs, outputs):
        self._i.set_value(0)
        for c, v in zip(self.fn.inputs, inputs[1:]):
            c.storage[0] = v
        self.fn.fn(n_calls=inputs[0])
        assert len(self.fn.outputs) == len(outputs)
        for o, c in zip(outputs, self.fn.outputs):
            o[0] = c.storage[0]

# Since OpFromGraph contains a Theano compiled function, we should let
# DebugMode know about it
ops_with_inner_function[OpFromGraph] = 'fn'
