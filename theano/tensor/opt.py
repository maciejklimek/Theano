"""Tensor optimizations addressing the ops in basic.py
"""
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0


from .. import gof
from ..gof import opt, InconsistencyError, TopoOptimizer, graph
from elemwise import Elemwise, DimShuffle
from .. import scalar
import basic as T
import inplace as I
import numpy as N
import operator
import itertools
import sys
from .. import compile  #to register the optimizer built by this file


# Utilities

def out2in(*local_opts):
    """WRITEME """
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order = 'out_to_in',
                             failure_callback=TopoOptimizer.warn_inplace)

def in2out(*local_opts, **kwargs):
    """WRITEME """
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order = 'in_to_out',
                             failure_callback=TopoOptimizer.warn_inplace,
                             **kwargs)



@gof.optimizer
def insert_inplace_optimizer(env):
    """
    Usage: inplace_optimizer.optimize(env)
    
    Attempts to replace all Broadcast ops by versions of them
    that operate inplace. It operates greedily: for each Broadcast
    Op that is encountered, for each output, tries each input to
    see if it can operate inplace on that input. If so, makes the
    change and go to the next output or Broadcast Op.

    Examples:
      x + y + z -> x += y += z
      (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)
    """
    for node in list(graph.io_toposort(env.inputs, env.outputs)):
        op = node.op
        if not isinstance(op, Elemwise):
            continue
        baseline = op.inplace_pattern
        candidate_outputs = [i for i in xrange(len(node.outputs)) if i not in baseline]
        candidate_inputs = [i for i in xrange(len(node.inputs)) if i not in baseline.values()]
        for candidate_output in candidate_outputs:
            for candidate_input in candidate_inputs:
                inplace_pattern = dict(baseline, **{candidate_output: candidate_input})
                try:
                    new = Elemwise(
                        op.scalar_op.__class__(
                            scalar.transfer_type(
                                *[inplace_pattern.get(i, None) \
                                        for i in xrange(len(node.outputs))])),
                        inplace_pattern).make_node(*node.inputs)
                    env.replace_all_validate(zip(node.outputs, new.outputs),
                            reason="insert_inplace_optimizer")
                except (ValueError, TypeError, InconsistencyError), e:
                    continue
                candidate_inputs.remove(candidate_input)
                node = new
                baseline = inplace_pattern
                break
compile.optdb.register('inplace_opt', insert_inplace_optimizer, 75, 'fast_run', 'inplace') 

def register_canonicalize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['canonicalize'].register(name, lopt, 'fast_run', *tags)

def register_specialize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['specialize'].register(name, lopt, 'fast_run', *tags)

######################
# DimShuffle lifters #
######################

@gof.local_optimizer([None, None])
def local_dimshuffle_lift(node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.
    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise):
        return inode.op.make_node(*[DimShuffle(input.type.broadcastable,
                                               op.new_order,
                                               op.inplace)(input) for input in inode.inputs]).outputs
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == 'x' and 'x' or inode.op.new_order[x] for x in op.new_order]
        inplace = op.inplace and inode.op.inplace
        iinput = inode.inputs[0]
        if new_order == range(len(new_order)):
            return [iinput]
        else:
            return DimShuffle(iinput.type.broadcastable, new_order, inplace).make_node(iinput).outputs

register_canonicalize(local_dimshuffle_lift)



#################
# Shape lifters #
#################

@gof.local_optimizer([T.shape, None])
def local_shape_lift_elemwise(node):
    """
    shape(elemwise_op(..., x, ...)) -> shape(x)

    Where x contains the maximal shape information.
    """
    if not opt.check_chain(node, T.shape, T.Elemwise):
        return False
    
    output = node.inputs[0]
    parent = output.owner

    for input in parent.inputs:
        if input.type.broadcastable == output.type.broadcastable:
            return T.shape.make_node(input).outputs

    return False

register_canonicalize(local_shape_lift_elemwise, 'shape_lift')


@gof.local_optimizer([T.shape, None])
def local_shape_lift_sum(node):
    """
    shape(sum{n}(x)) -> [shape(x)[0], ..., shape(x)[n-1], shape(x)[n+1], ...]
    """
    if not opt.check_chain(node, T.shape, T.Sum):
        return False

    input = node.inputs[0].owner.inputs[0]
    axis = node.inputs[0].owner.op.axis
    if axis is None:# or len(axis) != 1:
        axis = range(input.type.ndim)


    ish = T.shape(input)
    return T.make_lvector.make_node(*(ish[i] for i in xrange(input.type.ndim) if i not in axis)).outputs
#    return T.vertical_stack.make_node(ish[:axis], ish[axis+1:]).outputs

register_canonicalize(local_shape_lift_sum, 'shape_lift')


@gof.local_optimizer([T.shape, T.dot])
def local_shape_lift_dot(node):
    """
    shape(dot(a, b)) -> [shape(a)[0], shape(b)[1]]
    """
    if not opt.check_chain(node, T.shape, T.dot):
        return False
    a, b = node.inputs[0].owner.inputs
    return T.make_lvector.make_node(T.shape(a)[0], T.shape(b)[1]).outputs

register_canonicalize(local_shape_lift_dot, 'shape_lift')


# local_shape_lift = opt.LocalOptGroup(local_shape_lift_elemwise,
#                                      local_shape_lift_sum,
#                                      local_shape_lift_dot)


################
# Fill lifters #
################

def encompasses_broadcastable(b1, b2):
    """
    Returns True if the broadcastable patterns b1 and b2 are such that b2 is
    broadcasted to b1's shape and not the opposite.

    :param b1: the broadcastable attribute of a tensor type
    :param b2: the broadcastable attribute of a tensor type
    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2):]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))

def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]

@gof.local_optimizer([T.fill, None])
def local_fill_lift(node):
    """
    fill(f(a), b) -> fill(a, b)
    If a.type == f(a).type.

    fill(a, b) -> b
    If a.type == b.type.
    """
    if not opt.check_chain(node, T.fill):
        return False

    model, filling = node.inputs

    mb, fb = model.type.broadcastable, filling.type.broadcastable
    if model.type.dtype == filling.type.dtype and encompasses_broadcastable(fb, mb):
        return False# [filling]

    parent = model.owner
    if parent is None or not isinstance(parent, T.Elemwise):
        return False
    for input in parent.inputs:
        if input.type == model.type:
            return [T.fill(input, filling)]

    return False

register_canonicalize(local_fill_lift, 'fill_lift')


##################
# Subtensor opts #
##################


@gof.local_optimizer([None, None])
def local_subtensor_make_vector(node):
    """
    [a,b,c][0] -> a
    [a,b,c][0:2] -> [a,b]

    If the index or slice is constant.
    """
    if not opt.check_chain(node, T.Subtensor, T.MakeVector):
        return False

    joined_r = node.inputs[0]

    try: 
        #check that join is being used to join scalars
        veclen = T.join.vec_length(joined_r)
    except:
        return False

    idxlist = node.op.idx_list
    if len(idxlist) != 1:
        return False
    idx = idxlist[0]
    if isinstance(idx, int):
        return [node.inputs[0].owner.inputs[idx]]
    try:
        return T.make_vector(*(node.owner.inputs[0].owner.inputs.__getslice__(idx)))
    except TypeError:
        return False

register_canonicalize(local_subtensor_make_vector)

#after priority 50 Destructive inplace operations
#gemm is the first one now, at priority 70

@gof.local_optimizer([None])
def local_inplace_setsubtensor(node):
    if isinstance(node.op, T.SetSubtensor) and not node.op.inplace:
        new_op = T.SetSubtensor(node.op.idx_list, inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('inplace_setsubtensor', TopoOptimizer(local_inplace_setsubtensor,
    failure_callback=TopoOptimizer.warn_inplace), 60, 'fast_run', 'inplace') #DEBUG

##################
# Reshape opts   #
##################


@gof.local_optimizer([None, None])
def local_reshape_chain(node):
    """
    Reshape(Reshape(shape1),shape2) -> Reshape(shape2)
    """
    if not opt.check_chain(node, T.Reshape, T.Reshape):
        return False
    
    return [node.op(node.inputs[0].owner.inputs[0], node.inputs[1])]
register_canonicalize(local_reshape_chain)


##################
# Middleman cuts #
##################

@gof.local_optimizer([None, T.fill])
def local_fill_cut(node):
    """
    f(fill(a,b), c) -> f(b, c)
    If c.type == a.type.
    """

    if not opt.check_chain(node, T.Elemwise):
        return False
    
    output = node.outputs[0]
    try:
        reference = [input
                     for input in node.inputs
                     if input.type == output.type and (not input.owner or input.owner.op != T.fill)][0]
    except IndexError:
        return False

    new_inputs = []
    for input in node.inputs:
        if opt.check_chain(input, T.fill):
            model, filling = input.owner.inputs
            if encompasses_broadcastable(reference.type.broadcastable,
                                         filling.type.broadcastable):
                new_inputs.append(filling)
                continue
        new_inputs.append(input)

    if new_inputs == node.inputs:
        return False
    return node.op.make_node(*new_inputs).outputs

register_canonicalize(local_fill_cut)

register_canonicalize(gof.OpRemove(T.tensor_copy), name='remove_tensor_copy' )

@gof.local_optimizer([None, T.fill])
def local_fill_sink(node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(a, fill(c, f(b, d, e)))
    """
    if not (node.op and isinstance(node.op, T.Elemwise) and node.op != T.fill):
        return False
    models = []
    inputs = []
    for input in node.inputs:
        if input.owner and input.owner.op == T.fill:
            models.append(input.owner.inputs[0])
            inputs.append(input.owner.inputs[1])
        else:
            inputs.append(input)
    if inputs == node.inputs:
        return False
    c = node.op(*inputs)
    for model in models:
        c = T.fill(model, c)
    return [c]

register_canonicalize(local_fill_sink)


################
# Canonization #
################

class Canonizer(gof.LocalOptimizer):
    """
    Simplification tool.

    Usage: Canonizer(main, inverse, reciprocal, calculate)
    
    * main: a suitable Op class that is commutative, associative and
            takes one to an arbitrary number of inputs, e.g. add or
            mul
    * inverse: an Op class such that inverse(main(x, y), y) == x
               e.g. sub or div
    * reciprocal: a function such that main(x, reciprocal(y)) ==
                  inverse(x, y) e.g. neg or inv

    * calculate: function that takes a list of numpy.ndarray instances
                 for the numerator, another list for the denumerator,
                 and calculates inverse(main(*num), main(*denum)). It
                 takes a keyword argument, aslist. If True, the value
                 should be returned as a list of one element, unless
                 the value is such that value = main(). In that case,
                 the return value should be an empty list.

    The result is a local_optimizer. It is best used with a TopoOptimizer in
    in_to_out order.

    Examples:
      T = theano.tensor
      add_canonizer = Canonizer(T.add, T.sub, T.neg, lambda n, d: sum(n) - sum(d))
      mul_canonizer = Canonizer(T.mul, T.div, T.inv, lambda n, d: prod(n) / prod(d))
    
    Examples of optimizations mul_canonizer can perform:
      x / x -> 1
      (x * y) / x -> y
      x / y / x -> 1 / y
      x / y / z -> x / (y * z)
      x / (y / z) -> (x * z) / y
      (a / b) * (b / c) * (c / d) -> a / d
      (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
      2 * x / 2 -> x
    """

    def __init__(self, main, inverse, reciprocal, calculate, use_reciprocal = True):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

    def tracks(self):
        return [[self.main, None], [self.inverse, None], [self.reciprocal, None]]

    def get_num_denum(self, input):
        """
        This extract two lists, num and denum, such that the input is:
        self.inverse(self.main(*num), self.main(*denum)). It returns
        the two lists in a (num, denum) pair.

        For example, for main, inverse and reciprocal = *, / and inv(),

        input -> returned value (num, denum)

        x*y -> ([x, y], [])
        inv(x) -> ([], [x])
        inv(x) * inv(y) -> ([], [x, y])
        x*y/z -> ([x, y], [z])
        log(x) / y * (z + x) / y -> ([log(x), z + x], [y, y])
        (((a / b) * c) / d) -> ([a, c], [b, d])
        a / (b / c) -> ([a, c], [b])
        log(x) -> ([log(x)], [])
        x**y -> ([x**y], [])
        """

        if input.owner is None or input.owner.op not in [self.main, self.inverse, self.reciprocal]:
            if input.owner and isinstance(input.owner.op, T.DimShuffle):
                # If input is a DimShuffle of some input which does something like this:
                # * change a vector of length N into a 1xN row matrix
                # * change a scalar into a 1x1x1 tensor
                # * in general, complete the shape of a tensor with broadcastable 1s to the *left*
                # Then we will simply discard the DimShuffle and return the num/denum of its input
                dsn = input.owner    # dimshuffle node
                dsop = dsn.op        # dimshuffle op
                dsi0 = dsn.inputs[0] # the first input of the dimshuffle i.e. the ndarray to redim

                # The compatible order is a DimShuffle "new_order" of the form:
                # ('x', ..., 'x', 0, 1, 2, ..., dimshuffle_input.type.ndim)

                # That kind of DimShuffle only adds broadcastable
                # dimensions on the left, without discarding any
                # existing broadcastable dimension and is inserted
                # automatically by Elemwise when the inputs have
                # different numbers of dimensions (hence why we can
                # discard its information - we know we can retrieve it
                # later on).
                compatible_order = ('x',) * (input.type.ndim - dsi0.type.ndim) + tuple(range(dsi0.type.ndim))
                if dsop.new_order == compatible_order:
                    # If the "new_order" is the one we recognize,
                    # we return the num_denum of the dimshuffled input.
                    return self.get_num_denum(input.owner.inputs[0])
                else:
                    # This is when the input isn't produced by main, inverse or reciprocal.
                    return [input], []
            else:
                return [input], []
        num = []
        denum = []
        parent = input.owner

        # We get the (num, denum) pairs for each input
        pairs = [self.get_num_denum(input) for input in parent.inputs]

        if parent.op == self.main:
            # If we have main(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, numy) and denum is concat(denumx, denumy)
            # note that main() can have any number of arguments >= 0
            # concat is list concatenation
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            # If we have inverse(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, denumy) and denum is concat(denumx, numy)
            # note that inverse() is binary
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            # If we have reciprocal(x), numx, denumx
            # then num is denumx and denum is numx
            # note that reciprocal() is unary
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        """
        Utility function which takes two lists, num and denum, and
        returns something which is equivalent to inverse(main(*num),
        main(*denum)), but depends on the length of num and the length
        of denum (in order to minimize the number of operations).

        Let n = len(num) and d = len(denum):

        n=0, d=0: neutral element (given by self.calculate([], []))
                  (for example, this would be 0 if main is addition
                  and 1 if main is multiplication)
        n=1, d=0: num[0]
        n=0, d=1: reciprocal(denum[0])
        n=1, d=1: inverse(num[0], denum[0])
        n=0, d>1: reciprocal(main(*denum))
        n>1, d=0: main(*num)
        n=1, d>1: inverse(num[0], main(*denum))
        n>1, d=1: inverse(main(*num), denum[0])
        n>1, d>1: inverse(main(*num), main(*denum))

        Given the values of n and d to which they are associated, all
        of the above are equivalent to:
        inverse(main(*num), main(*denum))
        """

        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return T.as_ndarray_result(self.calculate([], []))
        if not ln:
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist = False)]
        if not ld:
            if ln == 1:
                if isinstance(num[0], gof.Result):
                    return num[0]
                else:
                    return T.as_ndarray_result(num[0])
            else:
                return self.main(*num)
        return self.inverse(self.merge_num_denum(num, []),
                            self.merge_num_denum(denum, []))

    @classmethod
    def get_constant(cls, v):
        """

        Returns a numeric constant if v is a gof.Constant or, well, a
        numeric constant. If v is a plain Result, returns None.

        """
        if isinstance(v, N.generic):
            return v # doesn't the not hasattr() condition below catch this?
        if isinstance(v, gof.Constant):
            return v.data
        if not hasattr(v, 'owner'):
            return v

#         NOTE: the following code was buggy, but while I was fixing
#         it I realized it is probably made useless by constant
#         folding, so screw that. Commented-out code is the half-fixed
#         version.

#         if v.owner and isinstance(v.owner.op, DimShuffle):
#             # see the comments in get_num_denum
#             # TODO: this should apply the 
#             dsn = v.owner
#             dsop = dsn.op
#             dsi0 = dsn.inputs[0]
#             compatible_order = ('x',) * (input.type.ndim - dsi0.type.ndim) + tuple(range(dsi0.type.ndim))
#             if dsop.new_order == compatible_order:
#                 return cls.get_constant(v.owner.inputs[0])

        return None

    def simplify(self, num, denum):
        """
        Shorthand for: self.simplify_constants(*self.simplify_factors(num, denum))
        """
        return self.simplify_constants(*self.simplify_factors(num, denum))

    def simplify_factors(self, num, denum):
        """
        For any Result r which is both in num and denum, removes it
        from both lists. Modifies the lists inplace. Returns the
        modified lists. For example:

        [x], [x] -> [], []
        [x, y], [x] -> [y], []
        [a, b], [c, d] -> [a, b], [c, d]
        """
        for v in list(num):
            if v in denum:
                num.remove(v)
                denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum):
        """

        Finds all constants in orig_num and orig_denum (using
        get_constant) and puts them together into a single
        constant. The constant is inserted as the first element of the
        numerator. If the constant is the neutral element, it is
        removed from the numerator. Examples:

        Let main be multiplication:

        [2, 3, x], [] -> [6, x], []
        [x, y, 2], [4, z] -> [0.5, x, y], [z]
        [x, 2, y], [z, 2] -> [x, y], [z]
        """

        # Lists representing the numerator and denumerator
        num, denum = list(orig_num), list(orig_denum)

        # Lists representing the *constant* elements of num and denum
        numct, denumct = [], []
        
        for v in orig_num:
            ct = self.get_constant(v)
            if ct is not None:
                # We found a constant in the numerator!
                # We remove it from num
                num.remove(v)
                # We add it to numct
                numct.append(ct)
        for v in orig_denum:
            ct = self.get_constant(v)
            if ct is not None:
                denum.remove(v)
                denumct.append(ct)

        if self.use_reciprocal or num:
            # This will calculate either:
            # [inverse(main(*numct), main(*denumct))]
            # [] - if inverse(main(*numct), main(*denumct)) is the neutral element
            ct = self.calculate(numct, denumct, aslist = True)
        else:
            # This happens if we don't allow the reciprocal and the
            # numerator is empty. That means we will need to represent
            # reciprocal(x) like inverse(neutral_element, x) so
            # we can't allow ct == []
            # TODO: why is this branch needed when merge_num_denum does it for us?
            ct = [self.calculate(numct, denumct, aslist = False)]
        # TODO: why are we not wrapping ct in a gof.Constant right now?

        if orig_num and len(numct) == 1 and len(denumct) == 0 and ct and N.all(ct == self.get_constant(orig_num[0])):
            # this is an important trick :( if it so happens that:
            # * there's exactly one constant on the numerator and none on the denominator
            # * it's not the neutral element (ct is an empty list in that case)
            # * the constant is the same as the first argument in the numerator
            # Then we return very exactly the original num/denum
            # If we don't do that the optimizer will just loop infinitely because
            # it will not catch on that there are no changes to be made and everytime
            # it will want to replace something by the same thing...
            return orig_num, orig_denum
        return ct + num, denum

    def transform(self, node):
        op = node.op
        inputs = node.inputs
        out = node.outputs[0]
        if op not in [self.main, self.inverse, self.reciprocal]:
            return False

        # I'm not sure if this is actually needed but the following
        # block of code puts into "reorg" whether or not we are going
        # to change the structure of the graph. For example if we have
        # inverse operating on an inverse, we can make it so that only
        # one inverse is used, so we'll reorganize that.
        iops = set(input.owner.op for input in inputs if input.owner)
        reorg = False
        if op == self.main:
            reorg = len(iops.intersection([self.main, self.inverse, self.reciprocal])) != 0
        elif op == self.inverse:
            reorg = len(iops.intersection([self.inverse, self.reciprocal])) != 0
        elif op == self.reciprocal:
            reorg = len(iops.intersection([self.inverse, self.reciprocal])) != 0

        # just in case
        assert len(node.outputs) == 1

        # Here we make the canonical version of the graph around this node
        # See the documentation of get_num_denum and simplify
        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = list(orig_num), list(orig_denum)
        num, denum = self.simplify(num, denum)

        def same(x, y):
            return len(x) == len(y) and all(N.all(xe == ye) for xe, ye in zip(x, y))

        if not reorg and same(orig_num, num) and same(orig_denum, denum):
            # We return False if there are no changes
            # TODO: what's the purpose of reorg? isn't same() sufficient?
            return False

        new = self.merge_num_denum(num, denum)
        if new.dtype != out.dtype:
            #new = T.fill(out, new)
            elem_op = T.Elemwise(scalar.Identity(scalar.specific_out(getattr(scalar, out.type.dtype))))
            new = T.fill(out, elem_op(new))

        if new.broadcastable != out.broadcastable:
            #this case is tricky... we need to provide exactly the same kind of broadcastable
            #pattern, but only if legal...
            dlen = len(new.broadcastable) - len(out.broadcastable)

            if dlen > 0:
                #try to take the leading ranks of new.broadcastable, which should be broadcastable
                # ranks
                #if this means skipping over nonbroadcastable ranks, then DimShuffle will fail
                dimshuffle_op = T.DimShuffle(new.broadcastable, 
                        range(dlen, len(new.broadcastable)))
                new = dimshuffle_op(new)
            elif dlen < 0:
                #we have to boost up a scalar or something
                dimshuffle_op = T.DimShuffle(new.broadcastable, 
                        ['x' for x in range(-dlen)] + range(0, len(new.broadcastable)))
                new = dimshuffle_op(new)

        # if our if's above worked, this should be true. OTW investigate.
        if new.type != out.type:
            print >> sys.stderr, 'CANONIZE FAILED: new out = ', new, out
            assert new.type == out.type

        return [new]

    def __str__(self):
        return getattr(self, 'name', 'Canonizer(%s, %s, %s)' % (self.main, self.inverse, self.reciprocal))


def mul_calculate(num, denum, aslist = False):
    v = reduce(N.multiply, num, 1.0) / reduce(N.multiply, denum, 1.0)
    if aslist:
        if N.all(v == 1):
            return []
        else:
            return [v]
    return v

local_mul_canonizer = Canonizer(T.mul, T.div, T.inv, mul_calculate, False)
register_canonicalize(local_mul_canonizer, name = 'local_mul_canonizer')

@gof.local_optimizer([T.neg])
def local_neg_to_mul(node):
    if node.op == T.neg:
        return [T.mul(-1, node.inputs[0])]
    else:
        return False
register_canonicalize(local_neg_to_mul)

@gof.local_optimizer([T.mul])
def local_mul_to_neg(node):
    if node.op == T.mul and N.all(local_mul_canonizer.get_constant(node.inputs[0]) == -1.0):
        return [-local_mul_canonizer.merge_num_denum(node.inputs[1:], [])]
    else:
        return False
register_specialize(local_mul_to_neg)

@gof.local_optimizer([T.div])
def local_div_to_inv(node):
    if node.op == T.div and N.all(local_mul_canonizer.get_constant(node.inputs[0]) == 1.0):
        return [T.inv(local_mul_canonizer.merge_num_denum(node.inputs[1:], []))]
    else:
        return False
register_specialize(local_div_to_inv)

@gof.local_optimizer([T.inv])
def local_inv_canon(node):
    if node.op == T.inv:
        return [T.pow(node.inputs[0], -1.0)]
    else:
        return False
register_canonicalize(local_inv_canon)

@gof.local_optimizer([T.pow])
def local_pow_canonicalize(node):
    if node.op == T.pow:
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 1.0):
            return [T.fill(node.inputs[1], node.inputs[0])]
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 0.0):
            #extra fills here are to make sure the size of the output stays constant.
            return [T.fill(node.inputs[0], T.fill(node.inputs[1], 1.0))]
    else:
        return False
register_canonicalize(local_pow_canonicalize)

@gof.local_optimizer([T.pow])
def local_pow_specialize(node):
    #here, we are past the point of canonicalization, so we don't want to put in un-necessary fills.
    if node.op == T.pow:
        #the idea here is that we have pow(x, y)
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable, ysym.type.broadcastable):
            if N.all(y == 2.0):
                return [T.sqr(xsym)]
            if N.all(y == 1.0):
                return [xsym]
            if N.all(y == 0.0):
                return [T.fill(xsym, 1.0)]
            if N.all(y == 0.5):
                return [T.sqrt(xsym)]
            if N.all(y == -0.5):
                return [T.inv(T.sqrt(xsym))]
            if N.all(y == -1.0):
                return [T.inv(xsym)]
            if N.all(y == -2.0):
                return [T.inv(T.sqr(xsym))]
    else:
        return False
register_specialize(local_pow_specialize)

@gof.local_optimizer([T.mul])
def local_mul_specialize(node):
    #here, we are past the point of canonicalization, so we don't want to put in un-necessary fills.
    if node.op == T.mul:
        #the idea here is that we have pow(x, y)
        neg = False
        new_inputs = []
        for input in node.inputs:
            y = local_mul_canonizer.get_constant(input)
            if N.all(y == 1.0):
                continue
            elif N.all(y == -1.0):
                neg ^= True #toggles
            elif N.all(y == 0.0):
                return [input]
            else:
                new_inputs.append(input)
        if len(new_inputs) < len(node.inputs):
            if len(new_inputs) == 0:
                newval = -y.flatten()[0] if neg else y.flatten()[0]
                return [T.NDArrayConstant(T.NDArrayType(dtype=node.outputs[0].type.dtype,
                    broadcastable = [True] * node.outputs[0].ndim), N.asarray(newval))]

            if len(new_inputs) == 1:
                return [-new_inputs[0]] if neg else new_inputs
            else:
                return [-T.mul(*new_inputs)] if neg else \
                        [T.mul(*new_inputs)] 
    else:
        return False
register_specialize(local_mul_specialize)



# neg_to_mul = out2in(gof.LocalOptGroup(local_neg_to_mul))
# mul_to_neg = out2in(gof.LocalOptGroup(local_mul_to_neg))

mul_canonizer = in2out(gof.LocalOptGroup(local_mul_canonizer, local_fill_cut, local_fill_sink))



def add_calculate(num, denum, aslist = False):
    v = reduce(N.add, num, 0.0) - reduce(N.add, denum, 0.0)
    if aslist:
        if N.all(v == 0):
            return []
        else:
            return [v]
    return v

local_add_canonizer = Canonizer(T.add, T.sub, T.neg, add_calculate)
add_canonizer = in2out(gof.LocalOptGroup(local_add_canonizer, local_fill_cut, local_fill_sink))

register_canonicalize(local_add_canonizer, name = 'local_add_canonizer')


##################
# Distributivity #
##################


def distribute_greedy(pos_pairs, neg_pairs, num, denum, minscore = 0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    score = len(num) + div_cost * len(denum) # score is number of operations saved, higher is better
    new_pos_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n+num, d+denum) for (n, d) in pos_pairs]))
    new_neg_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n+num, d+denum) for (n, d) in neg_pairs]))
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs + new_neg_pairs):
        # We calculate how many operations we are saving with the new num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs

def attempt_distribution(factor, num, denum):
    # we try to insert each num and each denum in the factor
    # returns: changes?, new_factor, new_num, new_denum
    # if there are changes, new_num and new_denum contain all the numerators
    # and denumerators that could not be distributed in the factor
    pos, neg = local_add_canonizer.get_num_denum(factor)
    if len(pos) == 1 and not neg:
        return False, factor, num, denum
    pos_pairs = map(local_mul_canonizer.get_num_denum, pos)
    neg_pairs = map(local_mul_canonizer.get_num_denum, neg)
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs, neg_pairs, [n], [])
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs, neg_pairs, [], [d])
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return change, local_add_canonizer.merge_num_denum(
            list(itertools.starmap(local_mul_canonizer.merge_num_denum, pos_pairs)),
            list(itertools.starmap(local_mul_canonizer.merge_num_denum, neg_pairs))), num, denum

@gof.local_optimizer([T.mul, T.add, T.mul], [T.mul, T.sub, T.mul],
                     [T.mul, T.add, T.div], [T.mul, T.sub, T.div])
def local_greedy_distributor(node):
    """
    This optimization tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ((a/x + b/y) * x * y) --> a*y + b*x
    2. ((a/x + b) * x) --> a + b*x

    The following expressions are not simplified:
    3. ((a + b) * x) -/-> a*x + b*x

    This optimization aims to reduce computational cost. It may also
    increase numerical stability, e.g. when x and/or y tend to 0 in
    example 1.
    """

    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(candidate, num, denum)
        change |= _change
        new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(candidate, denum, num)
        change |= _change
        new_denum.append(candidate)

    if not change:
        return False

    new_num += num
    new_denum += denum

    rval = local_mul_canonizer.merge_num_denum(new_num, new_denum)

    if rval.type != out.type:  
        #WHY DOES THIS HAPPEN?
        return False

    return [rval]

register_canonicalize(local_greedy_distributor)



@gof.local_optimizer([None])
def constant_folding(node):
    for input in node.inputs:
        if not isinstance(input, gof.Constant):
            return False
    storage = [[None] for output in node.outputs]
    node.op.perform(node, [x.data for x in node.inputs], storage)
    return [gof.Constant(output.type, s[0]) for s, output in zip(storage, node.outputs)]

register_canonicalize(constant_folding)


inplace_matrix_transpose = T.DimShuffle([False,False], [1,0], inplace=True)
local_transposed_dot = gof.PatternSub((inplace_matrix_transpose, (T.dot, 'x', 'y')),
        (T.dot, (inplace_matrix_transpose, 'y'), (inplace_matrix_transpose, 'x')))
register_canonicalize(local_transposed_dot, name='local_transposed_dot')

# ###############
# # Loop fusion #
# ###############

# def make_composite(inputs, outputs):
#     scalar_inputs = [scalar.Scalar(dtype = i.type.dtype)() for i in inputs]
#     def transform(r):
#         if r in inputs:
#             return scalar_inputs[inputs.index(r)]
#         node = r.owner
#         if node is None:
#             if isinstance(r, gof.Constant):
#                 if r.data.size == 1:
#                     return gof.Constant(scalar.Scalar(dtype = r.type.dtype), r.data)
#                 else:
#                     return scalar.Scalar(dtype = r.type.dtype)
#             else:
#                 print r, inputs
#                 raise Exception('bluh')
#                 #return scalar.Scalar(dtype = r.type.dtype)
#         elif isinstance(node.op, DimShuffle):
#             new_r = transform(node.inputs[0])
#         elif isinstance(node.op, Elemwise):
#             new_r = node.op.scalar_op(*map(transform, node.inputs))
#         else:
#             raise Exception('bluh2')
#         return new_r
#     scalar_outputs = map(transform, outputs)
#     return scalar.Composite(scalar_inputs, scalar_outputs)

# def loop_fusion(env):

#     def grab(node, out, seen, grabbed, inputs, outputs):
#         if node in grabbed:
#             return True
#         if node is None or isinstance(node, str) or node in seen:
#             return False
#         seen.add(node)
#         if node and isinstance(node.op, Elemwise) and node.outputs[0].type.broadcastable == out.type.broadcastable:
#             grabbed.add(node)
#             for output in node.outputs:
#                 output_is_temp = True
#                 for node2, i in output.clients:
#                     grab(node2, out, seen, grabbed, inputs, outputs)
#                     if node2 not in grabbed:
#                         output_is_temp = False
#                 if not output_is_temp:
#                     outputs.add(output)
#             for input in node.inputs:
#                 node2 = input.owner
#                 grab(node2, out, seen, grabbed, inputs, outputs)
#                 if node2 not in grabbed:
#                     inputs.add(input)
#             return True
#         elif node and isinstance(node.op, DimShuffle):
#             input = node.inputs[0]
#             if node.op.new_order[-input.type.ndim:] == range(input.type.ndim):
#                 inputs.add(input)
#                 grabbed.add(node)
#             return True
#             #return grab(node.inputs[0].owner, out, seen, grabbed, inputs, outputs)
#         else:
#             return False
                
#     #for node in list(env.toposort()): # reversed(list(env.toposort())):
#     for node in reversed(list(env.toposort())):
#         if node in env.nodes and isinstance(node.op, Elemwise):
#             inputs = set()
#             outputs = set() # set(node.outputs)
#             out = node.outputs[0]
#             grab(node, out, set(), set(), inputs, outputs)
#             if inputs == set(node.inputs) and outputs == set(node.outputs):
#                 continue
#             print 'AAAAAAAAAAA', [__i in outputs for __i in inputs]
#             inputs, outputs = list(inputs), list(outputs)
#             composite = make_composite(inputs, outputs)
#             #print composite
#             #print gof.Env(*gof.graph.clone(inputs, outputs))
#             new_node = Elemwise(composite).make_node(*inputs)
#             print 'yea!!!!!', len(new_node.inputs), len(new_node.outputs)
#             print new_node.outputs[0].type, outputs[0].type
#             env.replace_all_validate(zip(outputs, new_node.outputs))
#             print env
#             gof.graph.io_toposort(env.inputs, env.outputs)










            

# compile.optdb.register('merge1.5', gof.MergeOptimizer(), 97, 'fast_run')
# loop_fusion = gof.optimizer(loop_fusion)
# compile.optdb.register('loop_fusion', loop_fusion, 98, 'fast_run')











#     def grab_up(input, out, seen, inputs, outputs):
#         if input in seen:
#             return
#         seen.add(input)
#         node = input.owner
#         if node and isinstance(node.op, Elemwise) and input.type.broadcastable == out.type.broadcastable:
#             for input in node.inputs:
#                 grab_up(input, out, seen, inputs, outputs)
#             for output in node.outputs:
#                 grab_down(output, out, seen, inputs, outputs)
#         elif node and isinstance(node.op, DimShuffle):
#             grab_up(node.inputs[0], out, seen, inputs, outputs)
#         else:
#             inputs.add(input)

#     def grab_down(r, out, seen, inputs, outputs):
#         for node, i in r.clients:
#             if isinstance(node, str):
#                 outputs.add(r)
#             elif isinstance(node.op, Elemwise) and node.outputs[0].type.broadcastable == out.type.broadcastable:
#                 for input in node.inputs:
#                     grab_up(input, out, seen, inputs, outputs)
#                 for output in node.outputs:
#                     grab_down(output, out, seen, inputs, outputs)
#             else:
#                 outputs.add(r)
                
#     for node in reversed(list(env.toposort())):
#         if node in env.nodes and isinstance(node.op, Elemwise):
#             inputs = set()
#             outputs = set(node.outputs)
#             out = node.outputs[0]
#             for input in node.inputs:
#                 grab_up(input, out, set(), inputs, outputs)
#             if inputs == set(node.inputs) and outputs == set(node.outputs):
#                 continue
#             print 'AAAAAAAAAAA', [__i in outputs for __i in inputs]
#             inputs, outputs = list(inputs), list(outputs)
#             composite = make_composite(inputs, outputs)
#             print composite
#             print gof.Env(*gof.graph.clone(inputs, outputs))
#             new_node = Elemwise(composite).make_node(*inputs)
#             print 'yea!!!!!', len(new_node.inputs), len(new_node.outputs)
#             env.replace_all_validate(zip(outputs, new_node.outputs))


# add(mul(input, neg(softplus(neg(*2 -> add(<Tensor(float64, matrix)>, InplaceDimShuffle{x,0}(b2)))))), mul(*1 -> sub([[ 1.]], input), neg(softplus(*2))))
# sub(neg(mul(input, Elemwise{sub}([[ 1.]], *3 -> Elemwise{scalar_sigmoid}(*2)))), neg(mul(*1, *3)))



# #     for input in node.inputs:
# #         i, o = grab_up(input, out)
# #         inputs += i
# #         outputs += o

# # aaaaaaaaaaaaaaa
# #             i, o = [], []
# #             for output in node.outputs:
# #                 results = grab_down(output, out)
# # #                 if results is None:
# # #                     return [input], []
# #                 i += results[0]
# #                 o += results[1]
# #             return i, o




#             inputs = []
#             scalar_inputs = []
#             these_inputs = []
#             change = False
#             for input in node.inputs:
#                 owner = input.owner
#                 if input.type.broadcastable == node.outputs[0].type.broadcastable \
#                         and owner and isinstance(owner.op, Elemwise):
#                     new_inputs = [i.type.to_scalar_type()() for i in owner.inputs]
#                     inputs += owner.inputs
#                     scalar_inputs += new_inputs
#                     these_inputs.append(owner.op.scalar_op(*new_inputs))
#                     change = True
#                 #elif 
#                 else:
#                     inputs.append(input)
#                     scalar_input = input.type.to_scalar_type()()
#                     scalar_inputs.append(scalar_input)
#                     these_inputs.append(scalar_input)
#             if not change:
#                 return False
#             scalar_outputs = node.op.scalar_op.make_node(*these_inputs).outputs
#             new_scalar_op = scalar.Composite(scalar_inputs, scalar_outputs)
#             new_op = Elemwise(new_scalar_op)
#             new_node = new_op.make_node(*inputs)
#             ##print 'changed:', node, new_node.inputs
#             ##print 'new!!', new_node
#             print 'ding!', [input.type.broadcastable for input in new_node.inputs]
#             return new_node.outputs



# @gof.local_optimizer([None, None])
# def local_loop_fusion(node):
#     if not isinstance(node.op, Elemwise):
#         return False
#     ##print 'looking at:', node
#     inputs = []
#     scalar_inputs = []
#     these_inputs = []
#     change = False
#     for input in node.inputs:
#         owner = input.owner
#         if input.type.broadcastable == node.outputs[0].type.broadcastable \
#                 and owner and isinstance(owner.op, Elemwise):
#             new_inputs = [i.type.to_scalar_type()() for i in owner.inputs]
#             inputs += owner.inputs
#             scalar_inputs += new_inputs
#             these_inputs.append(owner.op.scalar_op(*new_inputs))
#             change = True
#         #elif 
#         else:
#             inputs.append(input)
#             scalar_input = input.type.to_scalar_type()()
#             scalar_inputs.append(scalar_input)
#             these_inputs.append(scalar_input)
#     if not change:
#         return False
#     scalar_outputs = node.op.scalar_op.make_node(*these_inputs).outputs
#     new_scalar_op = scalar.Composite(scalar_inputs, scalar_outputs)
#     new_op = Elemwise(new_scalar_op)
#     new_node = new_op.make_node(*inputs)
#     ##print 'changed:', node, new_node.inputs
#     ##print 'new!!', new_node
#     print 'ding!', [input.type.broadcastable for input in new_node.inputs]
#     return new_node.outputs

# loop_fusion = gof.EquilibriumOptimizer([local_loop_fusion], max_depth = 3, max_use_ratio = 1)
# compile.optdb.register('loop_fusion', loop_fusion, 98, 'fast_run')






