"""Exact algebraic simplification of column expressions.

The expression-level counterparts of the identity rewrites in ``_numeric_rewrites``: an
operation with its identity element is dropped (``x * 1``, ``x - 0``) and
``abs(abs(x))`` is one ``abs``. The input schemas must establish that values,
dtype, and the sign of zero are preserved: ``bool + 0``
is an int, ``int / 1`` a float, and ``-0.0 + 0`` is ``+0.0``.

The inverse and fused-function rewrites (``log(exp(x))``, ``exp(x) - 1``,
``sqrt(x * x)``) change float results, so they are not applied to expressions.
"""
from __future__ import annotations

from math import copysign
import operator

from stratum.optimizer._op_utils import replace_op_in_outputs, topological_iterator
from stratum.optimizer.logical._column_expr import (
    BinOpExpr, Col, ColumnExpr, Const, OperandLeaf, UfuncExpr, map_exprs)
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._ops import Op, OutputType
from stratum.optimizer.logical._selection_ops import MaskSelectionOp

#: Operator -> (its identity element, whether it may stand on the left too,
#: the AlgebraicRewritesConfig flag, whether an integer operand stays an integer).
_IDENTITIES = {
    operator.add: (0, True, "add_zero", True),
    operator.sub: (0, False, "identity_subtract", True),
    operator.mul: (1, True, "identity_op", True),
    operator.truediv: (1, False, "div_by_one", False),
    operator.pow: (1, False, "pow_by_one", True),
}


def simplify_expressions(root: Op, config) -> Op:
    """Simplify Map and predicate expressions using their explicit input schemas."""
    for op in list(topological_iterator(root)):
        if not isinstance(op, (ExpressionMapOp, MaskSelectionOp)):
            continue
        simplify = _Simplifier(op.inputs, config)
        for name in op.fields:
            setattr(op, name, map_exprs(getattr(op, name), simplify))
        if not isinstance(op, ExpressionMapOp):
            continue
        # Repeated abs calls remain separate logical maps before fusion.
        expr = op.expr
        if (config.abs_abs and isinstance(expr, UfuncExpr) and expr.func == "abs"
                and isinstance(expr.operand, OperandLeaf)):
            source = op.inputs[expr.operand.ref.k]
            if (isinstance(source, ExpressionMapOp) and isinstance(source.expr, UfuncExpr)
                    and source.expr.func == "abs"
                    and _numeric(_Simplifier(source.inputs, config).dtypes(source.expr.operand))):
                op.expr = expr.operand
        if isinstance(op.expr, OperandLeaf):
            source = op.inputs[op.expr.ref.k]
            if source.output_type is op.output_type:
                op.output_schema = source.output_schema
                if not (op.is_X or op.is_y):
                    replace_op_in_outputs(op, source)
                    op.detach_and_prune(keep=source)
                    if root is op:
                        root = source
    return root


def _numeric(dtypes):
    return bool(dtypes) and all(dtype.is_integer() or dtype.is_float() for dtype in dtypes)


class _Simplifier:
    """Simplifies an op's expressions bottom-up against its source's column dtypes.

    A node nothing simplifies is kept as it is, and each node is simplified once,
    so a subexpression the folder shared between entries stays one object.
    """

    def __init__(self, inputs, config):
        self.inputs = inputs
        self.config = config
        self._done: dict[int, ColumnExpr] = {}

    def __call__(self, expr: ColumnExpr) -> ColumnExpr:
        if id(expr) not in self._done:
            self._done[id(expr)] = self._simplify(expr)
        return self._done[id(expr)]

    def _simplify(self, expr: ColumnExpr) -> ColumnExpr:
        rebuilt = expr.map_children(self)
        if all(new is old for new, old in zip(rebuilt.children(), expr.children())):
            rebuilt = expr
        while True:
            simpler = self._rewrite(rebuilt)
            if simpler is None:
                return rebuilt
            rebuilt = simpler

    def _rewrite(self, expr: ColumnExpr) -> ColumnExpr | None:
        if (self.config.abs_abs and isinstance(expr, UfuncExpr) and expr.func == "abs"
                and isinstance(expr.operand, UfuncExpr) and expr.operand.func == "abs"
                and _numeric(self.dtypes(expr.operand.operand))):
            return expr.operand
        if isinstance(expr, BinOpExpr) and expr.op in _IDENTITIES:
            identity, commutes, flag, keeps_int = _IDENTITIES[expr.op]
            if not getattr(self.config, flag):
                return None
            sides = [(expr.left, expr.right)] + ([(expr.right, expr.left)] if commutes else [])
            for operand, constant in sides:
                if self._is_identity(operand, constant, identity, keeps_int, expr.op):
                    return operand
        return None

    def dtypes(self, operand):
        if isinstance(operand, Col):
            schema = self.inputs[0].output_schema
            return (schema[operand.name],) if schema is not None and operand.name in schema else ()
        if isinstance(operand, OperandLeaf):
            source = self.inputs[operand.ref.k]
            if source.output_type in (OutputType.SERIES, OutputType.FRAME):
                schema = source.output_schema
                return tuple(schema.values()) if schema is not None else ()
        return ()

    def _is_identity(self, operand, constant, identity, keeps_int, op) -> bool:
        """Whether ``operand <op> constant`` preserves values, dtype, and signed zero."""
        if not isinstance(constant, Const):
            return False
        dtypes = self.dtypes(operand)
        if not _numeric(dtypes):
            return False
        value = constant.value
        if type(value) is int:
            fits = all(dtype.is_float() or (dtype.is_integer() and keeps_int) for dtype in dtypes)
        elif type(value) is float:
            fits = all(dtype.is_float() for dtype in dtypes)
        else:
            return False
        # Adding positive zero, or subtracting negative zero, changes -0.0.
        if any(dtype.is_float() for dtype in dtypes):
            if op is operator.add:
                return False
            if op is operator.sub and type(value) is float and value == 0 and copysign(1.0, value) < 0:
                return False
        return fits and value == identity
