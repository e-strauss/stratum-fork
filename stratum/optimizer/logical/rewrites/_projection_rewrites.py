"""Remove assignment when its only selected column already has the base rows."""
from stratum.optimizer._op_utils import replace_op_in_outputs, topological_iterator
from stratum.optimizer.logical._base import IndexEffect, OperandRef, OutputType
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._metadata_ops import RenameOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp


def _has_base_rows(value, base):
    """Prove row correspondence through projections and expression maps only."""
    while value is not base:
        if not isinstance(value, (ColumnProjectionOp, ExpressionMapOp)):
            return False
        if value.index_effect is IndexEffect.KEEP:
            value = value.inputs[0]
        elif value.index_effect is IndexEffect.COMMON and len(value.inputs) == 1:
            value = value.inputs[0]
        else:
            return False
    return True


def simplify_projected_assignments(root):
    """``df.assign(z=expr)[['z']]`` becomes ``expr.to_frame('z')`` when aligned.

    Restrict this rule to one assigned Series and one selected column. Scalars,
    positional arrays, callable assignments, and independent Series retain the
    assignment's broadcasting or alignment semantics.
    """
    for op in list(topological_iterator(root)):
        if (not isinstance(op, ColumnProjectionOp) or not isinstance(op.key, list)
                or len(op.key) != 1 or op.is_X or op.is_y):
            continue
        assign = op.inputs[0]
        if (not isinstance(assign, AssignOp) or list(assign.kwargs) != op.key
                or assign.is_X or assign.is_y):
            continue
        name = op.key[0]
        ref = assign.kwargs[name]
        if not isinstance(ref, OperandRef):
            continue
        value = assign.inputs[ref.k]
        if (value.output_type is not OutputType.SERIES
                or not _has_base_rows(value, assign.inputs[0])):
            continue
        replacement = RenameOp(label=(name,), as_frame=True, inputs=[value])
        replacement.propagate_output_schema()
        value.add_output(replacement)
        replace_op_in_outputs(op, replacement)
        op.detach_and_prune(keep=replacement)
        if root is op:
            root = replacement
    return root
