# Linting is disabled on this file because the function is almost entirely copied from Qiskit code
# and should not be used as a long-term solution.

# flake8: noqa
# pylint: skip-file
import copy

from qiskit.circuit import ControlFlowOp, SwitchCaseOp, _classical_resource_map
from qiskit.circuit.classical import expr
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit.dagcircuit import (
    _additional_wires,
    _may_have_additional_wires,
)
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.dagcircuit.exceptions import DAGCircuitError


def substitute_node_with_dag(
    self, node, input_dag, wires=None, propagate_condition=True
):
    """Replace one node with dag.

    Args:
        node (DAGOpNode): node to substitute
        input_dag (DAGCircuit): circuit that will substitute the node.
        wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
            in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
            and the order gets matched to the node wires by qargs first, then cargs, then
            conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
            that the ``node`` acts on.

            Standalone :class:`~.expr.Var` nodes cannot currently be remapped as part of the
            substitution; the ``input_dag`` should be defined over the correct set of variables
            already.

            ..
                The rule about not remapping `Var`s is to avoid performance pitfalls and reduce
                complexity; the creator of the input DAG should easily be able to arrange for
                the correct `Var`s to be used, and doing so avoids us needing to recurse through
                control-flow operations to do deep remappings.
        propagate_condition (bool): If ``True`` (default), then any ``condition`` attribute on
            the operation within ``node`` is propagated to each node in the ``input_dag``.  If
            ``False``, then the ``input_dag`` is assumed to faithfully implement suitable
            conditional logic already.  This is ignored for :class:`.ControlFlowOp`\\ s (i.e.
            treated as if it is ``False``); replacements of those must already fulfill the same
            conditional logic or this function would be close to useless for them.

    Returns:
        dict: maps node IDs from `input_dag` to their new node incarnations in `self`.

    Raises:
        DAGCircuitError: if met with unexpected predecessor/successors
    """
    if not isinstance(node, DAGOpNode):
        raise DAGCircuitError(f'expected node DAGOpNode, got {type(node)}')

    if isinstance(wires, dict):
        wire_map = wires
    else:
        wires = input_dag.wires if wires is None else wires
        node_cargs = set(node.cargs)
        node_wire_order = list(node.qargs) + list(node.cargs)
        # If we're not propagating it, the number of wires in the input DAG should include the
        # condition as well.
        if not propagate_condition and _may_have_additional_wires(node):
            node_wire_order += [
                wire
                for wire in _additional_wires(node.op)
                if wire not in node_cargs
            ]
        if len(wires) != len(node_wire_order):
            raise DAGCircuitError(
                f'bit mapping invalid: expected {len(node_wire_order)}, got {len(wires)}'
            )
        wire_map = dict(zip(wires, node_wire_order))
        if len(wire_map) != len(node_wire_order):
            raise DAGCircuitError(
                'bit mapping invalid: some bits have duplicate entries'
            )
    for input_dag_wire, our_wire in wire_map.items():
        if our_wire not in self.input_map:
            raise DAGCircuitError(
                f'bit mapping invalid: {our_wire} is not in this DAG'
            )
        if isinstance(our_wire, expr.Var) or isinstance(
            input_dag_wire, expr.Var
        ):
            raise DAGCircuitError(
                '`Var` nodes cannot be remapped during substitution'
            )
        # Support mapping indiscriminately between Qubit and AncillaQubit, etc.
        check_type = Qubit if isinstance(our_wire, Qubit) else Clbit
        if not isinstance(input_dag_wire, check_type):
            raise DAGCircuitError(
                f'bit mapping invalid: {input_dag_wire} and {our_wire} are different bit types'
            )
    if _may_have_additional_wires(node):
        node_vars = {
            var
            for var in _additional_wires(node.op)
            if isinstance(var, expr.Var)
        }
    else:
        node_vars = set()
    dag_vars = set(input_dag.iter_vars())
    if dag_vars - node_vars:
        raise DAGCircuitError(
            'Cannot replace a node with a DAG with more variables.'
            f' Variables in node: {node_vars}.'
            f' Variables in DAG: {dag_vars}.'
        )
    for var in dag_vars:
        wire_map[var] = var

    reverse_wire_map = {b: a for a, b in wire_map.items()}
    # It doesn't make sense to try and propagate a condition from a control-flow op; a
    # replacement for the control-flow op should implement the operation completely.
    if (
        propagate_condition
        and not node.is_control_flow()
        and node.condition is not None
    ):
        in_dag = input_dag.copy_empty_like()
        # The remapping of `condition` below is still using the old code that assumes a 2-tuple.
        # This is because this remapping code only makes sense in the case of non-control-flow
        # operations being replaced.  These can only have the 2-tuple conditions, and the
        # ability to set a condition at an individual node level will be deprecated and removed
        # in favour of the new-style conditional blocks.  The extra logic in here to add
        # additional wires into the map as necessary would hugely complicate matters if we tried
        # to abstract it out into the `VariableMapper` used elsewhere.
        target, value = node.condition
        if isinstance(target, Clbit):
            new_target = reverse_wire_map.get(target, Clbit())
            if new_target not in wire_map:
                in_dag.add_clbits([new_target])
                wire_map[new_target], reverse_wire_map[target] = (
                    target,
                    new_target,
                )
            target_cargs = {new_target}
        else:  # ClassicalRegister
            mapped_bits = [
                reverse_wire_map.get(bit, Clbit()) for bit in target
            ]
            for ours, theirs in zip(target, mapped_bits):
                # Update to any new dummy bits we just created to the wire maps.
                wire_map[theirs], reverse_wire_map[ours] = ours, theirs
            new_target = ClassicalRegister(bits=mapped_bits)
            in_dag.add_creg(new_target)
            target_cargs = set(new_target)
        new_condition = (new_target, value)
        for in_node in input_dag.topological_op_nodes():
            if getattr(in_node.op, 'condition', None) is not None:
                raise DAGCircuitError(
                    'cannot propagate a condition to an element that already has one'
                )
            if target_cargs.intersection(in_node.cargs):
                # This is for backwards compatibility with early versions of the method, as it is
                # a tested part of the API.  In the newer model of a condition being an integral
                # part of the operation (not a separate property to be copied over), this error
                # is overzealous, because it forbids a custom instruction from implementing the
                # condition within its definition rather than at the top level.
                raise DAGCircuitError(
                    'cannot propagate a condition to an element that acts on those bits'
                )
            new_op = copy.copy(in_node.op)
            if new_condition:
                if not isinstance(new_op, ControlFlowOp):
                    new_op = new_op.c_if(*new_condition)
                else:
                    new_op.condition = new_condition
            in_dag.apply_operation_back(
                new_op, in_node.qargs, in_node.cargs, check=False
            )
    else:
        in_dag = input_dag

    if in_dag.global_phase:
        self.global_phase += in_dag.global_phase

    # Add wire from pred to succ if no ops on mapped wire on ``in_dag``
    # rustworkx's substitute_node_with_subgraph lacks the DAGCircuit
    # context to know what to do in this case (the method won't even see
    # these nodes because they're filtered) so we manually retain the
    # edges prior to calling substitute_node_with_subgraph and set the
    # edge_map_fn callback kwarg to skip these edges when they're
    # encountered.
    for in_dag_wire, self_wire in wire_map.items():
        input_node = in_dag.input_map[in_dag_wire]
        output_node = in_dag.output_map[in_dag_wire]
        if in_dag._multi_graph.has_edge(
            input_node._node_id, output_node._node_id
        ):
            pred = self._multi_graph.find_predecessors_by_edge(
                node._node_id, lambda edge, wire=self_wire: edge == wire
            )[0]
            succ = self._multi_graph.find_successors_by_edge(
                node._node_id, lambda edge, wire=self_wire: edge == wire
            )[0]
            self._multi_graph.add_edge(pred._node_id, succ._node_id, self_wire)
    for contracted_var in node_vars - dag_vars:
        pred = self._multi_graph.find_predecessors_by_edge(
            node._node_id, lambda edge, wire=contracted_var: edge == wire
        )[0]
        succ = self._multi_graph.find_successors_by_edge(
            node._node_id, lambda edge, wire=contracted_var: edge == wire
        )[0]
        self._multi_graph.add_edge(
            pred._node_id, succ._node_id, contracted_var
        )

    # Exclude any nodes from in_dag that are not a DAGOpNode or are on
    # wires outside the set specified by the wires kwarg
    def filter_fn(node):
        if not isinstance(node, DAGOpNode):
            return False
        for _, _, wire in in_dag.edges(node):
            if wire not in wire_map:
                return False
        return True

    # Map edges into and out of node to the appropriate node from in_dag
    def edge_map_fn(source, _target, self_wire):
        wire = reverse_wire_map[self_wire]
        # successor edge
        if source == node._node_id:
            wire_output_id = in_dag.output_map[wire]._node_id
            out_index = in_dag._multi_graph.predecessor_indices(
                wire_output_id
            )[0]
            # Edge directly from from input nodes to output nodes in in_dag are
            # already handled prior to calling rustworkx. Don't map these edges
            # in rustworkx.
            if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
                return None
        # predecessor edge
        else:
            wire_input_id = in_dag.input_map[wire]._node_id
            out_index = in_dag._multi_graph.successor_indices(wire_input_id)[0]
            # Edge directly from from input nodes to output nodes in in_dag are
            # already handled prior to calling rustworkx. Don't map these edges
            # in rustworkx.
            if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
                return None
        return out_index

    # Adjust edge weights from in_dag
    def edge_weight_map(wire):
        return wire_map[wire]

    node_map = self._multi_graph.substitute_node_with_subgraph(
        node._node_id,
        in_dag._multi_graph,
        edge_map_fn,
        filter_fn,
        edge_weight_map,
    )
    self._decrement_op(node.name)

    variable_mapper = _classical_resource_map.VariableMapper(
        self.cregs.values(), wire_map, add_register=self.add_creg
    )
    # Iterate over nodes of input_circuit and update wires in node objects migrated
    # from in_dag
    for old_node_index, new_node_index in node_map.items():
        # update node attributes
        old_node = in_dag._multi_graph[old_node_index]
        m_op = None
        if not old_node.is_standard_gate() and isinstance(
            old_node.op, SwitchCaseOp
        ):
            m_op = SwitchCaseOp(
                variable_mapper.map_target(old_node.op.target),
                old_node.op.cases_specifier(),
                label=old_node.op.label,
            )
        elif old_node.condition is not None:
            m_op = old_node.op
            if old_node.is_control_flow():
                m_op.condition = variable_mapper.map_condition(m_op.condition)
            else:
                new_condition = variable_mapper.map_condition(m_op.condition)
                if new_condition is not None:
                    m_op = m_op.c_if(*new_condition)
        # ----- START OF PATCHED CODE
        else:
            m_op = old_node.op
        # ----- END OF PATCHED CODE
        m_qargs = [wire_map[x] for x in old_node.qargs]
        m_cargs = [wire_map[x] for x in old_node.cargs]
        old_instruction = old_node._to_circuit_instruction()
        if m_op is None:
            new_instruction = old_instruction.replace(
                qubits=m_qargs, clbits=m_cargs
            )
        else:
            new_instruction = old_instruction.replace(
                operation=m_op, qubits=m_qargs, clbits=m_cargs
            )
        new_node = DAGOpNode.from_instruction(new_instruction)
        new_node._node_id = new_node_index
        self._multi_graph[new_node_index] = new_node
        self._increment_op(new_node.name)

    return {k: self._multi_graph[v] for k, v in node_map.items()}
