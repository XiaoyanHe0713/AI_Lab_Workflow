"""Counted arithmetic: a scalar type that records every operation as a node of an expression DAG.

Standalone on purpose (no lab imports, no numpy): the judge's sandbox loads this file by path and
hands the candidate `Counted` inputs, so the operation count of a straight-line program is read
off the outputs rather than trusted from the candidate. Only operations that reach an output are
counted, each distinct operation once (a shared subexpression is one operation), which is the
cost of the straight-line program the candidate actually computed. A plain Python number is a
constant: multiplying by a precomputed twiddle factor is one multiplication; computing the twiddle
itself from cos and sin is free, as in every published operation count. Negation is free.

An output that is not a Counted expression of the inputs (for example a float obtained by leaving
counted arithmetic) is reported as untraceable, and the certifier fails the attempt.
"""
from __future__ import annotations

OPS = ("add", "mul", "div")


class Counted:
    __slots__ = ("value", "op", "args")

    def __init__(self, value, op="input", args=()):
        self.value = float(value)
        self.op = op          # input | const | add | mul | div | neg
        self.args = args

    @staticmethod
    def _wrap(x):
        if isinstance(x, Counted):
            return x
        if isinstance(x, bool):
            return NotImplemented
        if isinstance(x, (int, float)):
            return Counted(x, "const")
        return NotImplemented

    def _bin(self, other, op, value):
        return Counted(value, op, (self, other))

    def __add__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else self._bin(w, "add", self.value + w.value)

    def __radd__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else w._bin(self, "add", w.value + self.value)

    def __sub__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else self._bin(w, "add", self.value - w.value)

    def __rsub__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else w._bin(self, "add", w.value - self.value)

    def __mul__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else self._bin(w, "mul", self.value * w.value)

    def __rmul__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else w._bin(self, "mul", w.value * self.value)

    def __truediv__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else self._bin(w, "div", self.value / w.value)

    def __rtruediv__(self, o):
        w = Counted._wrap(o)
        return NotImplemented if w is NotImplemented else w._bin(self, "div", w.value / self.value)

    def __neg__(self):
        return Counted(-self.value, "neg", (self,))   # a sign flip is free

    def __pos__(self):
        return self

    def __pow__(self, k):
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise TypeError("only small nonnegative integer powers are counted arithmetic")
        if k == 0:
            return Counted(1.0, "const")
        out = self
        for _ in range(k - 1):
            out = out * self
        return out

    def __float__(self):
        raise TypeError("a Counted value cannot leave counted arithmetic (float() is not allowed)")

    def __bool__(self):
        raise TypeError("a Counted value has no truth value: straight-line programs do not branch on data")

    def __repr__(self):
        return f"Counted({self.value!r}, {self.op})"


def count_ops(outputs) -> dict:
    """Count the distinct operations reachable from the outputs. Returns counts by kind, the total,
    and `untraceable_outputs`, the number of outputs that are not Counted expressions."""
    counts = {k: 0 for k in OPS}
    untraceable = 0
    seen = set()
    stack = []
    for o in _flatten(outputs):
        if isinstance(o, Counted):
            stack.append(o)
        else:
            untraceable += 1
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node.op in counts:
            counts[node.op] += 1
        stack.extend(node.args)
    counts["total"] = sum(counts[k] for k in OPS)
    counts["untraceable_outputs"] = untraceable
    return counts


def values(outputs):
    """The float values of a (nested) output structure."""
    if isinstance(outputs, Counted):
        return outputs.value
    if isinstance(outputs, (list, tuple)):
        return [values(o) for o in outputs]
    if isinstance(outputs, dict):
        return {str(k): values(v) for k, v in outputs.items()}
    return outputs


def wrap_inputs(x):
    """Wrap a (nested) list of numbers as Counted inputs."""
    if isinstance(x, (list, tuple)):
        return [wrap_inputs(v) for v in x]
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return Counted(x, "input")
    return x


def _flatten(o):
    if isinstance(o, (list, tuple)):
        for v in o:
            yield from _flatten(v)
    elif isinstance(o, dict):
        for v in o.values():
            yield from _flatten(v)
    else:
        yield o
