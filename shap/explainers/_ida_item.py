"""Item-level coalition SHAP using public SHAP APIs; no IDA installation needed.

Each explainer describes ONE fixed binary and its item layout. File offsets, not
virtual addresses, index model inputs. Scores must be deterministic continuous
outputs. This is perturbation SHAP, independent of EmbeddingDeepExplainer.
"""

from dataclasses import dataclass

import numpy as np

from .._explanation import Explanation
from ._exact import ExactExplainer
from ._permutation import PermutationExplainer


@dataclass(frozen=True)
class IDAItem:
    """A contiguous file-backed instruction/data item, with exclusive stop."""

    name: str
    start: int
    stop: int
    address: int | None = None  # Display metadata only; never used as an offset.


class IDAItemMasker:
    """Replace whole items with aligned reference bytes without shifting bytes.

    Bytes outside the items are FIXED to x, even in the empty coalition. Multiple
    references define a uniform empirical average, not conditional imputation.
    Padding ID 256 is supported: integer arrays are never cast to uint8.
    """

    def __init__(self, x, references, items):
        x = np.asarray(x)
        references = np.asarray(references)
        if x.ndim != 1 or not x.size or x.dtype.kind not in "iu":
            raise ValueError("x must be a nonempty 1D integer token array")
        if references.ndim == 1:
            references = references[None, :]
        if (
            references.ndim != 2
            or not len(references)
            or references.shape[1] != len(x)
            or references.dtype.kind not in "iu"
        ):
            raise ValueError("references must be nonempty integer [references, length]")
        if np.any(x < 0) or np.any(x > 256) or np.any(references < 0) or np.any(references > 256):
            raise ValueError("Expected byte IDs 0..255 or padding ID 256")
        self.x = x.astype(np.int32, copy=True)
        self.references = references.astype(np.int32, copy=True)
        self.items = tuple(items)
        if not self.items or any(not isinstance(item, IDAItem) for item in self.items):
            raise ValueError("Provide at least one IDAItem")
        if len({item.name for item in self.items}) != len(self.items):
            raise ValueError("Item names must be unique")
        self.owner = np.full(len(x), -1, dtype=np.int32)
        for i, item in enumerate(self.items):
            if (
                not isinstance(item.name, str)
                or not item.name
                or not isinstance(item.start, (int, np.integer))
                or not isinstance(item.stop, (int, np.integer))
                or not 0 <= item.start < item.stop <= len(x)
            ):
                raise ValueError("Items need a name and valid nonempty file-offset ranges")
            if np.any(self.owner[item.start : item.stop] != -1):
                raise ValueError("Overlapping items are ambiguous and are not supported")
            self.owner[item.start : item.stop] = i
        self.covered = self.owner >= 0
        # Prevent accidental edits to the coalition definition after construction.
        for array in (self.x, self.references, self.owner, self.covered):
            array.setflags(write=False)

    @property
    def n_items(self):
        return len(self.items)

    def __call__(self, mask):
        """One boolean/binary switch per item -> [n_references, length]."""
        return next(self.masked_batches(mask, len(self.references)))

    def masked_batches(self, mask, batch_size):
        """Yield reference substitutions without constructing all references at once."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        mask = np.asarray(mask)
        if mask.shape != (self.n_items,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must contain one binary switch per item")
        hidden = np.zeros(len(self.x), dtype=bool)
        hidden[self.covered] = ~mask.astype(bool)[self.owner[self.covered]]
        for start in range(0, len(self.references), batch_size):
            refs = self.references[start : start + batch_size]
            result = np.broadcast_to(self.x, refs.shape).copy()
            result[:, hidden] = refs[:, hidden]
            yield result


class IDAItemExplainer:
    """Explain the item-presence game g(S) = mean_r f(mask(x, r, S)).

    ``model`` accepts integer [batch, length] NumPy arrays and returns NumPy
    [batch] or [batch, outputs] scores. Wrap TF/Torch inference explicitly.
    Model batches are bounded even when a binary has millions of bytes.
    """

    def __init__(self, model, x, references, items, *, batch_size=8, output_names=None):
        if not callable(model):
            raise TypeError("model must be a callable returning NumPy scores")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.model = model
        self.masker = IDAItemMasker(x, references, items)
        self.batch_size = batch_size
        self.output_names = output_names
        self.model_rows_evaluated = 0
        self._n_outputs = None

    def coalition_scores(self, masks):
        """Evaluate item-presence rows, averaging aligned references per row."""
        masks = np.asarray(masks)
        if masks.ndim != 2 or masks.shape[1] != self.masker.n_items or not len(masks):
            raise ValueError("Expected nonempty [coalitions, items] matrix")
        scores = []
        for mask in masks:
            total = None
            for inputs in self.masker.masked_batches(mask, self.batch_size):
                outputs = np.asarray(self.model(inputs))
                self.model_rows_evaluated += len(inputs)
                if outputs.ndim == 1:
                    outputs = outputs[:, None]
                if (
                    outputs.ndim != 2
                    or outputs.shape[0] != len(inputs)
                    or not outputs.shape[1]
                    or outputs.dtype.kind not in "fiu"
                    or not np.isfinite(outputs).all()
                ):
                    raise ValueError("Model must return finite [batch] or [batch, outputs] NumPy scores")
                if self._n_outputs is None:
                    self._n_outputs = outputs.shape[1]
                if outputs.shape[1] != self._n_outputs:
                    raise ValueError("Model output count changed")
                subtotal = outputs.astype(np.float64).sum(axis=0)
                total = subtotal if total is None else total + subtotal
            scores.append(total / len(self.masker.references))
        return np.asarray(scores)

    def explain(self, *, algorithm="permutation", max_evals=None, seed=0):
        """Return one Explanation with values [items, outputs].

        Exact enumeration is capped at 16 items. Permutation is approximate;
        max_evals counts coalition evaluations, each averaging all references.
        Explanation.data contains presence switches (ones), not byte values.
        """
        n = self.masker.n_items
        background = np.zeros((1, n), dtype=np.int8)
        names = [item.name for item in self.masker.items]
        if algorithm == "exact":
            if n > 16:
                raise ValueError("Exact enumeration is capped at 16 items; use permutation")
            if max_evals is None:
                max_evals = 2**n
            explainer = ExactExplainer(self.coalition_scores, background, feature_names=names)
        elif algorithm == "permutation":
            if max_evals is None:
                max_evals = 10 * (2 * n + 1)
            explainer = PermutationExplainer(self.coalition_scores, background, feature_names=names, seed=seed)
        else:
            raise ValueError("algorithm must be 'exact' or 'permutation'")
        result = explainer(np.ones((1, n), dtype=np.int8), max_evals=max_evals, silent=True)[0]
        if self.output_names is not None and len(self.output_names) != self._n_outputs:
            raise ValueError("output_names must match the number of model outputs")
        # SHAP can squeeze a singleton output. Keep a stable [items, outputs]
        # shape regardless of scalar vs one-column model scores or SHAP version.
        return Explanation(
            values=np.asarray(result.values).reshape(n, self._n_outputs),
            base_values=np.asarray(result.base_values).reshape(self._n_outputs),
            data=np.ones(n, dtype=np.int8),
            feature_names=names,
            output_names=self.output_names,
        )
