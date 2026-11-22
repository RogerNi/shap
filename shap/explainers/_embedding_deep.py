"""Explicit embedding-output Deep SHAP for TensorFlow and PyTorch."""

from copy import deepcopy

import numpy as np

from .._explanation import Explanation
from ._deep import DeepExplainer
from ._explainer import Explainer


class EmbeddingDeepExplainer(Explainer):
    """Explain integer tokens through their embedding-output contributions.

    Parameters
    ----------
    model : keras.Model or torch.nn.Module
        Single-input model with one output tensor of shape [batch, outputs].
    data : numpy.ndarray or torch.Tensor
        Nonempty integer token background. Callable and multi-input backgrounds
        are not supported. Missing tokens are replaced by reference tokens in place.
    embedding_layer : keras.layers.Embedding or torch.nn.Embedding
        Explicit plain embedding layer belonging to model. Masked, reused and
        specialized TensorFlow embeddings are not supported.
    tail_model : torch.nn.Module, optional
        Downstream model taking [batch, positions, embedding_dim]. Required for
        non-Sequential PyTorch models; caller must include every downstream path.
        For Sequential models starting with the embedding it is inferred.
        TensorFlow builds the tail from the existing graph; do not supply a tail.

    Notes
    -----
    Models are copied with identical weights at construction and evaluated in
    inference mode. Recreate the explainer after training. TensorFlow models must
    be cloneable. Original weights, graph metadata and PyTorch training mode are
    not modified. Forward equivalence is checked on background and every input;
    this is a guard, not proof of equivalence on untested inputs for supplied tails.

    Returns token-position sums of embedding-coordinate Deep SHAP contributions,
    not exact discrete-token Shapley values. Existing downstream operator limits,
    nonlinear approximations and max-pool tie behavior remain. Frameworks are
    imported lazily; constructing this class requires only the selected backend.
    """

    def __init__(self, model, data, embedding_layer, *, tail_model=None):
        if callable(data) or isinstance(data, (list, tuple)):
            raise ValueError("Provide one fixed integer background array or tensor.")
        # TensorFlow/Keras models do not expose nn.Module.named_parameters.
        self.framework = "pytorch" if hasattr(model, "named_parameters") else "tensorflow"
        if self.framework == "tensorflow":
            self._init_tensorflow(model, embedding_layer, tail_model)
        else:
            self._init_pytorch(model, embedding_layer, tail_model)
        self.model = self._model
        self.data = data
        embedded = self._embed(data)
        self._check_forward(data, embedded)
        self._delegate = DeepExplainer(self._tail, embedded)
        self.expected_value = self._delegate.expected_value
        self.embedding_boundary = self._boundary_name
        self._token_shape = tuple(data.shape)[1:]

    def _init_tensorflow(self, model, embedding, tail):
        import tensorflow as tf

        if not tf.executing_eagerly():
            raise ValueError("EmbeddingDeepExplainer currently supports TensorFlow eager mode only.")
        if tail is not None or len(model.inputs) != 1 or len(model.outputs) != 1:
            raise ValueError("TensorFlow requires one input/output tensor and no supplied tail_model.")
        if not any(layer is embedding for layer in model.layers):
            raise ValueError("embedding_layer must belong to the model.")
        if (
            type(embedding) is not tf.keras.layers.Embedding
            or embedding.mask_zero
            or getattr(embedding, "lora_enabled", False)
            or getattr(embedding, "quantization_mode", None) is not None
            or len(embedding._inbound_nodes) != 1
            or embedding.input is not model.inputs[0]
        ):
            raise ValueError("Use a plain, unmasked, unreused Embedding directly on the input.")
        pending, visited = list(model.outputs), set()
        while pending:
            tensor = pending.pop()
            if tensor is embedding.output or id(tensor) in visited:
                continue
            visited.add(id(tensor))
            layer, node_index, _ = tensor._keras_history
            if isinstance(layer, tf.keras.layers.InputLayer):
                raise ValueError("Model input bypasses the embedding boundary.")
            pending.extend(tf.nest.flatten(layer._inbound_nodes[node_index].input_tensors))
        self._model = tf.keras.models.clone_model(model)
        self._model.set_weights(model.get_weights())
        copied = self._model.get_layer(embedding.name)
        self._lookup = copied
        # Build on a second clone: Keras3 may replay graph nodes for intermediate inputs.
        tail_copy = tf.keras.models.clone_model(model)
        tail_copy.set_weights(model.get_weights())
        self._tail = tf.keras.Model(tail_copy.get_layer(embedding.name).output, tail_copy.outputs[0])
        self._vocabulary = embedding.input_dim
        self._boundary_name = embedding.name
        self._input_shape = tuple(model.inputs[0].shape)[1:]

    def _init_pytorch(self, model, embedding, tail):
        import torch

        if type(embedding) is not torch.nn.Embedding:
            raise ValueError("Use a plain torch.nn.Embedding.")
        if embedding.max_norm is not None:
            raise ValueError("Embedding max_norm modifies weights during inference and is unsupported.")
        names = [name for name, layer in model.named_modules() if layer is embedding]
        if len(names) != 1:
            raise ValueError("embedding_layer must belong to the model.")
        self._model = deepcopy(model).eval()
        self._lookup = self._model.get_submodule(names[0])
        if tail is None:
            children = list(self._model.children())
            if not isinstance(model, torch.nn.Sequential) or not children or children[0] is not self._lookup:
                raise ValueError("Supply tail_model for a non-Sequential PyTorch model.")
            self._tail = torch.nn.Sequential(*children[1:])
        else:
            self._tail = deepcopy(tail).eval()
        self._vocabulary = embedding.num_embeddings
        self._boundary_name = names[0]
        self._input_shape = None

    def _embed(self, values):
        if self.framework == "tensorflow":
            values = np.asarray(values)
            if values.dtype.kind not in "iu":
                raise ValueError("Token IDs must have an integer dtype.")
            if values.ndim < 2 or not values.size or len(values) == 0:
                raise ValueError("Token arrays must be nonempty [batch, positions, ...].")
            if len(values.shape[1:]) != len(self._input_shape) or any(
                expected is not None and actual != expected
                for actual, expected in zip(values.shape[1:], self._input_shape)
            ):
                raise ValueError("Token shape does not match the model.")
            if values.min() < 0 or values.max() >= self._vocabulary:
                raise ValueError("Token IDs are outside the vocabulary.")
            return self._lookup(values).numpy()
        import torch

        if not isinstance(values, torch.Tensor) or values.dtype not in (torch.int32, torch.int64):
            raise ValueError("PyTorch token IDs must be an int32 or int64 tensor.")
        if values.ndim < 2 or not values.numel() or not len(values):
            raise ValueError("Token tensors must be nonempty [batch, positions, ...].")
        if values.min().item() < 0 or values.max().item() >= self._vocabulary:
            raise ValueError("Token IDs are outside the vocabulary.")
        with torch.no_grad():
            return self._lookup(values).detach()

    def _check_forward(self, values, embedded):
        if self.framework == "tensorflow":
            original = np.asarray(self._model(values, training=False))
            predicted = np.asarray(self._tail(embedded, training=False))
        else:
            import torch

            with torch.no_grad():
                original = self._model(values).detach().cpu().numpy()
                predicted = self._tail(embedded).detach().cpu().numpy()
        if original.ndim != 2 or original.shape[0] != len(values):
            raise ValueError("Model output must be one tensor [batch, outputs].")
        np.testing.assert_allclose(
            original, predicted, rtol=1e-5, atol=1e-6, err_msg="Embedding tail does not reproduce original predictions."
        )

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=True):
        """Return [batch, token_positions, ..., outputs] contribution arrays."""
        embedded = self._embed(X)
        if tuple(X.shape)[1:] != self._token_shape:
            raise ValueError("Token shape must match the background.")
        self._check_forward(X, embedded)
        values = self._delegate.shap_values(
            embedded, ranked_outputs, output_rank_order, check_additivity=check_additivity
        )
        if ranked_outputs is not None:
            contributions, ranks = values
            return contributions.sum(axis=-2), ranks
        return values.sum(axis=-2)

    def __call__(self, X):
        values = self.shap_values(X)
        data = X.detach().cpu().numpy() if self.framework == "pytorch" else np.asarray(X)
        baseline = self.expected_value
        if hasattr(baseline, "numpy"):
            baseline = baseline.numpy()
        return Explanation(
            values=values,
            data=data,
            base_values=np.broadcast_to(np.asarray(baseline), (len(data), len(np.asarray(baseline).reshape(-1)))),
        )
