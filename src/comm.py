"""Turning model weights into BYTES, the thing that would travel over a network.

We measure communication as len(payload): the real number of bytes of the
serialized message. (sys.getsizeof() would only measure Python object memory,
which is not what a network carries.)

Format: a NumPy .npz archive, uncompressed. It holds the weight arrays and,
for client updates, one small integer: n_examples (how many training windows
the client used). Nothing else is ever put in a message.
"""
import io

import numpy as np

COUNT_KEY = "n_examples"


def weights_to_bytes(weights: dict, n_examples: int = None) -> bytes:
    """Serialize weights (dict of NumPy arrays) and optionally the example count."""
    if COUNT_KEY in weights:
        raise ValueError(f"'{COUNT_KEY}' is reserved and cannot be a weight name.")
    arrays = dict(weights)
    if n_examples is not None:
        arrays[COUNT_KEY] = np.array(n_examples, dtype=np.int64)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)           # savez = uncompressed
    return buffer.getvalue()


def bytes_to_weights(payload: bytes):
    """Return (weights dict, n_examples or None). Refuses pickled objects."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    n_examples = int(arrays.pop(COUNT_KEY)) if COUNT_KEY in arrays else None
    return arrays, n_examples
