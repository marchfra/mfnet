import numpy as np

from mfnet.layer import Id, Layer, Linear
from mfnet.tensor import Tensor, tensor


def insert_bias_weights(w: Tensor) -> Tensor:
    if not w.size > 0:
        raise ValueError("Weights must be initialized with a non-empty shape.")

    if len(w.shape) == 1:
        w = np.expand_dims(w, axis=0)
    bias_weights = tensor([1] + [0] * (w.shape[1] - 1))
    return np.insert(w, 0, bias_weights, axis=0)


def manual_example(x: Tensor, y: Tensor, weights: list[Tensor]) -> None:
    from mfnet.loss import MSELoss  # noqa: PLC0415

    def g(x: Tensor) -> Tensor:
        x[0] = np.ones(x.shape[1])
        x[1:] = x[1:]
        return x

    def g_prime(x: Tensor) -> Tensor:
        d = np.ones_like(x)
        d[0] = np.zeros(x.shape[1])
        return d

    # --- Layer 1 ---
    w1 = weights[0]
    z1 = w1 @ x
    print(f"Layer 1 z:\n{z1}")
    a1 = g(z1)
    print(f"Layer 1 a:\n{a1}")

    # --- Layer 2 ---
    w2 = weights[1]
    z2 = w2 @ a1
    print(f"Layer 2 z:\n{z2}")
    a2 = g(z2)
    print(f"Layer 2 a:\n{a2}")

    # --- Layer 3 ---
    w3 = weights[2]
    z3 = w3 @ a2
    print(f"Layer 3 z:\n{z3}")
    a3 = g(z3)
    print(f"Layer 3 a:\n{a3}")

    print()

    y_hat = a3
    dJ_dy_hat = MSELoss().grad(y_hat, y)  # noqa: N806

    delta3 = dJ_dy_hat * g_prime(z3)
    dJ_dw3 = delta3 @ a2.T  # noqa: N806
    print(f"Layer 3 dJ/dw:\n{dJ_dw3}")

    delta2 = (w3.T @ delta3) * g_prime(z2)
    dJ_dw2 = delta2 @ a1.T  # noqa: N806
    print(f"Layer 2 dJ/dw:\n{dJ_dw2}")

    delta1 = (w2.T @ delta2) * g_prime(z1)
    dJ_dw1 = delta1 @ x.T  # noqa: N806
    print(f"Layer 1 dJ/dw:\n{dJ_dw1}")


def coded_example(x: Tensor, y: Tensor, weights: list[Tensor]) -> None:
    from mfnet.loss import MSELoss  # noqa: PLC0415

    layers: list[Layer] = []
    for w in weights:
        layers.append(Linear(w.shape[1], w.shape[0] - 1))
        layers[-1].weights = w
        layers.append(Id())

    for layer in layers:
        x = layer.forward(x)

    print(f"Output after forward pass:\n{x}", end="\n\n")

    y_hat = x
    dJ_d_y_hat = MSELoss().grad(y_hat, y)  # noqa: N806
    grad = dJ_d_y_hat.copy()
    print(f"Initial gradient:\n{grad}", end="\n\n")
    layer_types = ["Linear", "Activation"]

    for layer in reversed(layers):
        grad = layer.backward(grad)
        layer_type = layer_types[layers.index(layer) % 2]
        print(f"Layer {(layers.index(layer) + 2) // 2} {layer_type}")
        if layer_type == "Linear":
            print(f"dJ/dw:\n{layer.dJ_dw}")
            print(f"Weights\n{layer.weights}")
        print(
            f"Gradient:\n{grad}",
            end="\n" * (2 - layers.index(layer) % 2),
        )
        # if layers.index(layer) + 1 == 5:
        #     break


def main() -> None:
    x = tensor([[1, 1, 2, 2], [2, 3, 2, 3]])
    x = np.insert(x, 0, 1, axis=0)  # Add bias "feature"

    y = tensor([[1, 1, 1, 1], [1, 1, 2, 2]])

    w1 = tensor([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
    w1 = insert_bias_weights(w1)

    w2 = tensor([[1, 1, 0, 1], [-1, 1, 2, -2]])
    w2 = insert_bias_weights(w2)

    w3 = tensor([0, 1, -1])
    w3 = insert_bias_weights(w3)

    print("Manual example:")
    manual_example(x, y, [w1, w2, w3])
    print("\n\nCoded example:")
    coded_example(x, y, [w1, w2, w3])


if __name__ == "__main__":
    main()
