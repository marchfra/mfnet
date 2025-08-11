from mfnet.tensor import Tensor, tensor


def main() -> None:
    x: Tensor = tensor([1, 2, 3])
    print(x)
    print(x.dtype)


if __name__ == "__main__":
    main()
