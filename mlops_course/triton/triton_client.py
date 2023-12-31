import numpy
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


def get_7_number():
    return np.array(
        [
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.6450,
                1.9305,
                1.5996,
                1.4978,
                0.3395,
                0.0340,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                2.4015,
                2.8088,
                2.8088,
                2.8088,
                2.8088,
                2.6433,
                2.0960,
                2.0960,
                2.0960,
                2.0960,
                2.0960,
                2.0960,
                2.0960,
                2.0960,
                1.7396,
                0.2377,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.4286,
                1.0268,
                0.4922,
                1.0268,
                1.6505,
                2.4651,
                2.8088,
                2.4396,
                2.8088,
                2.8088,
                2.8088,
                2.7578,
                2.4906,
                2.8088,
                2.8088,
                1.3577,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.2078,
                0.4159,
                -0.2460,
                0.4286,
                0.4286,
                0.4286,
                0.3268,
                -0.1569,
                2.5797,
                2.8088,
                0.9250,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.6322,
                2.7960,
                2.2360,
                -0.1951,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.1442,
                2.5415,
                2.8215,
                0.6322,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                1.2177,
                2.8088,
                2.6051,
                0.1358,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.3268,
                2.7451,
                2.8088,
                0.3649,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                1.2686,
                2.8088,
                1.9560,
                -0.3606,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.3097,
                2.1851,
                2.7324,
                0.3140,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                1.1795,
                2.8088,
                1.8923,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.5304,
                2.7706,
                2.6306,
                0.3013,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.1824,
                2.3887,
                2.8088,
                1.6887,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.3860,
                2.1596,
                2.8088,
                2.3633,
                0.0213,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.0595,
                2.8088,
                2.8088,
                0.5559,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.0296,
                2.4269,
                2.8088,
                1.0395,
                -0.4115,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                1.2686,
                2.8088,
                2.8088,
                0.2377,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                0.3522,
                2.6560,
                2.8088,
                2.8088,
                0.2377,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                1.1159,
                2.8088,
                2.8088,
                2.3633,
                0.0849,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                1.1159,
                2.8088,
                2.2105,
                -0.1951,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
        ],
        dtype=np.float32,
    )


def get_client():
    return InferenceServerClient(url="0.0.0.0:9000")


def call_triton_solver(data):
    client = get_client()
    input_data = InferInput(
        name="input", shape=data.shape, datatype=np_to_triton_dtype(data.dtype)
    )
    input_data.set_data_from_numpy(data, binary_data=True)

    infer_output = InferRequestedOutput("output", binary_data=True)

    response = client.infer("onnx-mnist", inputs=[input_data], outputs=[infer_output])

    return response.as_numpy("output")


def main():
    response = call_triton_solver(np.ones(shape=(1, 1, 28, 28), dtype=numpy.float32))
    assert np.argmax(response) == 7


if __name__ == '__main__':
    main()
