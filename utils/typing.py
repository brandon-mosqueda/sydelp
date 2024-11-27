import numpy as np

from numpy.typing import NDArray
from typing import Union, Tuple
from keras.src.models import Model as KerasModel


WeightsShapes = list[Tuple[int, ...]]

NNumeric = Union[
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64, np.float128
]

Int = Union[np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
            np.int8, np.int16, np.int32, np.int64, np.int_, int]

Float = Union[float, np.float_, np.float16,
              np.float32, np.float64, np.float128]

NumArray = NDArray[NNumeric]
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float_]
BoolArray = NDArray[np.bool_]
