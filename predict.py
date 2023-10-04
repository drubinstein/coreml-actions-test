import coremltools as ct
import onnxruntime as ort
import numpy as np

loaded_model = ct.models.MLModel("MobileNet.mlpackage")
x = loaded_model.predict({"input_1": np.random.rand(1, 224, 224, 3).astype(np.float32)})
print(x)
