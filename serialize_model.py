import coremltools as ct

# Load TensorFlow model
import tensorflow as tf  # Tf 2.2.0

tf_model = tf.keras.applications.MobileNet()
print(tf_model.inputs)
# Convert using the same API
model_from_tf = ct.convert(
    tf_model, compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS12
)
model_from_tf.save("MobileNet.mlpackage")
