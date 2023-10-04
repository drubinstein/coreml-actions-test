import coremltools as ct
import onnxruntime as ort
import numpy as np

FFT_HOP = 256
N_FFT = 8 * FFT_HOP

NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3
# base frequency of the CENTRAL bin of the first semitone (i.e., the
# second bin if annotations_bins_per_semitone is 3)
ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_CHANNELS = 1
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

# ANNOT_N_TIME_FRAMES is the number of frames in the time-frequency representations we compute
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

# AUDIO_N_SAMPLES is the number of samples in the (clipped) audio that we use as input to the models
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP

loaded_model = ct.models.MLModel("MobileNet.mlpackage")
x = loaded_model.predict({"input_1": np.random.rand(1, 224, 224, 3).astype(np.float32)})
print(x)

loaded_model_2 = ct.models.MLModel("nmp.mlpackage")
x = loaded_model_2.predict(
    {"input_2": np.random.rand(1, AUDIO_N_SAMPLES, 1).astype(np.float32) * 2 - 1}
)
print(x)
