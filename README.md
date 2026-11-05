# CNN Visual Learning

An interactive dark-themed learning lab for understanding how convolutional neural networks turn pixels into features. Built with Streamlit, NumPy, TensorFlow, and PyTorch. Created by Shivaji Chaulagain.

## Explore the app

The home screen links to three learning paths, with all activities available from the sidebar:

| Path | Activities |
| --- | --- |
| Learn | Introduction, guided tutorials, photo lab, animation lab |
| Experiment | Light curve lab, sample images, image uploads, custom filters, advanced visualizations |
| Build & Train | NumPy CNN, TensorFlow/PyTorch CNNs, training visualization, layer-by-layer training, architecture builder, classification playground, implementation comparison |

### Photo Lab

Use real garden and flower photographs bundled with scikit-learn, or upload your own photo. Watch the original photograph, active window, and progressively built output together. Compare max/average pooling, edge detection, blur, sharpening, and activation functions. Photos are reduced to at most 128 pixels per side and processed in grayscale for responsive playback.

The sample-image picker starts with these photographs. Each filter's animation offers a **Whole image** view and a **Numbers close-up**. Full-image playback computes every output value and displays up to 48 checkpoints. On large uploaded feature maps, animation inputs are sampled to at most 128 pixels per side; the regular image-processing results retain their existing resolution.

### Light Curve Lab

Explore astronomical brightness time series with animated **1D convolution, maximum/average/minimum pooling, and activation functions**. Gold marks the moving sample window; mint diamonds show the maximum or minimum selected by pooling; the processed curve appears below. Play, pause, restart, or scrub through up to 80 checkpoints, and download the animation or processed values.

The default is an **observed Kepler-8 quarter-0 light curve** from NASA/MAST, bundled as 453 quality-filtered measurements. Exact source, units, quality selection, and preparation are documented in [data/README.md](data/README.md). Simulated transits, variable-star brightness, and flares are explicitly labeled. Upload a CSV with numeric time and brightness columns to explore your own data.

Time units are preserved. Uploaded rows with nonfinite time/flux are reported and removed; samples are sorted by time, and duplicate timestamps are rejected. Windows operate on sample order with no interpolation; by default, windows crossing gaps greater than three times the median cadence are skipped. Output timestamps are the mean of the input window's timestamps. Median normalization and centering are available, but neither constitutes detrending. Pooling can change transit depths and is presented as an educational operation, not an inference pipeline. Uploaded quality flags and measurement uncertainties are not applied or propagated automatically.

### Animation Lab

Watch **max pooling, average pooling, convolution, activations, and flattening** step by step. Play, pause, restart, or scrub the timeline. Gold outlines show the input region and the corresponding output value; every frame displays the calculation. Change image patterns, window size, stride, playback speed, and activation (ReLU, sigmoid, or tanh).

The sample-image and image-upload results also include **Animate feature operations with numbers** beneath each filter. Enable the animation, select an operation, and move the crop to inspect actual feature-map values. Crops are not resized; the animated output describes the selected crop.

Download an activity as a standalone interactive HTML file that works offline. Window operations use valid padding and skip incomplete border windows. Convolution in this lab follows the cross-correlation convention used by CNN libraries: multiply by the kernel without flipping it.

### Models and datasets

- The main training pages use **scikit-learn digits**, not MNIST: 1,797 grayscale 8×8 images with 10 classes. Pixels are normalized from 0–16 to 0–1. A stratified 80/20 split is reproducible and requires no download.
- Model weights are private to each Streamlit session and persist across reruns. Changing a model's configuration creates fresh weights. **Reset models** clears the main training models.
- The NumPy implementation performs a weight update per sample, grouped into batches for progress and loss reporting. It is an educational implementation, not a fast minibatch optimizer.
- The classification playground's **MNIST CNN architecture is untrained**. Its predictions illustrate random weights, not digit recognition. ResNet-18 and MobileNetV2 use pretrained ImageNet weights and download weights on first use; MobileNetV2 also downloads class labels when needed.
- The separate layer-by-layer activity uses a **synthetic blob dataset**, not handwritten digits.
- Initial comparison predictions use random weights; benchmark timings are illustrative and include framework overhead. They are not a controlled performance study.

## Run locally

Use **Python 3.12** for a conservative deployment target supported by the deep learning packages. Package availability varies by operating system and Python version.

```bash
git clone https://github.com/Shivaji-137/CNN-Visual-Learning-.git
cd CNN-Visual-Learning-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run full_apps.py
```

`streamlit run main.py` opens the same application.

For a lighter install without deep learning frameworks:

```bash
python -m pip install -r requirements-core.txt
streamlit run full_apps.py
```

The lighter install supports tutorials, image experiments, animations, the architecture builder/code export, and the NumPy CNN. Framework-specific activities explain which additional library is missing. Install `requirements.txt` to enable all activities. The home screen and basic activities do not import TensorFlow or PyTorch.

## Deploy on Streamlit Community Cloud

1. Choose this repository and the intended branch (currently `main`).
2. Set the entry point to **`full_apps.py`**.
3. Select **Python 3.12** in the deployment's advanced settings. Changing an existing app's Python version may require redeploying it.
4. Deploy using the root `requirements.txt`, which includes `requirements-core.txt` and the framework dependencies.

The repository's `.streamlit/config.toml` sets the dark theme and a 10 MB upload limit. OpenCV uses the headless package. Heavy training and first-time pretrained-model downloads may take longer on shared hosting; start with a small epoch/filter count.

After pushing changes to the connected branch, check the Streamlit build logs. A dependency syntax check alone does not confirm a successful deployment.

## Development and checks

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

Tests cover numeric operation outputs and animation frames, NumPy gradients and partial batches, PyTorch logits/layouts/callbacks, TensorFlow class counts/training/feature maps, generated model code, classifier preprocessing, and Streamlit page navigation. Framework tests skip when the corresponding framework is absent. CI runs the full suite on Python 3.12. Pretrained ImageNet downloads are not required by the automated suite.

## Project structure

| File | Purpose |
| --- | --- |
| `full_apps.py`, `main.py` | App shell, navigation, home screen, and entry points |
| `light_curve_lab.py`, `data/` | Astronomical time-series animations and bundled observed Kepler data |
| `photo_explorer.py` | Built-in photographs and full-image operation playback |
| `operations_explorer.py` | Numeric operation steps and interactive animations |
| `core_pages.py` | Main training and model comparison pages |
| `framework_loader.py` | Deferred framework imports and missing-library guidance |
| `cnn_scratch.py`, `cnn_tensorflow.py`, `cnn_pytorch.py` | CNN implementations |
| `complete_cnn_copy.py`, `filters.py` | Image processing experiments and custom filters |
| `guided_tutorials.py` | Step-by-step lessons |
| `architecture_builder.py` | Layer builder, shape checks, code export |
| `classification_playground.py` | Predictions, activation heatmaps, feature maps |
| `training_visualization.py` | Layer-by-layer training on synthetic data |
| `advanced_visualizations.py`, `visualizations.py`, `utils.py` | Learning visualizations and helpers |
| `tests/` | Regression and app navigation checks |

## License

Application code: [MIT](LICENSE). Please credit Shivaji Chaulagain when using this project.

The built-in photographs bundled by scikit-learn are separately licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/): [garden/China photo by danielbuechele](https://www.flickr.com/photos/danielbuechele/6061409035/) and [flower photo from vultilion](https://www.flickr.com/photos/vultilion/6056698931/). Photo examples are resized, converted to grayscale, and filtered for visualization. Attribution is also displayed in the app and exported photo animations.
