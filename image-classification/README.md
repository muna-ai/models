# Image Classification Models
This folder contains a few image classification models.

## Running a Classification Sample
The first step is to run the prediction function directly. First, we recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) as it simplifies working with Python dependencies. Once `uv` is installed, you can run 
any of the image classification model by simply executing the script directly:
```bash
# Run this in Terminal
$ uv run image-classification/resnet_50.py
```

`uv` will automatically install any required Python packages then run the script.

## Compiling the Model
Compile the Python function with Muna:
```bash
# Run this in Terminal
$ muna compile --overwrite image-classification/resnet_50.py
```

Muna will generate and compile self-contained native code (C++, Rust, etc) that runs the image classification model.

## Running the Compiled Model
Once compiled, you can run the compiled model using our client libraries. For example, run in the command line:
```bash
# Run this in Terminal
$ muna predict @USERNAME/resnet-50 --image @path/to/image.jpg
```

> [!TIP]
> Muna compiles models to run on Android, iOS, macOS, Linux, visionOS, WebAssembly, and Windows. We provide
> client libraries to run these compiled models for JavaScript, Kotlin, Android, React Native, Unity, and more.
> [Learn more](https://docs.muna.ai/predictions/create).