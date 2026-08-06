# Image Generation Models
This folder contains a few image generation models.

## Running a Generation Sample
The first step is to run the prediction function directly. First, we recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) as it simplifies working with Python dependencies. Once `uv` is installed, you can run
any of the image generation models by simply executing the script directly:
```bash
# Run this in Terminal
$ uv run image-generation/flux_2_klein_9b.py
```

`uv` will automatically install any required Python packages then run the script.

> [!NOTE]
> [FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) is a gated repository on Hugging Face.
> Request access to the checkpoint, then set the `HF_TOKEN` environment variable to your Hugging Face access token
> before running or compiling the model.

## Compiling the Model
Compile the Python function with Muna:
```bash
# Run this in Terminal
$ muna compile --overwrite image-generation/flux_2_klein_9b.py
```

Muna will generate and compile self-contained native code (C++, Rust, etc) that runs the image generation model.

> [!IMPORTANT]
> The image generation models currently compile only for Linux x64 with CUDA, targeting datacenter GPUs.

## Running the Compiled Model
Once compiled, you can run the compiled model using our client libraries. For example, run in the command line:
```bash
# Run this in Terminal
$ muna predict @USERNAME/flux-2-klein-9b --prompt "A watercolor painting of a lighthouse at dawn"
```

> [!TIP]
> Muna compiles models to run on Android, iOS, macOS, Linux, visionOS, WebAssembly, and Windows. We provide
> client libraries to run these compiled models for JavaScript, Kotlin, Android, React Native, Unity, and more.
> [Learn more](https://docs.muna.ai/predictions/create).
