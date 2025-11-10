# CNN Pretrained Benchmark Architectures

## Overview
A small benchmark suite that implements, trains, and evaluates several classic CNN architectures (LeNet, AlexNet, VGG, ResNet) on CIFAR-10. The project provides model definitions, a single training harness that sequentially trains and checkpoints each architecture, and saves best-performing models to disk for later use.

## Features
- Implementations of LeNet, AlexNet, VGG, and a simple ResNet.
- Single training harness that:
  - loads CIFAR-10, normalizes inputs,
  - compiles models (Adam + sparse_categorical_crossentropy),
  - trains with ModelCheckpoint (save_best_only on val_accuracy),
  - evaluates on the test set and saves best models to `save_models/`.
- Checkpointed best models (.h5) saved to `save_models/` for inference.
- Minimal, easy-to-run codebase for learning and comparing architectures.

## Repository layout (key files)
- `app.py` - training harness and runner that loads CIFAR-10, trains each architecture and saves checkpoints to `save_models/`.
- `LeNet.py` - LeNet model builder.
- `AlexNet.py` - AlexNet-style model builder.
- `VGG.py` - VGG-style model builder.
- `ResNet.py` - ResNet builder with a residual block helper.
- `save_models/` - output directory created at runtime to store best model files (.h5).
- `requirements.txt` - runtime dependencies (e.g., tensorflow, numpy, matplotlib).

## Prerequisites
- Python 3.7+
- Recommended: a machine with a CUDA GPU for faster training (optional)
- Core Python packages: tensorflow, numpy, matplotlib (see `requirements.txt`).

## Installation (local)
1. Clone the repo:
   ```bash
   git clone https://github.com/Anurag0798/CNN_Benchmark_Architectures.git

   cd CNN_Benchmark_Architectures
   ```

2. (Optional) Create & activate a virtual environment:
   ```bash
   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quickstart - train & evaluate
Run the main training harness which will:
- load CIFAR-10 and normalize images,
- compile each model and train with ModelCheckpoint,
- evaluate best checkpoints on the CIFAR-10 test set.

```bash
python app.py
```

The script will create `save_models/` (if missing) and save best models as `save_models/LeNet.h5`, `save_models/AlexNet.h5`, etc.

## Training configuration (default settings)
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy
- Metrics: accuracy
- Epochs: 50 (configured in the harness)
- Batch size: 64
- Validation split: 0.2
- Checkpoint: monitor `val_accuracy`, `save_best_only=True`

Adjust these parameters in `app.py` as needed.

## Inference (example)
Load a saved model for inference:
```python
from tensorflow.keras.models import load_model
model = load_model('save_models/LeNet.h5')

# Preprocess input (CIFAR-10 normalization) then:
preds = model.predict(X_batch)
```

## Tips & suggestions
- Use a GPU-enabled environment to significantly reduce training time.
- Log training histories (loss/accuracy) to compare architectures visually.
- Extend models with augmentation, learning rate schedules, weight decay, or more epochs for improved performance.
- Pin package versions in `requirements.txt` for reproducibility.

## Troubleshooting
- OOM (out-of-memory): reduce batch size or use a GPU with more memory.
- Model load errors: ensure TensorFlow/Keras versions are compatible between save/load.
- Missing dependencies: ensure `requirements.txt` includes `tensorflow`, `numpy`, etc.

## Contributing
Contributions are welcome:
1. Fork the repository.
2. Create a feature branch.
3. Implement and test changes.
4. Open a pull request with a clear description.

## License
MIT License added. Please check the License file for more details.