# Zero-Shot Compound Fault Diagnosis

This repository implements a two-stage zero-shot learning (ZSL) framework for composite fault diagnosis using vibration signals.

## 📁 Structure

- `config.py`: Dataset and label configuration.
- `data_loader.py`: Load `.mat` files and prepare FFT samples.
- `models/`: Embedding network and transformer-based synthesizer.
- `trainer/`: Training logic for both embedding and synthesis.
- `evaluator.py`: Functions for evaluation and reporting.
- `evaluate_best_model.py`: **Standalone script to evaluate saved best model**
- `main.py`: Main training + testing entry point.
- `visualize_tsne.py`: Visualize semantic embedding.
- `example.ipynb`: Sample notebook demonstration.

## 🚀 Usage

### 🧠 Train the model
```bash
python main.py --mode train
```

### 🧪 Evaluate saved model (in main.py)
```bash
python main.py --mode test --model_path best_model.pt
```

### 🧪 Or run standalone evaluation
```bash
python evaluate_best_model.py
```

This will:
- Load `best_model.pt`
- Evaluate all 13 test domains
- Report per-domain accuracy
- Report per-class Precision, Recall, F1-score
- Summarize average performance

## 📦 Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- seaborn
- matplotlib

## 📬 Contact

For any issues or questions, please contact the project maintainer.