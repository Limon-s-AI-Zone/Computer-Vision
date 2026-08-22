# Architecture, Concepts & Interview Guide

A companion to `readmd.md` (the project README). That file tells you *how to run*
this project; this file explains *how it's built* and *why*, plus the underlying
ML concepts, so you can defend every design decision in an interview or a code review.

## Table of Contents
- [1. Architecture Overview](#1-architecture-overview)
- [2. Module-by-Module Walkthrough](#2-module-by-module-walkthrough)
- [3. Software Design Patterns Used](#3-software-design-patterns-used)
- [4. ML/DL Concepts & Definitions](#4-mldl-concepts--definitions)
- [5. Evaluation Metrics Explained](#5-evaluation-metrics-explained)
- [6. Interview Questions & Answers](#6-interview-questions--answers)

---

## 1. Architecture Overview

The project separates four concerns that are usually tangled together in a
notebook: **data**, **model**, **training loop**, and **experiment configuration**.
Each is independent and swappable without touching the others.

```
                     ┌─────────────────────┐
                     │   configs/*.yaml    │   <- experiment definition
                     │ (dataset+model+hp)  │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │     src/train.py     │   <- CLI orchestrator
                     └──┬────────┬───────┬──┘
             ┌──────────┘        │       └──────────┐
             ▼                   ▼                  ▼
   ┌───────────────────┐ ┌───────────────┐ ┌──────────────────┐
   │  src/dataio/         │ │ src/models/   │ │ src/engine/       │
   │  registry+loaders  │ │ registry+nets │ │ Trainer + metrics │
   └───────────────────┘ └───────────────┘ └──────────────────┘
             │                   │                  │
             └─────────► DataLoader, nn.Module, EvalResult ◄─────┘
                                │
                     ┌──────────▼──────────┐
                     │   outputs/ + checkpoints/  │  <- artifacts
                     └─────────────────────┘
```

**The core idea:** `train.py` never imports a specific dataset class or model
class. It asks the *registries* ("give me whatever is named `mnist`", "give me
whatever is named `resnet18`") and gets back objects that satisfy a known
interface (`(train_ds, test_ds)` tuple; `nn.Module` with `forward(x)`). This is
what makes "switch the architecture" or "switch the dataset" a one-line config
change instead of a code change.

---

## 2. Module-by-Module Walkthrough

### `src/dataio/registry.py` + `src/dataio/datasets.py`
- `registry.py` defines `@register_dataset(name, in_channels, image_size, num_classes)`
  — a decorator that stores a *factory function* in a dict (`DATASET_REGISTRY`).
- `datasets.py` calls that decorator once per dataset (MNIST, FashionMNIST,
  KMNIST, CIFAR-10, generic `ImageFolder`). Each factory takes
  `(root, transform, download)` and returns `(train_dataset, test_dataset)`.
- `dataset.py`'s `get_dataloaders()` looks up the entry by name, builds the
  datasets, does a reproducible train/valid split (via `SubsetRandomSampler` +
  a seeded `RandomState`), and wraps everything in `DataLoader`s.

### `src/models/registry.py` + `cnn.py` / `mlp.py` / `resnet.py`
- Same registry pattern as data, but for architectures. `@register_model("name")`
  on a class adds it to `MODEL_REGISTRY`; `build_model(name, **kwargs)` instantiates it.
- Every model takes `(in_channels, num_classes)` in its constructor and outputs
  raw logits of shape `(batch, num_classes)` — that's the whole contract the
  trainer relies on.

### `src/engine/trainer.py`
- `Trainer` owns the model, optimizer, criterion, and device. It has three jobs:
  1. `train_one_epoch()` — one pass over the training set, gradient updates.
  2. `evaluate()` — one pass with `torch.no_grad()`, returns an `EvalResult`
     dataclass (accuracy, precision/recall/f1, top-k accuracy, confusion matrix,
     per-class report, raw probabilities+labels for ROC/PR curves, loss).
  3. `fit()` — the outer loop: for each epoch, train then validate, print
     progress, and checkpoint whenever validation loss improves ("early-stopping
     checkpoint" pattern — always keep the best model seen, not just the last).

### `src/engine/metrics.py`
Pure functions, no framework state: `calculate_accuracy`, `precision_recall_f1`,
`confusion_matrix`, `top_k_accuracy`, `classification_report` (wraps
scikit-learn's version). Kept separate from `Trainer` so they're independently
testable (see `tests/test_metrics.py`).

### `src/utils/`
- `config.py` — loads YAML, deep-merges CLI `--set key=value` overrides on top,
  and substitutes `{model_name}`/`{dataset_name}` into output paths so every
  run's artifacts are uniquely named.
- `checkpoint.py` — `save_checkpoint`/`load_checkpoint`: serializes model +
  optimizer state + epoch + best metrics, so training can resume exactly where
  it left off.
- `seed.py` — seeds Python/NumPy/PyTorch RNGs for reproducibility.
- `visualize.py` — all matplotlib code (loss/accuracy curves, confusion matrix,
  ROC/PR curves), each with a `save_path` and `show` flag so it works headlessly
  in a CLI script (`show=False`) or interactively in a notebook (`show=True`).

### `src/train.py` / `src/predict.py`
CLI entrypoints. `train.py` wires everything above together and writes
`results.json` + all plots. `predict.py` loads a checkpoint and classifies a
single image. Both are "thin" — no business logic lives here, just wiring.

---

## 3. Software Design Patterns Used

| Pattern | Where | Why |
|---|---|---|
| **Registry / Plugin** | `models/registry.py`, `data/registry.py` | Add new models/datasets without editing the trainer, CLI, or existing model files. Open for extension, closed for modification (the "O" in SOLID). |
| **Factory function** | `build_model()`, dataset factories | Centralizes object construction so callers depend on a name (string), not a concrete class. |
| **Dependency injection via config** | `configs/default.yaml` + `--set` overrides | Hyperparameters and architecture choice are data, not code — reproducible, diffable, scriptable (e.g. for sweeps). |
| **Strategy pattern** | Any registered model is interchangeable inside `Trainer` | `Trainer` only depends on the `nn.Module` interface (`forward(x) -> logits`), not on any specific architecture. |
| **Dataclass as a typed return value** | `EvalResult` | Avoids "mystery tuples" (`return acc, prec, rec, f1, cm, ...`) that break silently when reordered; call sites use `.accuracy`, `.confusion_matrix`, etc. |
| **Separation of concerns** | `data/` vs `models/` vs `engine/` vs `utils/` | Each module can be tested, replaced, or reasoned about independently. |

---

## 4. ML/DL Concepts & Definitions

**Epoch** — one full pass over the entire training dataset.
**Batch / batch size** — number of samples processed together before one
gradient update; bigger batches = more stable gradients but more memory and
(often) worse generalization at very large sizes.
**Forward pass** — computing predictions from input through the network.
**Backward pass (backpropagation)** — computing gradients of the loss w.r.t.
every parameter via the chain rule.
**Loss function** — a differentiable scalar measuring how wrong the
predictions are. This project uses `CrossEntropyLoss`, which combines
`log_softmax` + negative log-likelihood — appropriate for multi-class,
single-label classification with raw logits as input (never apply softmax
yourself before this loss, or you'll double-apply it).
**Optimizer** — the update rule for parameters given gradients.
  - **SGD**: `param -= lr * grad`. Simple, can need momentum/tuning to converge well.
  - **Adam**: adapts a per-parameter learning rate using running estimates of the
    gradient's mean and variance (first/second moments). Converges faster and is
    less sensitive to learning-rate choice, at the cost of more memory (stores
    two extra tensors per parameter) and sometimes worse final generalization
    than well-tuned SGD+momentum.
**Learning rate** — step size for parameter updates. Too high → diverges/oscillates;
too low → slow convergence or stuck in poor local minima/plateaus.
**Weight decay (L2 regularization)** — penalizes large weights, added to the
loss (or applied directly to the update in decoupled variants like AdamW) to
reduce overfitting.
**Overfitting** — model fits training data (including its noise) so well that
it generalizes poorly to unseen data; shows as training loss ↓ but validation
loss ↑ (or flattens while train keeps dropping).
**Underfitting** — model is too simple / undertrained to capture the pattern;
both train and validation loss stay high.
**Dropout** — randomly zeroes activations during training (not at eval time) so
the network can't rely on any single neuron, reducing overfitting. Used in
`deep_cnn` and `mlp`.
**Batch Normalization** — normalizes layer activations per mini-batch (then
learns a scale/shift), stabilizing and speeding up training; used in `deep_cnn`.
**Convolution** — a learned filter slides over the input, computing local dot
products; captures spatial patterns (edges, textures) with far fewer parameters
than a fully-connected layer over the same input, because weights are shared
across positions.
**Kernel/filter size, stride, padding** — kernel size = filter width/height;
stride = step size when sliding; padding = zeros added around the input so
output size can be controlled (e.g. "same" padding keeps spatial size unchanged).
**Pooling (e.g. MaxPool)** — downsamples spatial dimensions, keeping the
strongest activation in each window; adds translation invariance and reduces
computation.
**Receptive field** — the region of the original input that a given neuron's
activation is computed from; grows with depth and with larger/stride convolutions.
**Fully-connected (Linear/Dense) layer** — every input unit connects to every
output unit; used at the end of a CNN to map extracted features to class scores,
or throughout an MLP.
**Softmax** — converts raw logits into a probability distribution over classes
(exponentiate, normalize to sum to 1). Used implicitly by `CrossEntropyLoss` and
explicitly in `Trainer.evaluate()` to get `probs` for ROC/PR curves.
**Transfer learning** — reusing a model pretrained on a large dataset (e.g.
ImageNet) as a starting point; `resnet18` in this project supports
`pretrained=True` to load ImageNet weights, only replacing the first conv layer
(for non-RGB input) and the final classifier head.
**Checkpoint** — a saved snapshot of model + optimizer state (+ metadata like
epoch/loss) so training can resume or the best model can be reloaded for inference.
**Seed / reproducibility** — fixing the RNG state (Python, NumPy, PyTorch) so
re-running produces the same shuffles/initializations/results (as much as the
hardware/backend allows).
**Train / validation / test split** — train to fit weights, validation to tune
hyperparameters and pick the best checkpoint (without touching test data),
test to report final unbiased performance. This project uses `SubsetRandomSampler`
to carve validation out of the training set (`valid_size` in config).

---

## 5. Evaluation Metrics Explained

**Accuracy** — fraction of correct predictions. Misleading on imbalanced classes
(e.g. 95% accuracy is trivial if 95% of samples are one class).

**Precision** (per class) — of everything predicted as class *c*, what fraction
actually is *c*? `TP / (TP + FP)`. High precision = few false alarms.

**Recall** (a.k.a. sensitivity, true positive rate) — of everything that
actually is class *c*, what fraction did we catch? `TP / (TP + FN)`. High recall
= few misses.

**Precision/Recall trade-off** — raising the decision threshold for a class
usually increases precision but decreases recall, and vice versa; F1 and the
PR curve summarize this trade-off.

**F1 score** — harmonic mean of precision and recall: `2PR / (P+R)`. Punishes
extreme imbalance between the two more than a simple average would.

**Macro average** — average the per-class metric with equal weight per class,
regardless of how many samples each class has. Sensitive to performance on rare
classes.

**Weighted average** — average per-class metrics weighted by each class's
support (sample count). Closer to overall accuracy when classes are imbalanced.

**Support** — number of true instances of a class in the evaluation set.

**Confusion matrix** — an `num_classes × num_classes` grid; row = true label,
column = predicted label. The diagonal is correct predictions; off-diagonal
cells show exactly which classes get confused with which (e.g. "4 mistaken for
9" is a classic MNIST confusion).

**Top-k accuracy** — counts a prediction correct if the true label is among the
model's *k* highest-probability classes, not just its single top guess. Useful
when several classes are visually similar or when downstream use allows a
shortlist (e.g. "show top 3 suggestions").

**ROC curve (Receiver Operating Characteristic)** — plots True Positive Rate vs
False Positive Rate as the classification threshold varies (per class, one-vs-rest
here since this is multi-class). **AUC** (Area Under the Curve) summarizes it in
one number: 1.0 = perfect ranking, 0.5 = random guessing.

**Precision-Recall (PR) curve** — plots precision vs recall as threshold varies.
**More informative than ROC when classes are imbalanced**, because ROC's false
positive rate can look good even with many false positives if the negative
class is huge — PR curves surface that directly. **AP (Average Precision)**
summarizes the PR curve in one number (area under it, computed at every recall
level from the ranked scores — not by trapezoidal interpolation).

**Loss curve** — train vs validation loss per epoch. The gap between them is
the standard visual diagnostic for overfitting; where validation loss stops
decreasing (or starts rising) is where the "early stopping checkpoint" in this
project saves the model.

**Accuracy curve** — same idea as the loss curve but in accuracy terms; loss is
usually preferred for early-stopping decisions because it's smoother/more
sensitive than accuracy (which is a step function over predictions).

---

## 6. Interview Questions & Answers

### General deep learning

**Q: Why use `CrossEntropyLoss` instead of applying softmax + a manual NLL loss?**
A: `CrossEntropyLoss` combines `LogSoftmax` + `NLLLoss` in a numerically stable
way (using the log-sum-exp trick internally), avoiding overflow/underflow you'd
risk computing softmax and log separately. It also expects raw logits, so
models in this project never apply softmax in their `forward()` — softmax is
only applied explicitly at evaluation time to get interpretable probabilities.

**Q: Why did you choose Adam over SGD (or vice versa)?**
A: Adam adapts per-parameter learning rates from gradient moment estimates, so
it converges quickly with minimal tuning — good for fast iteration across many
model/dataset combinations, which is this project's use case. Plain SGD (+momentum)
can generalize slightly better on some vision tasks with careful LR scheduling,
but needs more tuning. The config exposes `train.optimizer` so this is a
one-line experiment, not a hard commitment.

**Q: What's the difference between overfitting and underfitting, and how do you
detect each from the loss curve?**
A: Underfitting: both train and validation loss stay high — the model lacks
capacity or hasn't trained enough. Overfitting: train loss keeps dropping while
validation loss plateaus or rises — the model is memorizing training-set noise.
This project's `plot_loss` marks the epoch of minimum validation loss, which is
also the checkpoint that gets saved.

**Q: Why checkpoint on validation loss instead of validation accuracy?**
A: Loss is continuous and reflects the model's confidence, not just whether the
top prediction was right — it's more sensitive to small improvements and less
noisy epoch-to-epoch than accuracy, which only changes when a prediction flips
across the decision boundary.

**Q: What does batch normalization actually do, and why does it help?**
A: It normalizes each layer's activations (per batch, per channel) to zero mean
/ unit variance, then applies a learned scale and shift. This reduces internal
covariate shift (the input distribution to each layer changing as earlier
layers update), which lets you use higher learning rates and often acts as a
mild regularizer too.

**Q: Why does `resnet18` need its first conv layer replaced for MNIST?**
A: torchvision's ResNet-18 is built for 3-channel (RGB) ImageNet input. MNIST is
1-channel grayscale, so the very first `Conv2d(3, 64, ...)` layer's input
channel count would mismatch. `src/models/resnet.py` rebuilds that first layer
with `in_channels` from the config/dataset registry, keeping the rest of the
pretrained architecture (and optionally pretrained weights for everything past
that layer) intact.

**Q: Why macro-average precision/recall/F1 instead of micro or weighted?**
A: Macro-average treats every class equally regardless of how many samples it
has, which surfaces poor performance on rare/hard classes that a
sample-weighted (or micro, which reduces to accuracy in single-label
multi-class settings) average would hide. It's the right default when you care
about all classes equally (e.g. digits 0-9 are equally important); switch to
weighted if some classes matter more because they're more common in production.

**Q: When would you look at a PR curve instead of an ROC curve?**
A: When the positive class is rare (imbalanced data). ROC's false-positive rate
is computed against a large negative class, so it can look deceptively good
even with many false positives in absolute terms. PR curves plot precision
directly, which drops visibly in that situation. For roughly balanced classes
(like MNIST digits), ROC and PR tend to agree; the project reports both so it
generalizes to imbalanced datasets (e.g. a custom `image_folder` dataset) without
code changes.

### Project / systems design

**Q: Why a registry pattern instead of an `if/elif` chain choosing the model
class?**
A: An `if/elif` in the training script means every new architecture requires
editing that shared file, risking regressions to existing experiments and
creating merge conflicts as more people add models. The registry lets each
model file register itself independently — `train.py` and `Trainer` never
change when a new architecture is added, satisfying the open/closed principle.

**Q: What's the contract between `Trainer` and a model, that lets any
registered architecture "just work"?**
A: Any `nn.Module` whose constructor accepts `(in_channels, num_classes)` and
whose `forward(x)` returns `(batch, num_classes)` logits. `Trainer` never
inspects a model's internals — it only calls `model(data)`, `.train()`,
`.eval()`, `.parameters()` (via the optimizer), and `.state_dict()` (via
checkpointing). This is why `simple_cnn`, `mlp`, and `resnet18` are all
interchangeable despite very different internal architectures.

**Q: Why key checkpoint/output filenames by both model name AND dataset name?**
A: Without the dataset name, training `simple_cnn` on MNIST and then on
FashionMNIST would silently overwrite the first run's checkpoint and plots —
you'd lose the earlier experiment with no warning. Namespacing by
`{model}_{dataset}` makes every (architecture, dataset) combination's artifacts
independent and comparable side-by-side.

**Q: Why does `in_channels`/`num_classes` come from the dataset registry instead
of being hardcoded per model?**
A: So the same model class works unmodified across datasets with different
channel counts (1 for MNIST, 3 for CIFAR-10) and class counts. If it were
hardcoded in the model file, adding a new dataset would require also editing
every model file — reintroducing the coupling the registries are designed to
remove.

**Q: Why does `load_checkpoint_enabled=False` still allow saving a checkpoint?**
A: They're separate concerns: "should I resume from a previous run" (load) vs
"should I persist progress from this run" (save). Tying them together was a
bug fixed during this project's development — disabling resume shouldn't also
silently disable persisting the new run's best model.

**Q: Why does `plot_*` take both `save_path` and `show`?**
A: The same plotting code is used from a notebook (where `plt.show()` renders
inline and is wanted) and from the CLI training script (where a headless
server has no display, and calling `plt.show()` there would hang or block
until it eventually errors/times out). Making `show` an explicit flag rather
than auto-detecting the environment keeps behavior predictable in both contexts.

**Q: How would you extend this to a completely custom dataset with, say, 37 classes?**
A: Register it with `@register_dataset("my_data", in_channels=3, image_size=224, num_classes=37)`
in `src/dataio/datasets.py` pointing at an `ImageFolder`-style directory (or a
custom `Dataset` subclass), then run
`python -m src.train --set data.name=my_data model.name=resnet18`. No other
file changes — `in_channels`/`num_classes` propagate automatically to model
construction, the trainer, and every metric/plot function since they all read
`num_classes` from the same place.
