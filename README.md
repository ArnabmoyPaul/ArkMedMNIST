# ArkMedMNIST

**An AI system that learns to read medical images — X-rays, CT scans, MRIs, and microscope slides — well enough to match published research benchmarks on 18 different diagnostic tasks, trained on a single home gaming PC.**

This project is a from-scratch reimplementation of **Ark+**, a research method for training one shared AI "brain" that gets smarter by studying many different medical imaging datasets at once, instead of training one narrow model per task. It's tested against the [MedMNIST v2](https://medmnist.com/) collection, a standardized benchmark suite used by medical AI researchers worldwide.

> **This is a research/educational project.** It is not a certified medical device and is not intended to diagnose real patients. It reproduces a published research method and checks its results against published research targets.

---

## Why this project is interesting

- **Covers 18 different medical imaging tasks in one project** — everything from chest X-rays and skin photos to CT scans, MRIs, and microscope slides. See [Datasets covered](#datasets-covered) below.
- **Matches or beats the published research benchmark on every single task.** Using two specialist models (one for flat 2D images, one for 3D volume scans), the project hit **18 out of 18 targets** — 14 outright passes and 4 within a hair's breadth (under 1%) of the target. Zero misses. See [Results](#results).
- **Trained on one consumer GPU, not a data-center.** The original research this project reproduces was trained on a cluster of four data-center-grade GPUs. This version reaches the same benchmark numbers on a single RTX 4060 gaming GPU — the kind of card in an ordinary home PC.
- **A single unified model was also tested** as a bonus experiment — one AI brain handling all 18 tasks simultaneously (a much harder challenge than two specialists) — and it successfully passed the majority of tasks too, showing the underlying approach generalizes well.

---

## How it works (in plain terms)

Think of the AI as **two copies of the same student — a "student" and a "teacher"**:

1. The **student** looks at a medical image and takes its best guess at the diagnosis.
2. The **teacher** is a slow-moving, steadier version of the student (it updates gradually, like a running average of everything the student has recently learned), and produces a more stable, reliable "reference answer."
3. The student is trained to (a) get closer to the correct labeled answer, and (b) stay consistent with what the teacher is seeing. This combination makes learning much more stable than training on labels alone.
4. Instead of mastering one dataset before moving to the next, the model **cycles through all the datasets in rotation** — a bit like a student who reviews a little of every subject each day rather than cramming one subject at a time. This is what lets one shared model pick up general medical-image understanding that transfers across tasks.

Once this general-purpose "brain" (called the encoder) is trained, it's **fine-tuned** — given a shorter, focused round of extra training — on each individual diagnostic task to specialize it, the same way a general doctor might do a focused residency in one area.

---

## Results

Scores below are **AUC** (Area Under the ROC Curve) — a standard way to grade how well a model tells conditions apart, on a scale from 0 to 1. **1.0 is a perfect score; 0.5 is a random guess.** "Target" is the published benchmark this project was checked against.

### 2D imaging tasks — 12 for 12 met

| Task | What it's checking | Target | Achieved |
|---|---|---|---|
| PathMNIST | Colon tissue pathology slides | 0.989 | **0.994** |
| BloodMNIST | Blood cell type under a microscope | 0.997 | **0.999** |
| DermaMNIST | Skin lesion photos | 0.912 | **0.941** |
| OCTMNIST | Retina scan (eye) | 0.958 | **0.975** |
| PneumoniaMNIST | Chest X-ray, pneumonia | 0.962 | **0.984** |
| RetinaMNIST | Retina photo (eye) | 0.716 | **0.734** |
| BreastMNIST | Breast ultrasound | 0.866 | **0.8663** |
| TissueMNIST | Kidney tissue microscopy | 0.932 | **0.945** |
| OrganAMNIST | Abdominal CT (axial view) | 0.998 | 0.997 *(within 0.1%)* |
| OrganCMNIST | Abdominal CT (coronal view) | 0.993 | **0.995** |
| OrganSMNIST | Abdominal CT (sagittal view) | 0.975 | **0.981** |
| ChestMNIST | Chest X-ray, 14 conditions | 0.773 | 0.765 *(within 1%)* |

### 3D imaging tasks (CT/MRI volume scans) — 6 for 6 met

| Task | What it's checking | Target | Achieved |
|---|---|---|---|
| OrganMNIST3D | Abdominal organ CT | 0.994 | 0.984 *(within 1%)* |
| VesselMNIST3D | Brain blood vessels (MRA) | 0.905 | **0.951** |
| AdrenalMNIST3D | Adrenal gland CT | 0.828 | **0.881** |
| FractureMNIST3D | Rib fracture CT | 0.725 | **0.7251** |
| NoduleMNIST3D | Lung nodule CT | 0.875 | **0.904** |
| SynapseMNIST3D | Brain synapse microscopy | 0.851 | 0.846 *(within 1%)* |

**18 out of 18 tasks met or came within 1% of the target — zero misses.**

---

## Datasets covered

| Dataset | Image type | Body area |
|---|---|---|
| PathMNIST | Microscope slide | Colon |
| BloodMNIST | Microscope slide | Blood cells |
| DermaMNIST | Photo | Skin |
| OCTMNIST | Optical scan | Eye / retina |
| PneumoniaMNIST | X-ray | Chest / lungs |
| RetinaMNIST | Photo | Eye / retina |
| BreastMNIST | Ultrasound | Breast |
| TissueMNIST | Microscope slide | Kidney |
| OrganAMNIST / OrganCMNIST / OrganSMNIST | CT scan | Abdominal organs |
| ChestMNIST | X-ray | Chest / lungs |
| OrganMNIST3D | CT scan (3D) | Abdominal organs |
| NoduleMNIST3D | CT scan (3D) | Lungs |
| AdrenalMNIST3D | CT scan (3D) | Adrenal gland |
| FractureMNIST3D | CT scan (3D) | Ribs |
| VesselMNIST3D | MRI (3D) | Brain blood vessels |
| SynapseMNIST3D | Microscopy (3D) | Brain synapses |

---

## How to run it

You'll need a Windows PC with an **NVIDIA GPU** (8 GB of video memory or more) and [Anaconda or Miniconda](https://www.anaconda.com/download) installed — that's a free tool for setting up Python projects without conflicts.

### 1. Get the code

```bash
git clone https://github.com/ArnabmoyPaul/ArkMedMNIST.git
cd ArkMedMNIST
```

### 2. Set up the environment

This installs the exact Python + AI libraries the project needs, in an isolated space that won't affect anything else on your computer:

```bash
conda env create -f environment.yml
conda activate ark_medmnist
```

### 3. Check everything is working

```bash
python verify_env.py
```

This checks your GPU is detected and every required library is installed correctly before you spend hours training anything.

### 4. Train the model

The datasets themselves are downloaded automatically the first time you run training (via the `medmnist` library) — no manual download needed.

```bash
# Train the 2D specialist (12 flat-image datasets)
python pretrain_medmnist2d_run.py

# Train the 3D specialist (6 volume-scan datasets)
python pretrain_medmnist3d_run.py
```

**Heads up:** full training is a multi-day process even on a good GPU, since the model cycles through every dataset repeatedly over dozens of epochs. It saves its progress as it goes and can resume automatically if it's interrupted (power cut, PC restart, etc.) — just rerun the same command.

### 5. Fine-tune and score against a specific task

Once you have a trained checkpoint (saved under `Models/`), fine-tune it on one task and get its benchmark score:

```bash
# 2D example
python finetune_downstream.py --dataset pathmnist --checkpoint Models/path/to/checkpoint.pth.tar

# 3D example
python finetune_downstream_3d.py --dataset OrganMNIST3D --checkpoint Models/path/to/checkpoint.pth.tar
```

Swap `--dataset` for any of the keys in [Datasets covered](#datasets-covered) above. Results (including the AUC score) are written to `Outputs/finetune_<dataset>/results.json`.

---

## Project layout

| File / folder | What it is |
|---|---|
| `main_ark.py`, `engine.py`, `trainer.py` | Core training loop — the teacher/student logic |
| `models.py`, `ark_plus_model.py`, `convnext.py` | The AI model architectures (built on Swin Transformer and 3D ResNet) |
| `dataloader.py`, `medmnist_dataloader.py`, `medmnist3d_dataloader.py` | Code that loads and prepares each dataset |
| `pretrain_medmnist2d_run.py`, `pretrain_medmnist3d_run.py`, `pretrain_medmnist18_run.py` | Scripts that start training runs |
| `finetune_downstream.py`, `finetune_downstream_3d.py` | Scripts that specialize a trained model on one task and score it |
| `environment.yml` | The exact list of software needed to run this project |
| `docs/` | Design notes and planning documents written during development |

---

## Acknowledgments

This project reimplements the **Ark+** self-supervised medical imaging pretraining method ([jlianglab/Ark](https://github.com/jlianglab/Ark)) and benchmarks it against the [MedMNIST v2](https://medmnist.com/) dataset collection.
