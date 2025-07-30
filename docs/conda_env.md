# Conda Environment Setup for `openmmlab`

This guide documents how to recreate and use the Conda environment defined in `environment_full.yml`.

---

## 📦 Create the Environment

Clone your project and run:


```bash
conda env create -f environment_full.yml
```

## 🚀 Activate the Environment

```bash
conda activate mmopenlab
```

## 🔁 Update Environment

```bash
conda install (some_package)
```

Export the new spec:

```bash
conda env export > environment_full.yml
```

Then commit to git:

```bash
git add environment_full.yml
git commit -m "Update Conda environment"
```
