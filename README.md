# RESMA: Foundation Model Selection Agent for Remote Sensing

**RESMA** is a LLM agent that assists in selecting foundation models specifically for remote sensing applications. It uses a reasoning-driven approach based on model metadata to suggest the most appropriate models for your task.

---

## 📦 Installation

Clone the repository and set up your virtual environment using Conda:

```bash
conda env create -f environment_fms_agent.yaml
conda activate fms_agent
```

This installs all dependencies defined in the environment file.

---

## ⚙️ Configuration

RESMA is configurable through the `config.py` file. Here, you can:

- Select the LLM backend
- Config the LLM settings
- Specify your database
- Setup output threshold and customize the output

---

## 🚀 Usage

To run RESMA, simply execute:

```bash
python main.py
```

You will be prompted to provide input through the terminal. Follow the instructions and RESMA will return foundation model recommendations based on your criteria.

---

## 📂 Data

- `/data/`: Contains the processed internal database used during runtime.
- `/data/queries.txt`: Contains all the example queries we drafted for our experiments.
- `model_metadata/`: Contains raw metadata for available foundation models in remote sensing.

You can update or extend these files to include new models or metadata.
