# RESMA: Foundation Model Selection Agent for Remote Sensing

**RESMA** is a LLM agent that assists in selecting foundation models specifically for remote sensing applications. It uses a reasoning-driven approach based on model metadata to suggest the most appropriate models for your task.

---

## 📦 Installation

Clone the repository and set up your virtual environment using Conda:

```bash
conda env create -f environment_fms_agent.yaml
conda activate fms
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


# Foundation Model Metadata Generation

This tool allows you to extract structured model metadata from your foundation model PDF files.

## Usage

If you want to upload your model PDF file and generate structured metadata, follow these steps:

```bash
# Navigate to the directory
cd file_to_db

# Run the script with your configuration and file path
python src/run.py --config FoundationModels.yaml --file_path path_to_your_pdf
```

## Output

After running the command, your model metadata will be automatically stored as a JSON file under:

```bash
/file_to_db/model_metadata
```

# Creating a Vector Database from a JSONL File

To build the vector database, run the following command:

```bash
python build_vectorstore.py
```

## Configuration

You can customize the following parameters in the `config.py` file:

- **`FMD_JSONL_PATH`** — Path to your `.jsonl` file.  
- **`EMBEDDING_MODEL_NAME`** — Name of the embedding model to use.  
- **`VECTOR_INDEX_PATH`** — Path where the generated vector database will be stored.  

