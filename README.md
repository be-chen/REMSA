# RESMA: Foundation Model Selection Agent for Remote Sensing

![Alt text](remsa.png?raw=true "Architecture of REMSA")

This repository contains the code of the paper [RESMA: Foundation Model Selection Agent for Remote Sensing](https://arxiv.org/abs/2511.17442). The work introduces RS-FMD, the first structured database of Remote Sensing Foundation Models (RSFMs), and REMSA, the first LLM-based agent designed to automatically select suitable foundation models for a given remote sensing task.

This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) and [BIFOLD](https://www.bifold.berlin/) by [Binger Chen](https://rsim.berlin/team/members/binger-chen), [Tacettin Emre Bök](https://rsim.berlin/team/members/tacettin-bok), [Behnood Rasti](https://rsim.berlin/team/members/behnood-rasti), [Volker Markl](https://www.bifold.berlin/people/prof-dr-volker-markl.html), and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> B. Chen, T. E. Bök, B. Rasti, V. Markl, B. Demir, "[RESMA: Foundation Model Selection Agent for Remote Sensing](https://arxiv.org/abs/2511.17442)", IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3517150, 2024

```bibtex
@misc{chen2025remsallmagentfoundation,
      title={REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing}, 
      author={Binger Chen and Tacettin Emre Bök and Behnood Rasti and Volker Markl and Begüm Demir},
      year={2025},
      eprint={2511.17442},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.17442}, 
}
```

---

## Overview

**RESMA** REMSA automates RS foundation model selection by combining:

- A structured database (RS-FMD) containing metadata for 150+ RSFMs
- A modular agent architecture with:
    - query interpretation
    - metadata-grounded retrieval
    - LLM-based candidate ranking
    - clarification loops
    - transparent explanations

The system supports diverse RS tasks and modalities and provides reproducible, user-tailored model recommendations.

---

## Environment Setup

Set up the environment using Conda:

```bash
conda env create -f environment_remsa.yaml
conda activate remsa
```

This installs all required dependencies defined in the environment file.

## Configuration

RESMA is fully configurable via the `config.py` file. The following parameters can be adjusted:

- LLM backend selection
- LLM inference settings
- Database and vector index paths
- Output formatting and thresholds

Adjust these parameters to suit your runtime environment and model preferences.

## Running the Agent

To launch the agent REMSA:

```bash
python main.py
```

You will be prompted to provide a natural-language query describing your RS application, data modality, constraints, and requirements.

REMSA then interprets your input, retrieves suitable candidate models from RS-FMD, ranks them, and outputs recommendations with explanations.

---

## Data Structure

### RS-FMD Model Metadata

- Located in `model_metadata/`
- Contains structured JSON model metadata extracted from papers, repositories, and model cards
- Supports 150+ Remote Sensing Foundation Models

### Internal Runtime Data

- `data/`: processed database used at runtime
- `data/queries.txt`: example queries used for experiments

You may extend these files with additional models or metadata.

---

## RS-FMD: Generating Foundation Model Metadata

To extract structured metadata for a new foundation model from its documentation:

```bash
# Navigate to the directory
cd file_to_db

# Run the script with your configuration and file path
python src/run.py --config FoundationModels.yaml --file_path path_to_your_pdf
```

The resulting JSON metadata file will be stored under:

```bash
/file_to_db/model_metadata
```

## Building a Vector Database

To build the vectorstore from your `.jsonl` metadata file:

```bash
python build_vectorstore.py
```

Configurable parameters in `config.py` include:

- **`FMD_JSONL_PATH`**: Path to your `.jsonl` file.  
- **`EMBEDDING_MODEL_NAME`**: Name of the embedding model to use.  
- **`VECTOR_INDEX_PATH`**: Path where the generated vector database will be stored. 

---

## Authors
**Binger Chen**
https://rsim.berlin/team/members/binger-chen

**Tacettin Emre Bök**
https://rsim.berlin/team/members/tacettin-bok

**Behnood Rasti**
https://rsim.berlin/team/members/behnood-rasti

**Volker Markl**
https://www.bifold.berlin/people/prof-dr-volker-markl.html

**Begüm Demir**
https://rsim.berlin/team/members/begum-demir

For questions, requests and concerns, please contact [Binger Chen](mailto:chen@tu-berlin.de)

## License

The code in this repository is licensed under the **MIT License**.  
See the `LICENSE` file for more details.