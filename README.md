# 🧠 AI-Studio-ClearML

This repository provides a minimal, reproducible example of how to use [ClearML](https://clear.ml) to build machine learning pipelines, track experiments, and manage datasets using both **task-based pipelines** and **function-based pipelines**.

---

## 📦 Project Structure

```
AI-Studio-ClearML/
├── .github/workflows/
│   └── pipeline.yaml                  # CI/CD: runs s1 on PR to main
│
├── model_artifacts/                   # Example outputs or saved models
├── work_dataset/                      # Dataset samples (Iris.csv)
│
│── ─── Demo 1: Basic Pipeline (s1 → s3) ───
├── s1_dataset_artifact.py             # Step 1: Upload dataset as pickle artifact
├── s2_data_preprocessing.py           # Step 2: Preprocess (artifact API)
├── s3_train_model.py                  # Step 3: Train model (hardcoded params)
├── pipeline_from_tasks.py             # 3-step pipeline orchestrator
│
│── ─── Demo 2: HPO Pipeline (s1 → final model) ───
├── hpo_s1_dataset_artifact.py         # Step 1: Upload dataset (ClearML Dataset API)
├── hpo_s2_process_dataset.py          # Step 2: Preprocess (ClearML Dataset API)
├── hpo_s3_train_model.py              # Step 3: Train model (parameterized for HPO)
├── task_hpo.py                        # Step 4: Hyperparameter optimization
├── final_model.py                     # Step 5: Train final model with best params
├── pipeline_hpo.py                    # 5-step pipeline orchestrator
│
│── ─── Shared ───
├── main.py                            # Entry point (runs pipeline_from_tasks)
├── requirements.txt                   # Pinned Python dependencies
├── AI-Studio-Agent.ipynb              # Start/stop ClearML Agent daemon
├── AI-Studio-ClearML.ipynb            # End-to-end demo notebook
├── AI-Studio-ClearML_HPO_ZOE.ipynb    # HPO demo notebook (Colab)
└── ClearML_Pipeline_Demo.ipynb        # Task-based pipeline demo notebook
```

---

## 🧪 Features

- ✅ Task-based pipeline using `PipelineController.add_step(...)`
- ✅ Hyperparameter Optimization (HPO) with ClearML `HyperParameterOptimizer`
- ✅ Final model retraining with best HPO parameters
- ✅ CI/CD pipeline via GitHub Actions
- [TBD] Function-based pipeline using `PipelineController.add_function_step(...)`
- ✅ Reusable ClearML Task templates
- ✅ Dataset and model artifact management with ClearML
- ✅ End-to-end ML workflow: Dataset → Preprocessing → Training → HPO → Final Model
- ✅ Fully compatible with ClearML Hosted and ClearML Server

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure ClearML

Set up ClearML by running:

```bash
clearml-init
```

You will be prompted to enter:
- ClearML Credential

Use [https://app.clear.ml](https://app.clear.ml) to register for a free account if needed.

---

### 3. Create a ClearML Agent
- Install the ClearML agent on your machine or server.
```bash
pip install clearml-agent
```
---

## 🛠️ How to Use

### Using Colab: refer to ClearML_Pipeline_Demo.ipynb.

---

### 🔁 Demo 1: Basic Pipeline (3 Steps)

A simple pipeline demonstrating ClearML task-based pipelines with dataset artifacts.

#### Step 1: Register the Base Tasks

**Before running the pipeline, execute the following scripts once to create reusable ClearML Tasks:**
> **Note:** When running for the first time, comment out `task.execute_remotely()` in each .py file to successfully create a task template.

```bash
# Step 1: Upload dataset
python s1_dataset_artifact.py

# Step 2: Preprocess dataset
python s2_data_preprocessing.py

# Step 3: Train model
python s3_train_model.py
```

These will appear in your ClearML dashboard and serve as base tasks for the pipeline.

#### Step 1.5: Initial ClearML Queue
Create Queue with name as `basic_demo` (or your customized one), ensure it is consistent in `pipeline_from_tasks.py`:
```python
pipe.set_default_execution_queue("basic_demo")
```

Run the agent for queue worker:
```bash
clearml-agent daemon --queue "basic_demo" --detached
```

#### Step 2: Run the Pipeline

Once all base tasks are registered, run the pipeline:

```bash
python main.py  # Executes run_pipeline() from pipeline_from_tasks.py
```

---

### 🧬 Demo 2: HPO Pipeline (5 Steps)

An advanced pipeline that adds hyperparameter optimization and final model retraining.
Uses the ClearML Dataset API for more robust data management.

#### Step 1: Register the HPO Base Tasks

> **Note:** When running for the first time, comment out `task.execute_remotely()` in each .py file to successfully create a task template.

```bash
# Step 1: Upload dataset (ClearML Dataset API)
python hpo_s1_dataset_artifact.py

# Step 2: Preprocess dataset (ClearML Dataset API)
python hpo_s2_process_dataset.py

# Step 3: Train model (parameterized for HPO)
python hpo_s3_train_model.py

# Step 4: Hyperparameter optimization
python task_hpo.py

# Step 5: Final model with best parameters
python final_model.py
```

#### Step 1.5: Initial ClearML Queue
Create Queue with name as `hpo_demo` (or your customized one), ensure it is consistent in `pipeline_hpo.py`:
```python
EXECUTION_QUEUE = "hpo_demo"
```

Run the agent for queue worker:
```bash
clearml-agent daemon --queue "hpo_demo" --detached
```

#### Step 2: Run the HPO Pipeline

```bash
python pipeline_hpo.py
```

---

### 🔧 [TBD] Option 3: Pipeline from Local Python Functions

This version demonstrates using `add_function_step(...)` to wrap Python logic as pipeline steps.

---

### ⚙️ CI/CD

The repository includes a GitHub Actions workflow (`.github/workflows/pipeline.yaml`) that:
1. Triggers on pull requests to `main`
2. Sets up Python 3.10 and installs dependencies
3. Verifies ClearML connectivity using GitHub Secrets
4. Runs `s1_dataset_artifact.py` as a smoke test

**Required GitHub Secrets:**
- `CLEARML_API_ACCESS_KEY`
- `CLEARML_API_SECRET_KEY`
- `CLEARML_API_HOST`

---

## 📘 References

- [ClearML Documentation](https://clear.ml/docs)
- [ClearML Pipelines Guide](https://clear.ml/docs/latest/docs/getting_started/building_pipelines)
- [ClearML GitHub](https://github.com/allegroai/clearml)

---

## 🙌 Acknowledgments

This project is developed and maintained by:

- **Jacoo-Zhao** (GitHub: [@Jacoo-Zhao](https://github.com/Jacoo-Zhao))
- **Zoe Lin** (Github: [@Zoe Lin](https://github.com/ZoeLinUTS))

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
