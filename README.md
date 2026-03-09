# 🧠 AI-Studio-ClearML

This repository provides a minimal, reproducible example of how to use [ClearML](https://clear.ml) to build machine learning pipelines, track experiments, and manage datasets using both **task-based pipelines** and **function-based pipelines**.

---

## 📦 Project Structure

```
├── AI-Studio-ClearML.ipynb           # End-to-end demo notebook
├── ClearML_Pipeline_Demo.ipynb       # Task-based pipeline demo notebook
├── clearml_start_agent.ipynb         # Start ClearML Agent daemon (queue configurable)
├── clearml_stop_agent.ipynb          # Stop ClearML Agent daemon
├── main.py                           # Entry point (runs `run_pipeline()`)
├── pipeline_from_tasks.py            # Pipeline built from existing ClearML Tasks
├── s1_dataset_artifact.py            # Step 1: Upload dataset as artifact
├── s2_data_preprocessing.py          # Step 2: Preprocess dataset
├── s3_train_model.py                 # Step 3: Train model using preprocessed data
├── task_hpo.py                       # Hyperparameter optimization example (Optuna)
├── requirements.txt                  # Pinned Python dependencies
├── model_artifacts/                  # Example outputs or saved models
└── work_dataset/                     # Dataset samples and usage examples
```

---

## 🧪 Features

- ✅ Task-based pipeline using `PipelineController.add_step(...)`
- [TBD] Function-based pipeline using `PipelineController.add_function_step(...)`
- ✅ Reusable ClearML Task templates
- ✅ Dataset and model artifact management with ClearML
- ✅ End-to-end ML workflow: Dataset → Preprocessing → Training
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

> **Queue note:** this repo’s pipeline is configured with `pipe.set_default_execution_queue("task")` in `pipeline_from_tasks.py`.  
> If you use an agent, start it with `--queue task` (or change the queue name in the code / notebooks to match).

---

## 🛠️ How to Use

### Using notebooks

- `ClearML_Pipeline_Demo.ipynb`: task-based pipeline walkthrough
- `AI-Studio-ClearML.ipynb`: additional end-to-end examples
- `clearml_start_agent.ipynb` / `clearml_stop_agent.ipynb`: start/stop a `clearml-agent` daemon (edit the queue name if needed)

### 🔁 Option 1: Pipeline from Predefined ClearML Tasks

To use a task-based pipeline, follow these steps:

#### Step 1: Register the Base Tasks

**Before running the pipeline, execute the following scripts \*\*once\*\* to create reusable ClearML Tasks:**
> **Note:** When running for the first time, comment out `task.execute_remotely()` in each step file to create the task template locally (without dispatching to an agent).

```bash
# Step 1: Upload dataset
python s1_dataset_artifact.py

# Step 2: Preprocess dataset
python s2_data_preprocessing.py

# Step 3: Train model
python s3_train_model.py
```

These will appear in your ClearML dashboard and serve as base tasks for the pipeline.

#### Step 1.5: (Optional) Run on a ClearML Agent queue

By default, `pipeline_from_tasks.py` runs the pipeline locally via `pipe.start_locally()` (no agent needed).

If you want remote execution:
- Start a ClearML agent listening on the same queue name as the pipeline’s default execution queue (currently `"task"`).
- In `pipeline_from_tasks.py`, comment out `pipe.start_locally()` and use `pipe.start(queue="task")` instead.

#### Step 2: Run the Pipeline

Once all base tasks are registered, run the pipeline:

```bash
python main.py # Where we execute the run_pipeline()
```

---

### 🔧 [TBD] Option 2: Pipeline from Local Python Functions 

This version demonstrates using `add_function_step(...)` to wrap Python logic as pipeline steps.

---

### 🧩 Run Individual Pipeline Steps

You can run each task separately as well:

> **Note:** When running for the first time, comment out `task.execute_remotely()` in the code file to successfully create a task template.
> **Note:** `s2_data_preprocessing.py` and `s3_train_model.py` require a `dataset_task_id` (the pipeline wires this automatically). If running them standalone, set `args["dataset_task_id"]` to a valid upstream task ID.
> 
```bash
# Step 1: Upload dataset
python s1_dataset_artifact.py

# Step 2: Preprocess data
python s2_data_preprocessing.py

# Step 3: Train model
python s3_train_model.py
```


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
