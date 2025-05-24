from clearml import Task
from clearml.automation import PipelineController
import logging
# import os
# os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY")
# os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY")
# os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration - using same queue for everything
EXECUTION_QUEUE = "pipeline"

def run_pipeline():
    # Connecting ClearML with the current pipeline
    pipe = PipelineController(
        name="AI_Studio_Pipeline_Demo", 
        project="AI_Studio_Demo", 
        version="0.0.1", 
        add_pipeline_tags=False
    )

    # Set default queue for pipeline control
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")

    # Add dataset creation step
    pipe.add_step(
        name="stage_data",
        base_task_project="AI_Studio_Demo",
        base_task_name="Pipeline step 1 dataset artifact",
        execution_queue=EXECUTION_QUEUE
    )

    # Add dataset processing step
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="AI_Studio_Demo",
        base_task_name="Pipeline step 2 process dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.25,
            "General/random_state": 42
        }
    )

    # Add HPO step
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_process", "stage_data"],
        base_task_project="AI_Studio_Demo",
        base_task_name="HPO: Train Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_trials": 4,
            "General/time_limit_minutes": 20,
            "General/run_as_service": False,
            "General/dataset_task_id": "${stage_data.id}",
            "General/base_train_task_id": "8b3f72f435704677abe4e27323d3eba3",
            "General/num_epochs": 20,
            "General/batch_size": 16,
            "General/learning_rate": 1e-3,
            "General/weight_decay": 1e-5
        }
    )

    # Add final model training step
    pipe.add_step(
        name="stage_final_model",
        parents=["stage_hpo", "stage_process"],
        base_task_project="AI_Studio_Demo",
        base_task_name="Final Model Training",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/hpo_task_id": "${stage_hpo.id}",
            "General/test_queue": EXECUTION_QUEUE
        }
    )

    # Start the pipeline locally but tasks will run on queue
    logger.info("Starting pipeline locally with tasks on queue: %s", EXECUTION_QUEUE)
    pipe.start_locally()
    logger.info("Pipeline started successfully")


if __name__ == "__main__":
    run_pipeline()
