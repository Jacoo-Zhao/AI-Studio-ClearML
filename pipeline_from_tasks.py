from clearml import Task
from clearml.automation import PipelineController
# import os
# os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY")
# os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY")
# os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST")

def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


def run_pipeline():
    # Connecting ClearML with the current pipeline,
    # from here on everything is logged automatically
    pipe = PipelineController(
        name="AI_Studio_Basic_Pipeline", project="AI_Studio_Basic_Demo", version="0.0.1", add_pipeline_tags=False
    )

    # pipe.add_parameter(
    #     "url",
    #     "dataset_url",
    # )

    # pipe.add_parameter(
    #     name="url",
    #     default="https://files.clear.ml/AI_Studio_Demo/Pipeline%20step%202%20process%20dataset.d69cec0ccc5a4c6b8900e61489b01847/artifacts/X_test/X_test.npz",
    #     description="URL of the dataset"
    # )

    pipe.set_default_execution_queue("basic_demo")

    pipe.add_step(
        name="stage_data",
        base_task_project="AI_Studio_Basic_Demo",
        base_task_name="Pipeline step 1 dataset artifact",
        # parameter_override={"General/dataset_url": "${pipeline.url}"},
    )

    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="AI_Studio_Basic_Demo",
        base_task_name="Pipeline step 2 process dataset",
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.25,
            "General/random_state": 42
        },
    )

    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="AI_Studio_Basic_Demo",
        base_task_name="Pipeline step 3 train model",
        parameter_override={"General/dataset_task_id": "${stage_process.id}"},
    )

    # for debugging purposes use local jobs
    pipe.start_locally()

    # Starting the pipeline (in the background)
    # pipe.start(queue="basic_demo")
    # pipe.start(queue="pipeline_controller")
    # print("done")
