from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer, ClearmlJob
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
import logging
import time
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name='AI_Studio_Demo',
    task_name='HPO: Train Model',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'base_train_task_id': '8b3f72f435704677abe4e27323d3eba3',  # ID of the "template" task that performed model training
    'num_trials': 3,  # Reduced from 10 to 3 trials
    'time_limit_minutes': 20,  # Reduced from 60 to 5 minutes
    'run_as_service': False,
    
    # Parameters from the template task of model training
    'processed_dataset_id': '99e286d358754697a37ad75c279a6f0a',  # Will be set from pipeline
    'test_queue': 'pipeline',  # Queue for test tasks
    'num_epochs': 20,  # Reduced from 50 to 20 epochs
    'batch_size': 32,  # Default batch size
    'learning_rate': 1e-3,  # Default learning rate
    'weight_decay': 1e-5  # Default weight decay
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('General/processed_dataset_id')  # Get from General namespace
if not dataset_id:
    # Try getting from args as fallback
    dataset_id = args.get('processed_dataset_id')
    logger.info(f"No dataset_id in General namespace, using from args: {dataset_id}")

if not dataset_id:
    # Use fixed dataset ID as last resort
    dataset_id = "99e286d358754697a37ad75c279a6f0a"
    logger.info(f"Using fixed dataset ID: {dataset_id}")

logger.info(f"Using dataset ID: {dataset_id}")

# Get the actual training model task
try:
    BASE_TRAIN_TASK_ID = args['base_train_task_id']
    logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task ID: {e}")
    raise

# Verify dataset exists
try:
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Successfully verified dataset: {dataset.name}")
except Exception as e:
    logger.error(f"Failed to verify dataset: {e}")
    raise

# Override the template task parameters
base_task = ClearmlJob(
    base_task_id=BASE_TRAIN_TASK_ID,
    parameter_override={
        "processed_dataset_id": args['processed_dataset_id'],
        "test_queue": args['test_queue'],
        "num_epochs": args['num_epochs'],
        "batch_size": args['batch_size'],
        "learning_rate": args['learning_rate'],
        "weight_decay": args['weight_decay']
    },
    disable_clone_task=True, # Use the base_task_id directly (base-task must be in draft-mode / created)
)

# Create the HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=base_task.task_id(),
    hyper_parameters=[
        UniformIntegerParameterRange('num_epochs', min_value=10, max_value=args['num_epochs']),
        UniformIntegerParameterRange('batch_size', min_value=16, max_value=64),  # Reduced range
        UniformParameterRange('learning_rate', min_value=1e-4, max_value=1e-2),  # Reduced range
        UniformParameterRange('weight_decay', min_value=1e-6, max_value=1e-4)  # Reduced range
    ],
    objective_metric_title='validation',
    objective_metric_series='accuracy',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    compute_time_limit=None,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=args['num_epochs'],
    pool_period_min=1.0,  # Reduced from 2.0 to 1.0 to check more frequently
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,  # Reduced from 5 to 2
    parameter_override={
        'processed_dataset_id': dataset_id,
        'General/processed_dataset_id': dataset_id,
        'test_queue': args['test_queue'],
        'General/test_queue': args['test_queue'],
        'num_epochs': args['num_epochs'],
        'General/num_epochs': args['num_epochs'],
        'batch_size': args['batch_size'],
        'General/batch_size': args['batch_size'],
        'learning_rate': args['learning_rate'],
        'General/learning_rate': args['learning_rate'],
        'weight_decay': args['weight_decay'],
        'General/weight_decay': args['weight_decay']
    }
)

# Start the HPO task
logger.info("Starting HPO task...")
hpo_task.start()

# Wait for optimization to complete
logger.info(f"Waiting for optimization to complete (time limit: {args['time_limit_minutes']} minutes)...")
time.sleep(args['time_limit_minutes'] * 60)  # Wait for the full time limit

# Get the top performing experiments
try:
    top_exp = hpo_task.get_top_experiments(top_k=1)  # Get only the best experiment
    if top_exp:
        best_exp = top_exp[0]
        logger.info(f"Best experiment: {best_exp.id}")
        
        # Get the best parameters and accuracy
        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()
        best_accuracy = metrics['validation']['accuracy'] if metrics and 'validation' in metrics and 'accuracy' in metrics['validation'] else None
        
        # Log detailed information about the best experiment
        logger.info("Best experiment parameters:")
        logger.info(f"  - num_epochs: {best_params.get('num_epochs')}")
        logger.info(f"  - batch_size: {best_params.get('batch_size')}")
        logger.info(f"  - learning_rate: {best_params.get('learning_rate')}")
        logger.info(f"  - weight_decay: {best_params.get('weight_decay')}")
        logger.info(f"Best validation accuracy: {best_accuracy}")
        
        # Save best parameters and accuracy
        best_results = {
            'parameters': best_params,
            'accuracy': best_accuracy
        }
        
        # Save to a temporary file
        temp_file = 'best_parameters.json'
        with open(temp_file, 'w') as f:
            json.dump(best_results, f, indent=4)
        
        # Upload as artifact
        task.upload_artifact('best_parameters', temp_file)
        logger.info(f"Saved best parameters with accuracy: {best_accuracy}")
        
        # Also save as task parameters for easier access
        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_accuracy', best_accuracy)
        
        logger.info("Best parameters saved as both artifact and task parameters")
    else:
        logger.warning("No experiments completed yet. This might be normal if the optimization just started.")
except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    raise

# Make sure background optimization stopped
hpo_task.stop()
logger.info("Optimizer stopped")

print('We are done, good bye')