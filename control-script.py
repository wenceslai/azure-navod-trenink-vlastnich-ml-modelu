from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# získání workspace z config.json
ws = Workspace.from_config()

# název experimentu
experiment = Experiment(workspace=ws, name='cifar10-conv-test1')

# název compute targetu
compute_target_name = 'gpu-cluster1'

try:
    aml_compute = ComputeTarget(workspace=ws, name=compute_target_name)
    print('Nalezen již existující compute target: ', aml_compute.name)

except ComputeTargetException: # compute target s tímto jménem neexistuje
    print('Vytvářím nový compute target')

    # konfigurace compute target
    aml_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6", # CREATE INFOR ABOUT DIFF TYPES
        vm_priority="dedicated", # TRY WITH LOW PRIORITY, previosly this did not work for some reason
        min_nodes = 0, 
        max_nodes = 4,
        idle_seconds_before_scaledown=300 
    )

    aml_compute = ComputeTarget.create(
        ws, 
        name=compute_target_name, 
        provisioning_configuration=aml_config
    )

    # spuštěný kód čeká na dokončení vytvoření compute target
    aml_compute.wait_for_completion(show_output=True)


curated_env_name = 'AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu' # název currated prostředí
tf_env = Environment.get(workspace=ws, name=curated_env_name)

datastore = ws.get_default_datastore() # získání defaultního datastoru
data_ref = datastore.path('datasets').as_mount() # získání reference na úložiště

# argumenty pro training-script.py
args = [
    '--batch-size' , 64,
    '--learning-rate', 0.01,
    '--epochs', 1,
    '--data-path', str(data_ref),
    '--run-num', 3
]

# konfigurace trénování
config = ScriptRunConfig(
    source_directory='./src',
    script='training-script.py',
    compute_target=aml_compute,
    environment=tf_env,
    arguments=args
)

# přidání reference do konfigurace
config.run_config.data_references = {data_ref.data_reference_name : data_ref.to_config()}

run = experiment.submit(config) # začátek trénování na cloudu
aml_url = run.get_portal_url() # url pro sledování pokroku

print("Submitted to compute cluster. Click link below\n")
print(aml_url)