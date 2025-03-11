
from invoke import task


@task
def augment(c, back = False):
    if back:
        c.run("nohup python3.11 data_preparation/data_augmentation.py > output.log 2>&1 &")
    else:
        c.run("python3.11 data_preparation/data_augmentation.py")


@task
def clean(c, back=False):
    if not back:
        c.run("python3.11 data_preparation/data_cleaning.py")
    else:
        c.run("nohup python3.11 data_preparation/data_cleaning.py > output.log 2>&1 &")


@task
def requirements(c):
    c.run("docker build -t requirements -f docker/ReqDockerfile .")


@task
def similarity(c, build=False):
    if build:
        c.run("docker build -f docker/SimDockerfile -t similarity-app .")

    first = input("Enter the first sentence: ")
    second = input("Enter the second sentence: ")

    c.run(
        f"docker run -v hug_model:/app/huggingface similarity-app python /app/similarity.py '{first}' '{second}'"
    )


@task
def model(c, test=False, back=False):
    if test:
        c.run("python3.11 model/test.py")
    elif back:
        c.run("nohup python3.11 model/train.py > output.log 2>&1 &")

    else:
        c.run("export CUDA_LAUNCH_BLOCKING=1")
        c.run("python3.11 model/train.py")


# @task
#  def run(c):
