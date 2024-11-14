from invoke import task


@task
def augment(c):
    c.run("python data_preparation/data_augmentation.py")


@task
def clean(c):
    c.run("python data_preparation/data_cleaning.py")


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
def model(c, test=False, back = False):
    if test:
        c.run("python model/test.py")
    elif back:
        c.run("nohup python model/train.py > output.log 2>&1 &")

    else:
        c.run("export CUDA_LAUNCH_BLOCKING=1")
        c.run("python model/train.py")



# @task
#  def run(c):
