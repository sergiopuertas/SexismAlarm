from invoke import task


@task
def activate(c):
    c.run("source venv/bin/activate")


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
        f"docker run -v hug_model:/models similarity-app python /app/similarity.py '{first}' '{second}'"
    )


@task
def model(c, test=False):
    if test:
        c.run("python model/test.py")
    else:
        c.run("python model/train.py")


# @task
#  def run(c):
