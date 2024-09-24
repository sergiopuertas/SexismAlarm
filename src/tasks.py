from invoke import task


@task
def requirements(c):
    c.run("docker build -t requirements -f dockers/ReqDockerfile .")


@task
def buildmodel(c):
    c.run("docker build -t model-train -f dockers/ModelDockerfile .")


@task
def runmodel(c):
    c.run("docker run --rm model-train")


@task
def model(c):
    c.run("docker build -t model-train -f dockers/ModelDockerfile .")
    c.run("docker run --rm model-train")


# @task
# def run(c):
