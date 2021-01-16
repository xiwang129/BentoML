import pytest
import time
import urllib

import bentoml
from tests.bento_service_examples.fastai_classifier import FastaiClassifier

import fastai
import fastai.basics
from fastai.vision.all import *

path = untar_data(URLs.MNIST_TINY)
files = get_image_files(path/'train')
test_data = files[0]

def FastaiModel(): 

    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)
    learner = cnn_learner(dls, resnet34,metrics=error_rate )
    learner.fine_tune(1)

    return learner

@pytest.fixture(scope="module")
def fastai_svc():
    svc = FastaiClassifier()
    learner = FastaiModel()
    svc.pack('learner',learner)
    return svc 


@pytest.fixture(scope="module")
def fastai_svc_saved_dir(tmp_path_factory,fastai_svc):
    tmpdir = str(tmp_path_factory.mktemp('fastai_svc'))
    fastai_svc.save_to_dir(tmpdir)
    return tmpdir

@pytest.fixture()
def fastai_svc_loaded(fastai_svc_saved_dir):
    return bentoml.load(fastai_svc_saved_dir)
    
def test_fastai_artifact(fastai_svc_loaded):
    result = fastai_svc_loaded.predict(test_data)
    assert result == 7

@pytest.fixture()
def fastai_image(fastai_svc_saved_dir):
    import docker

    client = docker.from_env()

    image = client.images.build(
        path = fastai_svc_saved_dir,
        tag = 'fastai_example_service',
        rm=True
    )[0]
    yield image
    client.images.remove(image.id)

def _wait_until_ready(_host, timeout, check_interval=0.5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if(
                urllib.request.urlopen(f'http://{_host}/healthz', timeout=0.1).status == 200
            ):
                break
        except Exception:
            time.sleep(check_interval -0.1)
        else:
            raise AssertionError(f"server didn't get ready in {timeout} seconds")

@pytest.fixture()
def fastai_docker_host(fastai_image):
    import docker

    client = docker.from_env()
    with bentoml.utils.reserve_free_port() as port:
        pass

    command = 'bentoml serve-gunicorn /bento--workers 1'

    try:
        container = client.containers.run(
            command=command,
            image = fastai_image.id,
            auto_remove=True,
            tty=True,
            ports={'5000/tcp':port},
            detach =True,
        )
        _host = f'127.0.0.1:{port}'
        _wait_until_ready(_host,10)
        yield _host
    finally:
        container.stop()
        time.sleep(1)

def test_api_server_with_docker(fastai_docker_host):
    import requests

    response = requests.post(
        f"http://{fastai_docker_host}/predict",
        headers={"Content-Type":"application/json"},
        data = test_data.to_json(),
    )

    preds = response.json
    assert response.status_code == 200
    assert preds[0] == 7







