import numpy as np
import fastai # pylint: disable=import-error
from fastai.vision.all import *

import bentoml
from bentoml.adapters import FileInput
from bentoml.frameworks.fastai import FastaiModelArtifact

@bentoml.env(infer_pip_packages = True)
@bentoml.artifacts([FastaiModelArtifact('learner')])
class FastaiClassifier(bentoml.BentoService):
    @bentoml.api(input=FileInput(),batch=True)
    def predict(self,files):
        result = self.artifacts.learner.predict(files)[0]      
        return result


