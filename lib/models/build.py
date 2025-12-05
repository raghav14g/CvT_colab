from .registry import model_entrypoints
from .registry import is_model
#CNN+ Import your new CNN model

# Import all models here to register them with the MODEL_REGISTRY
from .vanilla_cnn import VanillaCNN 
from .cls_cvt import CvT
# The build_model function is imported from the builder
from .builder import build_model
#CNN-
def build_model(config, **kwargs):
    model_name = config.MODEL.NAME
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, **kwargs)
