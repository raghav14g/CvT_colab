# /content/CvT/lib/models/builder.py

# --- Registry Definition ---
class Registry(object):
    """Registry class to store model modules."""
    def __init__(self):
        self._modules = {}

    def register_module(self, module):
        """Registers a module (class) with its name as the key."""
        name = module.__name__
        if name in self._modules:
            # Handle possible re-registration by design
            print(f"Warning: Module {name} already registered. Overwriting.")
        self._modules[name] = module
        return module

    def __call__(self, name):
        """Allows calling the registry instance to retrieve a module."""
        if name not in self._modules:
            raise KeyError('Name not found: {}'.format(name))
        return self._modules[name]
        
# Instantiate the global MODEL_REGISTRY object
MODEL_REGISTRY = Registry()

# --- Build Function ---
def build_model(config):
    """Initializes a model instance based on configuration."""
    model_name = config.MODEL.NAME
    
    if model_name not in MODEL_REGISTRY._modules:
        # Note: If this fails, you likely forgot to import the model file in lib/models/build.py
        raise ValueError(f"Model {model_name} not registered in MODEL_REGISTRY. "
                         f"Available models: {list(MODEL_REGISTRY._modules.keys())}")
    
    model_class = MODEL_REGISTRY._modules[model_name]
    
    # Initialize the model using the entire configuration object
    return model_class(config)
