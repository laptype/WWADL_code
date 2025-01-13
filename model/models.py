

backbones = {}
def register_backbone(name):
    def decorator(cls):
        backbones[name] = cls
        return cls
    return decorator

# builder functions
def make_backbone(name, **kwargs):
    backbone = backbones[name](**kwargs)
    return backbone


# meta arch (the actual implementation of each model)
models = {}
models_config = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def register_model_config(name):
    def decorator(cls):
        models_config[name] = cls
        return cls
    return decorator

def make_model(name, **kwargs):
    model_config = models_config[name](**kwargs)
    meta_arch = models[name](model_config)
    return meta_arch