__all__ = ['_FrictionModel', '_AdhesionModel', '_WearModel', 'friction_model', 'adhesion_model',
           'wear_model']


class _FrictionModel:
    pass


class _AdhesionModel:
    pass


class _WearModel:
    pass


def wear_model(name, parameters):
    return _WearModel()


def adhesion_model(name, parametrs):
    return _AdhesionModel


def friction_model(name, parameters):
    return _FrictionModel
