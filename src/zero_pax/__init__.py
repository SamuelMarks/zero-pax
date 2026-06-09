"""zero_pax API."""


class BaseParameter:
    def __init__(self, value):
        self.value = value


class Layer:
    def __init__(self):
        self.params = {}

    def register_parameter(self, name, value):
        self.params[name] = BaseParameter(value)

    def __call__(self, inputs):
        return inputs
