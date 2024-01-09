from torch.autograd.function import _ContextMethodMixin
from torch.cuda import current_stream


def stream_barrier(stream):
    if stream is not None:
        current_stream().wait_stream(stream)


class Bunch(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            try:
                return object.__getattribute__(self, key)
            except AttributeError:
                raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class DummyContext(Bunch, _ContextMethodMixin):

    pass
