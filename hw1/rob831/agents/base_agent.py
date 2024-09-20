
class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        print(f" ************ train ************ " )
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        print(f" ************ add_to_replay_buffer ************ " )
        raise NotImplementedError

    def sample(self, batch_size):
        print(f" ************ sample ************ " )
        raise NotImplementedError

    def save(self, path):
        print(f" ************ save ************ " )
        raise NotImplementedError
