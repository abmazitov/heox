from .state import State


class Protocol:
    def __init__(self, invoke_every: int, steps_per_invoke: int):
        """
        Initialize the Protocol class.

        :param invoke_every: Invoke this protocol every n steps.
        :param steps_per_invoke: Number of steps to run for each method.
        """
        self.invoke_every = invoke_every
        self.steps_per_invoke = steps_per_invoke
        self.num_invokes = 0

    def initialize(self, state: State):
        """
        Initialize the protocol with the given state.
        """
        pass

    def step(self, state: State):
        """
        Perform a single step of the protocol.
        """
        pass

    def evolve(self, state: State):
        """
        Run the protocol for a given number of steps.

        :param steps: Number of steps to run for each method.
        """
        if state.properties["step"] % self.invoke_every == 0:
            for _ in range(self.steps_per_invoke):
                self.step(state)
        state.properties["step"] += 1
