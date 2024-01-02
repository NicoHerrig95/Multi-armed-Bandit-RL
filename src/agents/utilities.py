class Action(object):
    """
    Class object for an action the agent can take.
    
    Arguments:
    ----------
    initial_value: Initial values for q-estimates, either None or a numeric value. Type -> int, float, complex, NoneType 
    """

    def __init__(self, initial_value = None):


        if initial_value != None and not isinstance(initial_value, (int, float, complex)):
                raise ValueError("initial_value must be of types: [list, int, float, complex, NoneType]")

        # assigning q-estimates
        if initial_value == None:
            self.q = 0

        if initial_value != None:
            self.q = initial_value

        # counter for time action has been selected
        self.n = 0