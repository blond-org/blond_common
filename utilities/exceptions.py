###########################
#####BASIC DATA ERRORS#####
###########################
class InputError(Exception):
    pass

class DataDefinitionError(Exception):
    pass

class MismatchedListLengths(Exception):
    pass

class NotIterable(Exception):
    pass


###############################
#####POTENTIAL WELL ERRORS#####
###############################
class NoInnerPotentialWell(Exception):
    pass

class SingleWellOnly(Exception):
    pass