###########################
#####BASIC DATA ERRORS#####
###########################
class InputError(Exception):
    pass

class DataDefinitionError(Exception):
    pass


###############################
#####POTENTIAL WELL ERRORS#####
############################### 
class No_Inner_Potential_Well(Exception):
    pass