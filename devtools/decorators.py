import functools

#decorator for bucket object to call a function for all sub-buckets
def recursive_function(func):
    
    func_name = func.__name__
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        returnList = [func(self, *args, **kwargs)]

        for b in self.sub_buckets:
            returnList += getattr(b, func_name)(*args, **kwargs)

        return returnList

    return wrapper