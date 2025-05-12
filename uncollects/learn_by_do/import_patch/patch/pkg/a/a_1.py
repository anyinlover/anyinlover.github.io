from functools import wraps


def a_1_func1_wrapper(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        a_2 = f.__globals__.get("a_2")
        a_1_cls1 = f.__globals__.get("a_1_cls1")
        a_1_cls1().a_1_cls1_func1()
        a_2.a_2_func1()
        print("Calling patch_a_1_func1")

    return wrapper
