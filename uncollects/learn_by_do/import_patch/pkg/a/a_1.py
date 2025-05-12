from . import a_2


def a_1_func1():
    a_2.a_2_func1()
    print("Calling a_1_func1")


class a_1_cls1:
    def a_1_cls1_func1(self):
        print("Calling a_1_cls1_func1")
