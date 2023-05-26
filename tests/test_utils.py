from src.utils import example_func as func_s


def test_utils_example_func():
    val = func_s()
    assert  val == True
