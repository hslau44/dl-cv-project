from src.modules.utils import example_func as func_m


def test_modules_utils_example_func():
    val = func_m()
    assert  val == True
