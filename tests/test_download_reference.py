from importlib import import_module

 
def test_module_imports():
    mod = import_module("oss_validation.download_reference")
    assert hasattr(mod, "main") 