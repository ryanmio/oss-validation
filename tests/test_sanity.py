from oss_validation import config

 
def test_config_constants():
    assert isinstance(config.RANDOM_SEED, int)
    assert config.CRS_DISTANCE.startswith("EPSG")
    assert config.km_to_m(1) == 1000.0 