from archive import stratified_sampling as sampling, parsing


def test_sampling_size():
    df = parsing._read_raw_csv().head(500)
    df = sampling.build_strata(df)
    sample = sampling.stratified_sample(df, n=20)
    assert len(sample) == 20
    # Each sample has county label
    assert sample.county.isna().sum() == 0 