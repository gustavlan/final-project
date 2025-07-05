import os
from app import _cache_path


def test_cache_path_sanitizes(tmp_path):
    path = _cache_path('yahoo', '../evil', '2021-01-01', '2021-01-02')
    # Path should be within cache directory
    assert os.path.dirname(path).endswith('cache')
    filename = os.path.basename(path)
    assert '..' not in filename
    assert '/' not in filename and '\\' not in filename

