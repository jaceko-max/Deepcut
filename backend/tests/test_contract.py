import json
from pathlib import Path

import pytest

from app.models import JobResponse

CONTRACT = Path(__file__).resolve().parents[2] / "contract"


@pytest.mark.parametrize("name", ["job_done.example.json", "job_running.example.json"])
def test_examples_validate_and_roundtrip(name):
    raw = json.loads((CONTRACT / name).read_text())
    job = JobResponse.model_validate(raw)
    assert job.model_dump(mode="json") == raw  # same field names, types and nullability


def test_why_length_and_proxy_url_in_example():
    job = JobResponse.model_validate_json((CONTRACT / "job_done.example.json").read_text())
    for r in job.results:
        assert len(r.why) <= 90
        assert r.preview_proxy_url == f"/api/preview/{r.deezer_id}"
