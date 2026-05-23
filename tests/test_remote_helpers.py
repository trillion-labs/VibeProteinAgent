from __future__ import annotations

from vibeproteinagent.adapters._remote.fasta import parse_fasta, write_fasta
from vibeproteinagent.adapters._remote.http import HttpClient, HttpError, HttpResponse
from vibeproteinagent.adapters._remote.temporal import before_cutoff
from vibeproteinagent.adapters._remote.uniprot import first_public_date, search_accessions


def test_temporal_filter_is_strict():
    assert before_cutoff("2021-12-31", "2022-01-01")
    assert not before_cutoff("2022-01-01", "2022-01-01")
    assert not before_cutoff(None, "2022-01-01")
    assert before_cutoff(None, "2022-01-01", include_undated=True)


def test_uniprot_first_public_date_shapes():
    assert (
        first_public_date({"results": [{"entryAudit": {"firstPublicDate": "2020-01-01"}}]})
        == "2020-01-01"
    )
    assert first_public_date({"entryAudit": {"firstPublicDate": "2019-01-01"}}) == "2019-01-01"


def test_uniprot_search_accessions_caps_page_size():
    seen = {}

    class Client:
        def get_json(self, url, params):
            seen["url"] = url
            seen["params"] = params
            return {"results": [{"primaryAccession": "P12345"}]}

    assert search_accessions(Client(), "go:0052855", size=2000) == ["P12345"]
    assert seen["params"]["size"] == 500


def test_fasta_round_trip(tmp_path):
    records = parse_fasta(">a desc\nMKK\n>b\nLL\n")
    assert records[0].identifier == "a"
    out = tmp_path / "x.fasta"
    write_fasta(records, out)
    assert ">a desc" in out.read_text()


def test_http_client_cache_key_stable():
    client = HttpClient()
    key1 = client.cache_key("GET", "https://example.test", extra={"temporal_cutoff": "2022-01-01"})
    key2 = client.cache_key("GET", "https://example.test", extra={"temporal_cutoff": "2022-01-01"})
    assert key1 == key2


def test_http_client_mock_transport_retries_and_caches(tmp_path):
    calls = {"count": 0}

    def transport(url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise HttpError("temporary", status=503, url=url)
        return HttpResponse(url=url, status=200, headers=headers, text='{"ok": true}')

    client = HttpClient(cache_dir=tmp_path, transport=transport, retries=1, backoff_s=0)
    assert client.get_json("https://example.test/data", params={"q": "x"}) == {"ok": True}
    assert calls["count"] == 2

    cached = client.get("https://example.test/data", params={"q": "x"})
    assert cached.from_cache
    assert calls["count"] == 2
