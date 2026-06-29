"""Batch API endpoint tests."""

from yunshu_gateway.routers.batch_inference import BatchItem, BatchRequest


class TestBatchRequest:
    def test_defaults(self):
        item = BatchItem(custom_id="req_1", body={"model": "test", "messages": []})
        assert item.method == "POST"
        assert item.url == "/v1/chat/completions"

    def test_custom_url(self):
        item = BatchItem(
            custom_id="req_2",
            url="/v1/completions",
            body={"model": "test", "prompt": "hello"},
        )
        assert item.url == "/v1/completions"

    def test_batch_request(self):
        req = BatchRequest(
            requests=[
                BatchItem(custom_id=f"r{i}", body={"model": "test"}) for i in range(3)
            ]
        )
        assert len(req.requests) == 3
        assert req.max_concurrent == 4

    def test_batch_request_custom_concurrency(self):
        req = BatchRequest(
            requests=[BatchItem(custom_id="r1", body={})],
            max_concurrent=8,
        )
        assert req.max_concurrent == 8
