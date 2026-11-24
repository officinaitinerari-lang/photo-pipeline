from src.database import STATUS_COMPLETED, STATUS_FAILED, STATUS_PROCESSING, UPLOAD_UPLOADED, Database


def test_queue_lifecycle_and_idempotency(tmp_path):
    db = Database(tmp_path / "state" / "pipeline.db")
    src = tmp_path / "a.jpg"
    assert db.enqueue(src, "abc")[0] == "new"
    assert db.enqueue(src, "abc")[0] == "already_queued"

    job = db.claim_next()
    assert job is not None and job.status == STATUS_PROCESSING and job.attempts == 1
    assert db.claim_next() is None  # only one job, already claimed

    # crash -> restart recovery
    assert db.reset_stale_processing() == 1
    job = db.claim_next()
    assert job is not None and job.attempts == 2

    db.update_job(job.id, status=STATUS_COMPLETED, ai_json='{"x": 1}')
    assert db.enqueue(src, "abc")[0] == "duplicate"
    assert db.get_job(job.id).ai_data == {"x": 1}


def test_failed_job_can_be_requeued_and_deferred(tmp_path):
    db = Database(tmp_path / "p.db")
    _, job_id = db.enqueue(tmp_path / "b.jpg", "def")
    db.claim_next()
    db.update_job(job_id, status=STATUS_FAILED, error="boom")
    assert db.enqueue(tmp_path / "b.jpg", "def")[0] == "requeued"

    db.claim_next()
    db.defer_job(job_id, 3600, "quota", refund_attempt=True)
    assert db.claim_next() is None  # deferred into the future
    assert db.get_job(job_id).attempts == 0


def test_upload_tracking(tmp_path):
    db = Database(tmp_path / "p.db")
    _, job_id = db.enqueue(tmp_path / "c.jpg", "ghi")
    up = db.ensure_upload(job_id, "adobe", tmp_path / "c.jpg", "c.jpg")
    assert [u.id for u in db.uploads_needing_transfer()] == [up.id]
    db.set_upload_status(up.id, UPLOAD_UPLOADED)
    assert db.uploads_needing_transfer() == []
    assert db.ensure_upload(job_id, "adobe", tmp_path / "c.jpg", "c.jpg").status == UPLOAD_UPLOADED
