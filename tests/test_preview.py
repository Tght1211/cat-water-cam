import json

from catcam.app import main


def test_preview_does_not_start_detection_audio_mail_or_external_ai(tmp_path, monkeypatch):
    import catcam.app
    import catcam.preview

    def forbidden(*arguments, **keywords):
        raise AssertionError("仅预览不应启动识别、外部 AI、邮件或录音")

    monkeypatch.setattr(catcam.app.CatDetector, "from_path", forbidden)
    monkeypatch.setattr(catcam.app.AILabeler, "from_config", forbidden)
    monkeypatch.setattr(catcam.app, "Emailer", forbidden)
    monkeypatch.setattr(catcam.app, "AudioRing", forbidden)
    monkeypatch.setattr(catcam.preview.threading.Thread, "start", lambda thread: None)
    class Capture:
        released = False
        def release(self):
            self.released = True
    capture = Capture()
    monkeypatch.setattr(catcam.preview, "open_capture", lambda config: capture)
    servers = []
    monkeypatch.setattr(catcam.preview.uvicorn, "run", lambda app, **settings: servers.append(settings))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data_dir": str(tmp_path / "data"), "mail_enabled": True,
                                "ai_label_enabled": True, "ai_api_key": "synthetic", "record_audio": True,
                                "web_host": "0.0.0.0"}))
    main(str(path), preview_only=True)
    assert servers[0]["host"] == "127.0.0.1"
    assert capture.released
