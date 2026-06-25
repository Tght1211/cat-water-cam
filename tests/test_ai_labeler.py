import base64
import pytest
from catcam.ai_labeler import encode_frames, build_messages, parse_label


def test_encode_frames_data_url(tmp_path):
    p = tmp_path / "f.jpg"; p.write_bytes(b"\xff\xd8jpegbytes")
    urls = encode_frames([p])
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(urls[0].split(",", 1)[1]) == b"\xff\xd8jpegbytes"


def test_build_messages_has_all_images_and_prompt():
    msgs = build_messages(["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"])
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    texts = [c for c in content if c["type"] == "text"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert "喝水" in texts[0]["text"] and "JSON" in texts[0]["text"]


def test_parse_label_plain_json():
    out = parse_label('{"drinking": true, "confidence": 0.9, "reason": "舔水"}')
    assert out == {"drinking": True, "confidence": 0.9, "reason": "舔水"}


def test_parse_label_with_code_fence_and_noise():
    raw = '好的：\n```json\n{"drinking": false, "confidence": 0.6, "reason": "只是凑近"}\n```\n'
    out = parse_label(raw)
    assert out["drinking"] is False and out["confidence"] == 0.6


def test_parse_label_string_bool_and_missing_fields():
    out = parse_label('{"drinking": "yes"}')
    assert out["drinking"] is True and out["confidence"] is None and out["reason"] == ""


def test_parse_label_garbage_raises():
    with pytest.raises(ValueError):
        parse_label("完全不是 json")
