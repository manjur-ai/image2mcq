import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from image2mcq.models import MCQQuestion, MCQSet, ContentBlock
from image2mcq.prompts import build_system_prompt, build_user_prompt
from image2mcq.image_ocr import (
    ImageOCRExtractor,
    _download_image,
    _ocr_pytesseract,
    _ocr_vision_api,
    _DEFAULT_OCR_PRIORITY,
    _OCR_MODELS_ENV_VAR,
)


class TestModels:
    def test_question_to_dict_schema(self):
        q = MCQQuestion(
            question_html="<b>What is 2+2?</b>",
            options=["3", "4", "5", "6"],
            answers=[1],
            multi=False,
            marks=1.0,
            negative_marks=0.25,
            difficulty="easy",
            explaination="Basic addition.",
        )
        d = q.to_dict()
        assert d["question_html"] == "<b>What is 2+2?</b>"
        assert d["answers"] == [1]
        assert d["multi"] is False

    def test_mcqset_to_json_schema(self):
        q = MCQQuestion(
            question_html="Q", options=["A", "B", "C", "D"],
            answers=[0], multi=False,
            marks=1.0, negative_marks=0.25,
            difficulty="easy", explaination="",
        )
        s = MCQSet(
            source_url="test", page_title="Test",
            questions=[q, q], total_questions=2,
            content_summary="2 images",
        )
        data = json.loads(s.to_json())
        assert "total_exam_time" in data
        assert len(data["questions"]) == 2

    def test_filter_by_difficulty(self):
        q1 = MCQQuestion("Q1", ["A","B","C","D"],[0],False,1,0.25,"easy","")
        q2 = MCQQuestion("Q2", ["A","B","C","D"],[0],False,1,0.25,"hard","")
        s = MCQSet("u","T",[q1,q2],2,"2 images")
        filtered = s.filter_by_difficulty("easy")
        assert len(filtered.questions) == 1
        assert filtered.questions[0].difficulty == "easy"


class TestPrompts:
    def test_system_prompt_has_schema(self):
        prompt = build_system_prompt()
        assert "question_html" in prompt
        assert "options" in prompt
        assert "answers" in prompt
        assert "multi" in prompt

    def test_user_prompt_includes_images(self):
        blocks = [ContentBlock(type="image", content="https://example.com/img.png", alt_text="chart")]
        prompt = build_user_prompt(blocks, n=5)
        assert "IMAGE" in prompt
        assert "chart" in prompt
        assert "5" in prompt

    def test_user_prompt_with_text_blocks(self):
        blocks = [ContentBlock(type="text", content="Some extracted text content")]
        prompt = build_user_prompt(blocks, n=3)
        assert "EXTRACTED TEXT" in prompt
        assert "Some extracted text" in prompt

    def test_user_prompt_n999(self):
        blocks = [ContentBlock(type="text", content="Content")]
        prompt = build_user_prompt(blocks, n=999)
        assert "as many" in prompt.lower() or "cover all" in prompt.lower()


class TestOCRPipeline:
    def test_download_image_fails_gracefully(self):
        data = _download_image("https://nonexistent.example/img.png")
        assert data == b""

    def test_resolve_ocr_models_default(self):
        models = ImageOCRExtractor._resolve_ocr_models()
        assert models == _DEFAULT_OCR_PRIORITY

    def test_resolve_ocr_models_from_env(self, monkeypatch):
        monkeypatch.setenv(_OCR_MODELS_ENV_VAR, "model-a,model-b")
        models = ImageOCRExtractor._resolve_ocr_models()
        assert models == ["model-a", "model-b"]

    def test_resolve_ocr_models_explicit(self):
        models = ImageOCRExtractor._resolve_ocr_models(["custom-model"])
        assert models == ["custom-model"]

    def test_enrich_blocks_no_images(self):
        ocr = ImageOCRExtractor(backend="pytesseract")
        blocks = [ContentBlock(type="text", content="hello")]
        result = ocr.enrich_blocks(blocks)
        assert len(result) == 1
        assert result[0].type == "text"

    def test_enrich_blocks_empty_image_blocks(self):
        ocr = ImageOCRExtractor(backend="pytesseract")
        blocks = [ContentBlock(type="image", content="")]
        result = ocr.enrich_blocks(blocks)
        assert len(result) == 1
        assert result[0].type == "image"


class TestImageMCQGenerator:
    def test_unknown_method_raises(self):
        from image2mcq.generator import ImageMCQGenerator
        with pytest.raises(ValueError, match="twostep | images2mcq"):
            ImageMCQGenerator(api_key="sk-test", method="invalid")

    def test_missing_api_key_raises(self):
        from image2mcq.generator import ImageMCQGenerator
        with pytest.raises(ValueError, match="No API key"):
            ImageMCQGenerator(api_key="")

    def test_ollama_no_api_key_required(self):
        from image2mcq.generator import ImageMCQGenerator
        gen = ImageMCQGenerator(provider="ollama")
        assert gen.provider == "ollama"

    def test_parse_response_handles_markdown_fences(self):
        from image2mcq.generator import ImageMCQGenerator
        raw = '```json\n[{"question_html":"Q","options":["A","B","C","D"],"answers":[0],"difficulty":"easy","explaination":""}]\n```'
        qs = ImageMCQGenerator._parse_response(None, raw)
        assert len(qs) == 1

    def test_parse_response_handles_single_int_answer(self):
        from image2mcq.generator import ImageMCQGenerator
        raw = '[{"question_html":"Q","options":["A","B","C","D"],"answers":2,"difficulty":"medium","explaination":""}]'
        qs = ImageMCQGenerator._parse_response(None, raw)
        assert len(qs) == 1
        assert qs[0].answers == [2]

    def test_parse_response_skips_malformed(self):
        from image2mcq.generator import ImageMCQGenerator
        raw = '[null, {"question_html":"Q","options":["A","B","C","D"],"answers":[0],"difficulty":"easy","explaination":""}]'
        qs = ImageMCQGenerator._parse_response(None, raw)
        assert len(qs) == 1

    def test_parse_response_invalid_json_raises(self):
        from image2mcq.generator import ImageMCQGenerator
        with pytest.raises(ValueError, match="non-JSON"):
            ImageMCQGenerator._parse_response(None, "not json at all")

    def test_get_mcq_models_default(self):
        from image2mcq.generator import ImageMCQGenerator
        models = ImageMCQGenerator.get_mcq_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_set_and_get_mcq_models(self):
        from image2mcq.generator import ImageMCQGenerator
        ImageMCQGenerator.set_mcq_models("model-a,model-b")
        models = ImageMCQGenerator.get_mcq_models()
        assert models == ["model-a", "model-b"]

    def test_get_ocr_models_default(self):
        from image2mcq.generator import ImageMCQGenerator
        models = ImageMCQGenerator.get_ocr_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_set_and_get_ocr_models(self):
        from image2mcq.generator import ImageMCQGenerator
        ImageMCQGenerator.set_ocr_models("ocr-a,ocr-b")
        models = ImageMCQGenerator.get_ocr_models()
        assert models == ["ocr-a", "ocr-b"]

    def test_set_api_key_sets_when_empty(self):
        from image2mcq.generator import ImageMCQGenerator
        os.environ.pop("OPENROUTER_API_KEY", None)
        ImageMCQGenerator.set_api_key("openrouter", "sk-test-key")
        assert os.environ.get("OPENROUTER_API_KEY") == "sk-test-key"
        del os.environ["OPENROUTER_API_KEY"]

    def test_set_api_key_ignores_when_already_set(self, monkeypatch):
        from image2mcq.generator import ImageMCQGenerator
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-existing")
        ImageMCQGenerator.set_api_key("openrouter", "sk-new-key")
        assert os.environ["OPENROUTER_API_KEY"] == "sk-existing"

    def test_set_api_key_ollama_noop(self):
        from image2mcq.generator import ImageMCQGenerator
        os.environ.pop("OPENROUTER_API_KEY", None)
        ImageMCQGenerator.set_api_key("ollama", "should-not-set")
        assert os.environ.get("OPENROUTER_API_KEY") is None

    def test__build_summary(self):
        from image2mcq.generator import ImageMCQGenerator
        blocks = [
            ContentBlock(type="image", content="url1"),
            ContentBlock(type="text", content="text"),
        ]
        summary = ImageMCQGenerator._build_summary(blocks)
        assert "1 image" in summary
        assert "1 text" in summary

    @patch("image2mcq.generator._download_image", return_value=b"")
    def test_from_image_paths_twostep(self, mock_dl):
        from image2mcq.generator import ImageMCQGenerator
        tmp = Path(__file__).parent / "test_img.png"
        tmp.write_bytes(b"fake png data")
        try:
            gen = ImageMCQGenerator(api_key="sk-test", method="twostep")
            with pytest.raises(ValueError, match="No image data"):
                gen.from_image_paths(str(tmp), n=1)
        finally:
            tmp.unlink(missing_ok=True)

    @patch("image2mcq.generator.ImageMCQGenerator._vision_mcq")
    def test_from_image_paths_images2mcq(self, mock_vision):
        from image2mcq.generator import ImageMCQGenerator
        tmp = Path(__file__).parent / "test_img2.png"
        tmp.write_bytes(b"fake png data")
        try:
            mock_vision.return_value = []
            gen = ImageMCQGenerator(api_key="sk-test", method="images2mcq")
            mcq = gen.from_image_paths(str(tmp), n=1)
            assert mcq.total_questions == 0
        finally:
            tmp.unlink(missing_ok=True)


class TestCLI:
    def test_cli_version(self, monkeypatch):
        import sys
        from image2mcq import cli
        monkeypatch.setattr(sys, "argv", ["image2mcq", "--version"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0

    def test_cli_no_input_shows_error(self, monkeypatch):
        import sys
        from image2mcq import cli
        monkeypatch.setattr(sys, "argv", ["image2mcq"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_cli_image_folder_not_found(self, monkeypatch):
        import sys
        from image2mcq import cli
        monkeypatch.setattr(sys, "argv", ["image2mcq", "--image-folder", "/nonexistent",
                                          "--api-key", "sk-test"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_cli_image_folder(self, tmp_path, monkeypatch):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "slide1.png").write_text("dummy")
        (img_dir / "slide2.png").write_text("dummy")
        monkeypatch.setattr(sys, "argv", ["image2mcq", "--image-folder", str(img_dir),
                                          "-n", "1", "--api-key", "sk-test"])
        from image2mcq.generator import ImageMCQGenerator
        from image2mcq.models import MCQQuestion
        orig = ImageMCQGenerator.from_image_paths
        called = []
        def mock_method(self, paths, **kw):
            called.extend(paths)
            q = MCQQuestion("Q", ["A","B","C","D"],[0],False,1,0.25,"easy","")
            return MCQSet("test", "Images", [q], 1, "")
        try:
            ImageMCQGenerator.from_image_paths = mock_method
            from image2mcq import cli
            cli.main()
        finally:
            ImageMCQGenerator.from_image_paths = orig
        assert len(called) == 2
        assert "slide1.png" in called[0] or "slide2.png" in called[0]

    def test_cli_env_var_api_key(self, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv", ["image2mcq", "--image-url", "https://example.com/img.png", "-n", "1"])
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-test")
        from image2mcq.generator import ImageMCQGenerator
        from image2mcq.models import MCQQuestion
        orig = ImageMCQGenerator.from_image_urls
        called = []
        def mock_method(self, urls, **kw):
            called.extend(urls)
            q = MCQQuestion("Q", ["A","B","C","D"],[0],False,1,0.25,"easy","")
            return MCQSet("test", "Images", [q], 1, "")
        try:
            ImageMCQGenerator.from_image_urls = mock_method
            from image2mcq import cli
            cli.main()
        finally:
            ImageMCQGenerator.from_image_urls = orig
        assert len(called) == 1
        assert "example.com/img.png" in called[0]

    def test_cli_default_n_is_999(self, monkeypatch):
        import sys
        from image2mcq.generator import ImageMCQGenerator
        orig = ImageMCQGenerator.__init__
        captured_n = []
        def mock_init(self, **kw):
            captured_n.append(kw.get("batch_size", 10))
        try:
            ImageMCQGenerator.__init__ = mock_init
            monkeypatch.setattr(sys, "argv", ["image2mcq", "--image-url", "https://ex.com/i.png", "--api-key", "sk"])
            from image2mcq import cli
            with pytest.raises(SystemExit) as exc:
                cli.main()
        finally:
            ImageMCQGenerator.__init__ = orig
