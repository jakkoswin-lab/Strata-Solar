"""Tests for setup.py's Solar-Open-100B install (an own-path family, CUDA only): the pinned file, the menu entry,
steps 2-7 of install_own_path (one size, 128K context, the engine's --families check, the download, the tokenizer,
the config) and the two refusals that keep it out of an AMD install.  Mocked engine, downloads and processes.

    python -m unittest tools.test_setup_solar
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")
REPO = "mradermacher/Solar-Open-100B-i1-GGUF"
FILE = "Solar-Open-100B.i1-IQ3_XXS.gguf"


def args(**kw):
    base = dict(family="solar", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None,
                context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto",
                vram_reserve_mib=None, host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class SolarEntry(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.OWN_PATH_FAMILIES["solar"]
        self.assertEqual(setup.HF_REVISIONS[REPO], "b324d3dc6a2a1a82d77d52e20272b9c66cbbe569")
        self.assertRegex(fam["hf"], SHA)
        self.assertIn(REPO, fam["hf"])
        self.assertEqual(fam["arch"], "glm4moe")
        self.assertEqual(list(fam["models"]), ["IQ3_XXS"])
        size, sha = fam["sha256"][fam["file"].format(q="IQ3_XXS")]
        self.assertEqual(size, 39679136160)                       # the Hub's LFS pointer of this file
        self.assertRegex(sha, r"^[0-9a-f]{64}$")

    def test_it_is_an_own_path_family_not_a_pack(self):
        self.assertNotIn("solar", setup.FAMILIES)
        for d in setup.MODELS.values():
            self.assertNotIn("solar", d.get("families", ()))
        self.assertNotIn("mmproj", setup.OWN_PATH_FAMILIES["solar"])   # text only: no picture file to fetch

    def test_one_size_and_the_context_follows_the_card(self):
        fam = setup.OWN_PATH_FAMILIES["solar"]
        self.assertEqual(setup.own_path_size(fam, 16.0), "IQ3_XXS")    # the only size, whatever the RAM
        self.assertEqual(setup.own_path_size(fam, 64.0), "IQ3_XXS")
        # 192 KiB of KV per token: 128K would be 24 GiB of VRAM, so the recommendation keeps room for the expert cache
        self.assertEqual(setup.own_path_context(fam, 16.0, "IQ3_XXS"), 8192)
        self.assertEqual(setup.own_path_context(fam, 24.0, "IQ3_XXS"), 65536)
        self.assertEqual(setup.own_path_context(fam, 32.0, "IQ3_XXS"), 65536)
        self.assertEqual(setup.own_path_context(fam, 48.0, "IQ3_XXS"), 131072)
        self.assertEqual(setup.own_path_context(fam, 0.0, "IQ3_XXS"), 131072)   # no card known: as before
        self.assertEqual(setup.own_path_contexts(fam)[-1], 131072)             # the menu stops at the trained length
        self.assertNotIn(262144, setup.own_path_contexts(fam))

    def test_the_effort_default_is_the_template_s_minimal(self):
        self.assertEqual(setup.OWN_PATH_FAMILIES["solar"]["config"], {"reasoning_effort": "minimal"})

    def test_no_parallel_offer(self):
        self.assertFalse(setup.OWN_PATH_FAMILIES["solar"]["parallel"])


class InstallSolar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.families = {"qwen4exp", "qwen35moe", "glm4moe"}
        self.cfg = {}
        eng = self.root / "engine"

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision))
            return self.root / "llama", eng, {"version": "0.1.59.1"}, ["/libs"], gpu, vision

        def write_cfg(path, cfg, source=None, explicit=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False: self.calls.append(("sha", s.name))),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "engine_families", lambda e: self.families),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "recommend_pool_workers", lambda a: a),
                   mock.patch.object(setup, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def install(self, ram=64.0, vram=16.0, hip=False, **kw):
        gpu = {"index": 0, "count": 1, "vram_gb": vram, "name": "Test GPU"}
        return run(setup.install_own_path, args(models_dir=str(self.root / "models"), **kw), "solar", gpu, hip, [],
                   13, [], 8848, ram, None)

    def test_installs_the_pinned_file_and_writes_the_config(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual(len(get), 1)
        gguf = self.root / "models" / "solar-IQ3_XXS" / FILE
        self.assertEqual(get[0][1], setup.hf(REPO) + gguf.name)
        self.assertEqual(get[0][2], gguf)
        self.assertIn(("sha", gguf.name), self.calls)
        tok = [c for c in self.calls if c[0] == "run"]
        self.assertTrue(tok and "strata_tokenizer.py" in str(tok[0][1]))
        self.assertIn(("engine", "none"), self.calls)              # no picture file: images never offered
        self.assertNotIn("vision", self.cfg)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--native") + 1], str(gguf))
        self.assertEqual(a[a.index("--max-context") + 1], "8192")   # 128K would be 24 GiB of KV on this 16 GB card
        self.assertEqual(a[a.index("--mtp") + 1], "0")
        self.assertNotIn("--pack", a)
        self.assertNotIn("--vision", a)
        self.assertEqual(self.cfg["arch"], "glm4moe")
        self.assertEqual(self.cfg["model_name"], "solar-open-100b-iq3_xxs")
        self.assertEqual(self.cfg["reasoning_effort"], "minimal")
        self.assertEqual(Path(self.cfg["exe"]).parent, self.root / "engine")
        self.assertIn(("config", "strata-solar-iq3_xxs.json"), self.calls)

    def test_one_request_at_a_time_is_written_without_asking(self):
        rc, out = self.install()                                  # yes=True: nothing is asked anyway
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg.get("parallel"), 1)
        self.assertNotIn("--batch", self.cfg["args"])

    def test_an_engine_without_the_family_is_refused_before_the_download(self):
        self.families = {"qwen4exp", "qwen35moe"}                  # an engine from before the port
        rc, out = self.install()
        self.assertNotEqual(rc, 0)
        self.assertIn("does not run Solar-Open-100B", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_an_amd_install_is_refused_by_name(self):
        rc, out = self.install(hip=True)
        self.assertNotEqual(rc, 0)
        self.assertIn("NVIDIA cards only", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])


if __name__ == "__main__":
    unittest.main()
