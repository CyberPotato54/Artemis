"""Tests for tools/py/speed_llama.py: device choice by name, the server arguments and the affinity helpers. No
llama-server, no GPU, no network.

Run from the repository root: python -I -m unittest discover -s tools/py/tests -v
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "speed_llama.py"
spec = importlib.util.spec_from_file_location("speed_llama", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules["speed_llama"] = m
spec.loader.exec_module(m)

LISTING = HERE / "data" / "list-devices-b11509.txt"
XT = "AMD Radeon RX 9070 XT"


class Devices(unittest.TestCase):
    def test_parse_devices_reads_the_captured_listing(self):
        devices = m.parse_devices(LISTING.read_text(encoding="utf-8"))
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[1], ("Vulkan1", XT))
        self.assertEqual(devices[0][0], "Vulkan0")
        self.assertEqual(m.pick_device(devices, XT), ("Vulkan1", XT))

    def test_parse_devices_skips_the_ggml_vulkan_log_lines(self):
        text = (
            "ggml_vulkan: Found 2 Vulkan devices:\n"
            "ggml_vulkan: 0 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1\n"
            "ggml_vulkan: 1 = AMD Radeon RX 9070 XT (AMD proprietary driver) | uma: 0 | fp16: 1\n"
            "Available devices:\n"
            "  Vulkan0: AMD Radeon(TM) Graphics (48470 MiB, 46046 MiB free)\n"
            "  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)\n"
        )
        self.assertEqual(
            [d[0] for d in m.parse_devices(text)], ["Vulkan0", "Vulkan1"])
        only_log = "ggml_vulkan: 1 = AMD Radeon RX 9070 XT (AMD proprietary driver) | uma: 0\n"
        self.assertEqual(m.parse_devices(only_log), [])

    def test_pick_device_exact_then_unique_substring(self):
        devices = m.parse_devices(LISTING.read_text(encoding="utf-8"))
        self.assertEqual(m.pick_device(devices, XT)[0], "Vulkan1")
        self.assertEqual(m.pick_device(devices, "9070 XT")[0], "Vulkan1")
        self.assertEqual(m.pick_device(devices, "amd radeon rx 9070 xt")[0], "Vulkan1")

    def test_pick_device_refuses_ambiguity_and_absence(self):
        both = [("Vulkan0", XT), ("Vulkan1", "AMD Radeon RX 9070")]
        with self.assertRaises(SystemExit) as e:
            m.pick_device(both, "RX 9070")
        text = str(e.exception)
        self.assertIn("matches more than one llama.cpp device", text)
        self.assertIn(f"Vulkan0: {XT}", text)
        self.assertIn("Vulkan1: AMD Radeon RX 9070", text)
        # An exact name still decides between them.
        self.assertEqual(m.pick_device(both, "AMD Radeon RX 9070")[0], "Vulkan1")
        twins = [("Vulkan0", XT), ("Vulkan1", XT)]
        with self.assertRaises(SystemExit) as e:
            m.pick_device(twins, XT)
        self.assertIn("matches more than one llama.cpp device", str(e.exception))
        self.assertIn("Vulkan0", str(e.exception))
        self.assertIn("Vulkan1", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            m.pick_device(both, "RTX 4090")
        self.assertIn("no llama.cpp device named RTX 4090", str(e.exception))
        self.assertIn(f"Vulkan0: {XT}", str(e.exception))
        for blank in ("", "  "):
            with self.assertRaises(SystemExit) as e:
                m.pick_device(both, blank)
            self.assertIn("blank", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            m.pick_device([], XT)
        self.assertIn("no llama.cpp device named", str(e.exception))


class Arguments(unittest.TestCase):
    def test_base_args_name_no_device(self):
        self.assertNotIn("--device", m.BASE_ARGS)
        self.assertNotIn("Vulkan0", m.BASE_ARGS)
        self.assertEqual(m.BASE_ARGS[m.BASE_ARGS.index("-c") + 1], "32768")
        self.assertEqual(m.BASE_ARGS[m.BASE_ARGS.index("--n-cpu-moe") + 1], "24")

    def test_server_args_put_the_device_first(self):
        self.assertEqual(m.server_args("Vulkan1", ["-ngl", "99"]), ["--device", "Vulkan1", "-ngl", "99"])
        self.assertEqual(m.server_args("Vulkan1", m.BASE_ARGS)[:2], ["--device", "Vulkan1"])

    def test_ctx_replaces_the_context(self):
        args = m.with_ctx(m.BASE_ARGS, 65536)
        self.assertEqual(args[args.index("-c") + 1], "65536")
        self.assertEqual(len(args), len(m.BASE_ARGS))
        self.assertEqual(m.BASE_ARGS[m.BASE_ARGS.index("-c") + 1], "32768")
        self.assertEqual(m.with_ctx(["-ngl", "99"], 65536), ["-ngl", "99", "-c", "65536"])


class Masks(unittest.TestCase):
    def test_mask_hex(self):
        self.assertEqual(m.mask_hex(0xFFFF0000), "0xffff0000")
        self.assertEqual(m.mask_hex(0xFFFFFFFF), "0xffffffff")
        self.assertEqual(m.mask_hex(1), "0x1")

    def test_parse_mask(self):
        self.assertEqual(m.parse_mask("0xFFFF0000"), 0xFFFF0000)
        self.assertEqual(m.parse_mask("ffff"), 0xFFFF)
        for bad in ("", "0x", "zz", "0x0"):
            with self.assertRaises(SystemExit):
                m.parse_mask(bad)

    @unittest.skipUnless(os.name == "nt", "Windows affinity calls")
    def test_system_mask_and_process_affinity(self):
        system = m.system_affinity()
        self.assertGreater(system, 0)
        mine = m.process_affinity(os.getpid())
        self.assertEqual(mine & system, mine)

    @unittest.skipUnless(os.name == "nt", "Windows affinity calls")
    def test_set_affinity_reads_back_and_can_widen(self):
        system = m.system_affinity()
        before = m.process_affinity(os.getpid())
        lowest = system & -system
        try:
            self.assertEqual(m.set_affinity(lowest), lowest)
            self.assertEqual(m.process_affinity(os.getpid()), lowest)
            # Widening beyond the current mask works: the whole system mask comes back.
            self.assertEqual(m.set_affinity(system), system)
        finally:
            m.set_affinity(before)


class Config(unittest.TestCase):
    def test_a_search_result_replaces_the_args_and_affinity(self):
        best = {"best": {"args": ["-ngl", "99", "-c", "32768"], "affinity": "0xffff", "median": 90.0, "runs": []}}
        args, affinity = m.config_from_best(best)
        self.assertEqual(args, ["-ngl", "99", "-c", "32768"])
        self.assertEqual(affinity, 0xFFFF)
        with self.assertRaises(SystemExit):
            m.config_from_best({"best": {"args": ["-ngl"]}})


class FakeServer:
    pid = 4242
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class RunSix(unittest.TestCase):
    def run_with(self, masks):
        """run_six with a stub server and one stub request; the server's mask reads come from `masks`, in order."""
        from unittest import mock
        reads = iter(masks)
        fake_speed = mock.Mock()
        sampling = {"seeds": [1], "max_tokens": 1, "temp": 0, "top_p": 1, "top_k": 1, "min_p": 0}
        fake_speed.load.return_value = {"prompts": [{"id": "write"}], "sampling": sampling}
        fake_speed.message.return_value = "hi"
        reply = mock.MagicMock()
        body = b'{"timings": {"predicted_n": 5, "predicted_per_second": 90.5}}'
        reply.__enter__.return_value.read.return_value = body
        patches = [
            mock.patch.object(m, "port_in_use", return_value=False),
            mock.patch.object(m.urllib.request, "urlopen", return_value=reply),
            mock.patch.object(m, "healthy", return_value=True),
            mock.patch.object(m, "speed", fake_speed),
            mock.patch.object(m.subprocess, "Popen", return_value=FakeServer()),
            mock.patch.object(m, "process_affinity", side_effect=lambda pid: next(reads)),
        ]
        patches.append(mock.patch("builtins.print"))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return m.run_six(Path("bin"), "model.gguf", ("Vulkan1", XT), ["-ngl", "99"], 0xFFFF0000, Path("."))

    def test_a_server_on_another_mask_fails_the_run(self):
        out = self.run_with([0xFFFF, 0xFFFF])
        self.assertIn("affinity is 0xffff,", out["error"])
        self.assertEqual(out["affinity"], "0xffff")
        self.assertNotIn("median", out)

    def test_a_mask_that_changes_before_the_stop_fails_the_run(self):
        out = self.run_with([0xFFFF0000, 0xFFFF])
        self.assertIn("at stop", out["error"])
        self.assertEqual((out["affinity"], out["affinity_at_stop"]), ("0xffff0000", "0xffff"))

    def test_equal_masks_give_a_median(self):
        out = self.run_with([0xFFFF0000, 0xFFFF0000])
        self.assertNotIn("error", out)
        self.assertEqual(out["median"], 90.5)
        self.assertEqual(out["device"], {"id": "Vulkan1", "name": XT})
        self.assertEqual(out["args"], ["-ngl", "99"])
        self.assertEqual((out["affinity"], out["affinity_at_stop"]), ("0xffff0000", "0xffff0000"))
        self.assertEqual(out["affinity_asked"], "0xffff0000")


class Main(unittest.TestCase):
    def test_a_refusal_before_any_server_is_written_as_an_error_result(self):
        import json
        import shutil
        import tempfile
        from unittest import mock

        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        out = Path(folder) / "results.json"
        argv = ["speed_llama.py", ".", "model.gguf", "bin", "--out", str(out), "--gpu-name", "RTX 4090"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(m, "port_in_use", return_value=False):
            with mock.patch.object(m, "list_devices", return_value=[("Vulkan0", XT)]):
                with self.assertRaises(SystemExit):
                    m.main()
        written = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("no llama.cpp device named RTX 4090", written["error"])
        self.assertEqual(written["runs"], [])


if __name__ == "__main__":
    unittest.main()
