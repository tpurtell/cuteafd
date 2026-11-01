"""CPU-only Muse processor span goldens generated once from the official code."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
spec = importlib.util.spec_from_file_location("muse_resize_fixture", FIXTURES / "generate-muse-vision-resize.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_official_resize_for_all_sizes_and_decode_extremes(tmp_path):
    rustc = shutil.which("rustc")
    assert rustc, "rustc is required for the CPU processor golden sweep"
    fixture = zlib.decompress((FIXTURES / "muse-vision-resize.zlib").read_bytes())
    square_bytes = generator.SIDE * generator.SIDE * 2
    official = bytearray()
    for gh, gw in struct.iter_unpack("<BB", fixture[:square_bytes]):
        official.extend(struct.pack("<II", gh * 28, gw * 28))
    assert hashlib.sha256(official).hexdigest() == "1f3e9e0c60a8cf6900877a7788d765789118a0d7e1f97e131d177a64897f5154"
    source = tmp_path / "resize.rs"
    source.write_text(f'''#[path = "{ROOT}/rust/crates/cuteafd-loader/src/media/muse.rs"] mod muse;
use std::io::{{Read,Write}};
fn main() {{
 let mut input=Vec::new();std::io::stdin().read_to_end(&mut input).unwrap();
 let mut out=std::io::BufWriter::new(std::io::stdout().lock());
 let mut put=|h,w,compact| {{ let (rh,rw)=muse::resize_shape(h,w,28,4096).unwrap();
  let rows=(rh/28)*(rw/28);
  assert!(rh%28==0 && rw%28==0 && rows>0);
  if compact {{out.write_all(&[u8::try_from(rh/28).unwrap(),u8::try_from(rw/28).unwrap()]).unwrap();}}
  else {{out.write_all(&rh.to_le_bytes()).unwrap();out.write_all(&rw.to_le_bytes()).unwrap();}}
 }};
 for h in 1..=1792 {{for w in 1..=1792 {{put(h,w,true);}}}}
 for pair in input.chunks_exact(8) {{let h=u32::from_le_bytes(pair[..4].try_into().unwrap());
  let w=u32::from_le_bytes(pair[4..].try_into().unwrap());put(h,w,false);}}
}}
''')
    executable = tmp_path / "resize"
    subprocess.run([rustc, "--edition=2021", "-O", str(source), "-o", str(executable)], check=True, timeout=30)
    cases = b"".join(struct.pack("<II", *pair) for pair in generator.extreme_inputs())
    result = subprocess.run([str(executable)], input=cases, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()
    if result.stdout != fixture:
        count = min(len(result.stdout), len(fixture))
        mismatch = next((i for i in range(count) if result.stdout[i] != fixture[i]), count)
        if mismatch < square_bytes:
            pixel = mismatch // 2
            raise AssertionError(f"official resize differs at h={pixel // 1792 + 1}, w={pixel % 1792 + 1}: byte {mismatch}")
        raise AssertionError(f"official extreme resize differs at byte {mismatch}")


def test_fp32_yardstick_preserves_audio_error_ratio_and_floor():
    import numpy as np
    import pytest
    spec = importlib.util.spec_from_file_location("muse_vision_qualifier",
        ROOT / "python/tools/qualify/muse_spark/qualify-vision.py")
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    fp32 = np.ones((2, 3), dtype=np.float64)
    bf16 = fp32 + 1
    accepted = qualifier.stage_metrics(fp32 + 1.09, bf16, fp32)
    rejected = qualifier.stage_metrics(fp32 + 1.11, bf16, fp32)
    assert accepted["passed"] and accepted["error_ratio"] == pytest.approx(1.09)
    assert not rejected["passed"] and rejected["error_ratio"] == pytest.approx(1.11)
    assert qualifier.stage_metrics(fp32 + 1e-8, fp32, fp32)["passed"]
    assert not qualifier.stage_metrics(fp32 + 2e-7, fp32, fp32)["passed"]
    assert not qualifier.stage_metrics(fp32 * np.nan, bf16, fp32)["passed"]
    with pytest.raises(ValueError, match="shapes differ"):
        qualifier.stage_metrics(fp32[:1], bf16, fp32)
