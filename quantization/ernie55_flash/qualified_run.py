"""Bind the measured maxq forward to the production writer and layer scheduler.

The measured parity digest included two export-only files. The grouped HCA
writer changes how they describe the native row layout; they never participate
in the measured forward. The identity module only gained the qualified-session
callback. Match exactly those reviewed transitions when checking historical
receipts. Keep the actual code checksums in the frozen production identity.
"""
from . import identity
from .bootstrap import inputs_key

# (measured SHA256, production SHA256)
TRANSITIONS = {
    "quantization/inkling2/export.py": (
        "f5874858679778ffca5f4c7cade7b9424a72379dfdcf02cf4ae0245d39551aa5",
        "1f44fd955fcbc1c267a06a886d78cb3f27ea43e124f3cea824541010b9aaeaf7"),
    "quantization/mc_package.py": (
        "14477c7c7d534b9914543b7535c7f7393d50344c9928eb94640b5ee151e270b1",
        "6e14be96ad511d78aa3eb23aea2f53a532863e7081c8d00e62bfe31bd529553a"),
    "quantization/ernie55_flash/identity.py": (
        "622afb1fa66c4d4e2d916e6e516895fe1eabcab93151756127c7759126583a26",
        "9d8706e4581eed094b8200d51641cc7ffeb10e8dde0349f84a5f1967455fe28a"),
}
DEPENDENCIES = tuple("quantization/hy4_flash/" + name + ".py" for name in ("run", "gates", "forward"))


def measured_inputs(inputs):
    files = inputs["code"].copy()
    for path, (measured, production) in TRANSITIONS.items():
        if files.get(path) != production:
            raise ValueError(f"unreviewed ERNIE producer code transition: {path}")
        files[path] = measured
    return dict(inputs, code=files)


def bootstrap_key(inputs):
    return inputs_key(measured_inputs(inputs))


def qualify(inputs, vendor, local):
    measured = measured_inputs(inputs)
    checked = identity.verified_identity(measured, vendor, local)
    return dict(checked, inputs=inputs,
                measured_parity_input_sha256=identity.parity_input_digest(measured),
                shared_production_files=identity.committed_files(identity.REPO, *DEPENDENCIES))


def session(root, snapshot, corpus, vendor, local):
    return identity.producer_session(root, snapshot, corpus, vendor, local, qualify=qualify)
