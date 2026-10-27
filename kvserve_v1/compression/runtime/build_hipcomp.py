# SPDX-License-Identifier: Apache-2.0
"""Build the runtime HIP shim using the server's hipcc and hipCOMP installation.

python -m kvserve_v1.compression.runtime.build_hipcomp --hipcomp-root /opt/hipcomp
This does not download, install or substitute any CUDA/HIP dependencies.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def build_command(args) -> list[str]:
    root = Path(args.hipcomp_root).expanduser().resolve()
    include_dirs = [root / "include", *(Path(p).resolve() for p in args.include_dir)]
    library_dirs = [
        root / "lib",
        root / "lib64",
        *(Path(p).resolve() for p in args.library_dir),
    ]
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    hipcc = args.hipcc or shutil.which("hipcc") or str(rocm / "bin/hipcc")
    source = Path(__file__).resolve().parent / "csrc/kvs_hipcomp.cpp"
    command = [hipcc, "-x", "hip", "-O3", "-std=c++17", "-fPIC", "-shared", str(source)]
    command += [f"-I{p}" for p in include_dirs]
    command += [f"-L{p}" for p in library_dirs]
    command += [f"-Wl,-rpath,{p}" for p in library_dirs]
    command += [f"-l{args.hipcomp_library}", "-o", str(args.output)]
    if args.offload_arch:
        command.append(f"--offload-arch={args.offload_arch}")
    return command


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hipcomp-root",
        default=os.environ.get("HIPCOMP_ROOT", "/opt/rocm"),
        help="Installed hipCOMP prefix containing include/hipcomp and lib[64]",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="Extra header directory; repeatable",
    )
    parser.add_argument(
        "--library-dir",
        action="append",
        default=[],
        help="Extra library directory; repeatable",
    )
    parser.add_argument(
        "--hipcomp-library", default="hipcomp", help="Link name without lib prefix/.so"
    )
    parser.add_argument("--hipcc", help="Path to the DCU toolchain's hipcc")
    parser.add_argument(
        "--offload-arch", help="Target architecture, as accepted by your hipcc"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "build/hipcomp_runtime/libkvserve_hipcomp_runtime.so",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not shutil.which(command[0]):
        parser.error("hipcc not found; activate the DCU toolchain or pass --hipcc")
    if not any(
        (Path(p) / "hipcomp/snappy.h").is_file()
        and (Path(p) / "hipcomp/cascaded.h").is_file()
        for p in [Path(args.hipcomp_root) / "include", *map(Path, args.include_dir)]
    ):
        parser.error("hipCOMP headers missing; set --hipcomp-root/--include-dir")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    print(f"export KVSERVE_HIPCOMP_LIBRARY={shlex.quote(str(args.output))}")


if __name__ == "__main__":
    main()
