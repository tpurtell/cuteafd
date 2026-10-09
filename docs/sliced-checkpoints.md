# Sliced checkpoints

Each host can hold only the shards its role reads. MiMo, GLM Flash and Qwen
coordinators need only their role's shards; official V4.1 Spark workers need
only expert shards. No annotation or side file is involved: roles are derived
from the checkpoint's own index and config.

## File lists

`cuteafd plan MODEL --files --role coordinator --spark-ranks 4` prints an rsync
file list for one role.

## Copying a role

Add `--fetch --destination /path/to/snapshot --dry-run` to preview a role-only
copy; omit `--dry-run` to copy. Transfers materialize HF blob symlinks as plain
snapshot files; files of matching size are skipped and nothing is deleted.

## Host layouts

`--file-layout layout.json --host worker` takes a JSON host-to-role map such as
`{"worker":["spark0","vision"]}`; omit `--host` with `--json` to list every
host.

- `--host worker --fetch` copies over SSH. Remote size checks are batched into
  one SSH session before and after copying.
- `--source peer:/snapshot` pulls from a peer (MODEL still supplies the local
  index and config). `--forward-agent` opts into `ssh -A` for peer-to-target
  copies (default off).
- `--source auto` reports whether sparknest serves a sealed local copy or
  streams; without it, the local snapshot is used.
- Omit `--host` with `--file-layout --fetch` to copy the whole layout, capped
  by `--fetch-parallel` (default 2).

## Separate repositories

`--drafter-snapshot`, `--vision-snapshot` and `--audio-snapshot` with `--json`
add separate repos for the selected host's enabled roles; fetch those snapshot
roots individually.

## Known limit (v2.0.0)

`--files` lists MTP layers only with `--include-speculator` and omits a bundled
`dflash/` drafter directory. A MiMo slice without `dflash/mask_embedding.pt`
serves correctly but drafts worse and decodes slower, without a warning. Until
the next release, copy the checkpoint's `dflash/` directory into the slice by
hand.
