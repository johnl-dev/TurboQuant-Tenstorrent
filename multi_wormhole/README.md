# Multi-Wormhole / Mesh Implementation (planned)

Goal: scale TurboQuant across the 32-chip mesh, not just the single chip
the baseline uses.

## Current baseline

The working pipeline runs on `MeshDevice::create_unit_mesh(6)` -- a single
chip (chip 6, the local L-die that's PCIe-accessible). Of 32 physical
chips, 31 are completely idle. Cluster utilization: ~3%.

## What this folder is for

Mesh-scaled implementation:

- **Sharding strategy:** TurboQuant is embarrassingly parallel at the
  vector level (no cross-vector dependencies in any stage). Simplest
  sharding: N/32 vectors per chip, same per-vector pipeline independently
  on each. Fabric needed only for input distribution and output gather.

- **Multi-mesh layout:** Topology currently maps as 2x1 (the
  TopologyMapper warning in run logs flagged this as a downgrade from
  the expected layout). Worth understanding why before designing.
  32 chips split into 16 local (PCIe) and 16 remote across two physical
  meshes.

- **Dispatch caveats:**
  - Dispatch cores (0,7), (1,7), (2,7), (3,7) reserved per chip
  - Remote chips (16-31) reachable only via fabric routing
  - For streaming KV-cache workloads, keep cache on L-die and route only
    summary stats across fabric

- **Memory budget:** Each chip has 12 GB GDDR6 across 12 banks. Sharding
  1M vectors of d=128 across 32 chips is ~31k per chip, trivially small.

## Open design questions

- Contiguous vector ranges per chip, or interleaved?
- Single MeshWorkload spanning all chips, or one per chip in parallel?
- How does this integrate with the attention dequant kernel that will
  eventually read QJL output?

## Prerequisite

Don't start this until `multi_tile/` works on a single chip. Sharding
scalar-Brisc code across the mesh is the wrong order of optimization.
