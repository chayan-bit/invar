# Real scan sample

`pcl4096-f32-soa.bin` contains 4,096 untransformed XYZ points from PCL's LMS400 table scan.
The source revision, source and sample SHA-256 digests, license blob, field layout, selection rule, bounds, and query indices are in [the lineage receipt](pcl4096-lineage.json).
The upstream BSD-3-Clause notice is retained in [PCL-DATA-LICENSE.txt](PCL-DATA-LICENSE.txt).

The binary stores all X values, then Y, then Z, as little-endian binary32 words.
Sample `j` selects source point `floor(j * 460400 / 4096)`.
The demo's decimal literals round-trip to those words; exponent notation is expanded for the current lexer.
The query points are sample indices `0, 256, ..., 3840`, with radius and cell width `0x3d4ccccd` and output stride 32.

`tests/v02_spatial_demo.rs` verifies the sample digest, sums true counts from four independent 1,024-point oracle blocks, and compares the actual reference CLI result.
The Metal CLI and VHS recording return 264 total neighbors across the 16 queries.
This is a correctness demonstration, not a benchmark or normal-estimation example.
