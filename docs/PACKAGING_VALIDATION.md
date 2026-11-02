# Packaging validation (2026-11-02)

- Verified 10986 retained files byte-for-byte against their source archive members using SHA-256.
- Python syntax compilation passed for 343 files; this does not verify numerical correctness.
- The separate static data archive passed CRC and per-file SHA-256 checks.
- Numerical tests and simulations were not rerun: the current interpreter has no NumPy, SciPy or pytest installed.
- Historical numerical audits remain historical evidence; they are not presented as new tests.
- No code, frozen design, input or numerical output file was edited during packaging.
- This is a local publication candidate. Remote URLs, release tag, DOI and original-project licenses remain to be finalized by the authors.
