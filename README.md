# Pipeline support identification and dynamic strain prediction

Numerical data and reproducibility code accompanying:

**Dynamic strain prediction from statically identified pipeline support: effect of test-load uncertainty**

Jiahui Xing, Erxuan Cai, Zhiran Gao, Zhexi Xu, Minjie Wen and Yiming Zhang.
Jiahui Xing and Erxuan Cai contributed equally. Corresponding author: Yiming Zhang.

## Scientific scope

The study investigates how uncertainty in static test forces propagates through identified pipeline support fields into dynamic bending-strain predictions. All strain observations are synthetic. The dynamic inputs are controlled numerical excitations, not recorded earthquakes. Comparisons with published frequencies verify a forward calculation; they do not validate the inferred support against field measurements.

The elastic-layer models use finite reflective boundaries. Residual-incompatible static estimates and original unconverged grid cases are retained alongside the additional refinements. See `dynamic/DYNAMIC_PROTOCOL.md` and `dynamic/README_REPRO.md` for interpretation and which refined results enter the manuscript.

## Contents

- `static/`: static identification code, tests, frozen designs, numerical audit records and dependencies.
- `dynamic/`: dynamic code, frozen static fields and calibrations, complex strain arrays, pulse histories, soil-layer results, benchmark frequencies, tests and numerical audits.
- `docs/SOURCE_MANIFEST.csv`: source provenance and SHA-256 for every retained source/data file.
- `CITATION.cff`: author and citation metadata.

The complete static results and figure-summary tables are distributed separately in **`static-results.zip`**, as a GitHub Release asset and as part of the permanent research archive. They are intentionally excluded from Git history. The dynamic data are included in the repository.

Download both the source snapshot and `static-results.zip` for the same release. Extract the source snapshot first, then extract the data archive into its root, so that paths such as `static/results_load_information/` exist. Downloading GitHub's automatic source ZIP alone does not include the separately uploaded data asset.

## Reproduce and cite

Read [REPRODUCING.md](REPRODUCING.md) before executing scripts. Use an isolated Python environment and retain the distributed evidence when rerunning calculations.

Cite the exact released version and its archival DOI once assigned. The manuscript has not been represented here as accepted or published. See `CITATION.cff` for the named authors.

## Licenses

Original-project code/data licenses must be selected by the rights holders before public release; no new license has been assigned in this local preparation. Third-party license files remain in the `vendor/` trees and continue to apply to those components.

[中文说明](README_中文.md)
