# Synthetic Precitec CSV Fixtures

These files are fully synthetic test fixtures and contain no production or customer measurements.

- `dummy_Altitude_Peak_Processed.csv`: synthetic altitude signal (`IdSignal=16640`)
- `dummy_Intensity_Peak_Processed.csv`: synthetic intensity signal (`IdSignal=16641`)

Both files share the same non-measured mask pattern (value `0`) so parser tests can verify masking behavior deterministically.
