# Observed Kepler light curve

`kepler8_q0.csv` contains observed long-cadence measurements for **KIC 6922244 (Kepler-8), quarter 0**. This is observed data, not a simulation. The built-in simulated transit/variable/flare examples are separately labeled in the app.

Source product: `kplr006922244-2009131105131_llc.fits`.

- Provider: NASA/Ames Kepler pipeline, distributed by STScI's Mikulski Archive for Space Telescopes (MAST).
- Retrieved November 5, 2026 from the [STScI public AWS object](https://stpubdata.s3.amazonaws.com/kepler/public/lightcurves/0069/006922244/kplr006922244-2009131105131_llc.fits), using the [documented public dataset](https://registry.opendata.aws/mast-kepler/).
- FITS header: `OBJECT=KIC 6922244`, `KEPLERID=6922244`, `QUARTER=0`, `ORIGIN=NASA/Ames`, product date `2015-08-26`.
- `TIME` is retained as `time_bkjd`: BKJD = BJD − 2454833, in days, TDB time system. No epoch shift, phase folding, interpolation, or binning is applied.
- `PDCSAP_FLUX` and `PDCSAP_FLUX_ERR` are retained in pipeline flux units (electrons per second); `SAP_QUALITY` is retained as `sap_quality`.
- Original file: 476 cadences; the bundled CSV retains 453 rows, spanning BKJD 120.539794062 to 130.246117528.
- Selection: retain finite `TIME` and `PDCSAP_FLUX` rows with `SAP_QUALITY == 0`. Preserve acquisition order. No additional detrending, clipping, or transit selection is performed.
- Relative-flux scaling in the app divides the displayed flux by its sample median. Centering subtracts the median. These operations do not fit or detrend the light curve.
- Uncertainties are preserved in the CSV for provenance but are not propagated through the educational operations.

The public dataset registry identifies Kepler mission data as public domain and gives the scientific-publication attribution requirements. This project acknowledges NASA's Kepler mission and STScI/MAST; the application code's MIT license does not replace the dataset's source terms.

Preparation used Astropy solely to extract the FITS table. The deployed app reads the bundled CSV and requires neither Astropy nor a runtime network request.
