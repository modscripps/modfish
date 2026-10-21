# modfish
Python package for processing data from the FastCTD and Epsifish profilers of the Multiscale Ocean Dynamics group at Scripps.

- `modfish.modraw` decodes raw FCTD `.modraw` files into per-stream NetCDF (L0).
- `modfish.fctd` concatenates the L0 files of a deployment, detects casts, applies the T-C sensor corrections and writes an L1 time series and a depth-gridded product (`fctd.process_deployment`).
- `modfish.chi` computes microconductivity chi from the L0 `efe/c1` stream and adds it to both products.
- `modfish.tc` holds the T-C corrections (sensor lag, thermistor time constant, thermal mass) and the tools to estimate their parameters.
- `modfish.io` reads legacy Matlab-gridded FCTD and Epsifish output.

Documentation at https://modscripps.github.io/modfish

## Example
```python
from pathlib import Path
import modfish

raw = sorted(Path("raw").glob("*.modraw"))
modfish.modraw.convert(raw, "l0")
l0 = sorted(Path("l0").glob("*.nc"))
config = modfish.fctd.FCTDConfig.from_yaml("fctd.yml")  # start from examples/fctd_example.yml
l1_path, grid_path = modfish.fctd.process_deployment(l0, "proc", "d07", config=config)
```

The `examples/` folder holds three files.
`fctd_processing.ipynb` is an executed notebook that processes one deployment end to end and fits the T-C and chi gain values on its own data. Its header lists the uv and kernel setup steps.
`fctd_example.yml` is a template configuration with every key at its default, except the recommended chi `rail_lo`, `rail_hi` and `min_bins`, and with commented reference values.
`fctd_modfish_notes.md` compares modfish with the Matlab FCTD routines for users who port changes into Matlab.

## How to Install
Clone or download the repository and install via `pip install <path-to-modfish>` or, to be able to make changes to the code on the fly, as editable package via `pip install -e <path-to-modfish>`.

With [uv](https://docs.astral.sh/uv/), the package can be added via `uv add <path-to-modfish>` or `uv add --editable <path-to-modfish>`.

Note: The package has **not** been published to pip so you need to install from sources.

`tc.plot_spectra` needs matplotlib, available as the optional extra `modfish[plot]`.

## How to Contribute
- Fork the repository
- Install your fork as editable, e.g. via `pip install -e modfish`
- Create a development branch for your feature
- Commit changes to your branch and push to GitHub
- Create pull request on GitHub
- Once changes have been merged, integrate them into your main via `git fetch upstream` and `git merge upstream/main`
- Push changes to your fork via `git push origin`
- Safely delete your local development branch


## Tools
This software comes with [pdoc](https://pdoc.dev/) documentation.
The included [Makefile](Makefile) has recipes `servedocs` to display the documentation, including a watchdog that acts on file changes, and `docs` to generate the html files.
Note that make commands rely on uv.

After cloning this repository, run `git submodule update --init --recursive` to fetch the theme for the docs.

## Credits
This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[gunnarvoet/cookiecutter-python-package](https://github.com/gunnarvoet/cookiecutter-python-package)
project template.

## License
Free software: GNU General Public License v3
