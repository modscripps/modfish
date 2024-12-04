# modfish
Python package for reading data from FastCTD and Epsifish developed by the Multiscale Ocean Dynamics group.

Documentation at https://modscripps.github.io/modfish

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
This software comes with [pdoc](https://pdoc.dev/) documentation. The included [Makefile](Makefile) has recipes `servedocs` to display the documentation, including a watchdog that acts on file changes, and `docs` to generate the html files.

After cloning this repository, run `git submodule update --init --recursive` to fetch the theme for the docs.

## Credits
This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[gunnarvoet/cookiecutter-python-package](https://github.com/gunnarvoet/cookiecutter-python-package)
project template.

## License
Free software: GNU General Public License v3
