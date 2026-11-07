# Third-party software

e57stitch itself is MIT licensed (see `LICENSE`). It builds on the projects
below. The versions are the ones pinned in `constraints.txt` and shipped in the
Docker image.

## Included in this repository

| component | where | licence |
| --- | --- | --- |
| [three.js](https://threejs.org) r169 | `e57stitch/web/vendor/three.module.js` | MIT, Copyright 2010-2024 Three.js Authors (notice kept in the file header) |

## Installed as dependencies (not in this repository)

These are installed from PyPI by `pip install .` and are part of the Docker
image. Each package's own licence file is installed with it
(`site-packages/<name>-<version>.dist-info/licenses/`).

| package | used for | licence |
| --- | --- | --- |
| [Open3D](https://www.open3d.org) 0.20 | point clouds, normals, FPFH features, RANSAC/FGR global registration, point-to-plane, coloured and generalised ICP, DBSCAN, plane fitting, Poisson and ball-pivoting meshing | MIT; the wheel also contains third-party libraries under their own permissive licences (listed in Open3D's `3rdparty` folder), including Intel oneTBB (Apache-2.0) |
| [small_gicp](https://github.com/koide3/small_gicp) 1.0 | fast GICP / VGICP for the final polish | MIT |
| [pye57](https://github.com/davidcaron/pye57) 0.4 | reading and writing E57 | MIT; contains [libE57Format](https://github.com/asmaloney/libE57Format) (BSL-1.0) and [Xerces-C++](https://xerces.apache.org/xerces-c/) (Apache-2.0) |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause (plus bundled 0BSD/MIT/Zlib parts) |
| [SciPy](https://scipy.org) | KD-trees, rotations, least squares, image filters | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com) / [Starlette](https://www.starlette.io) | web API | MIT / BSD-3-Clause |
| [Uvicorn](https://www.uvicorn.org) | web server | BSD-3-Clause |
| [Pydantic](https://docs.pydantic.dev) | request models | MIT |
| [Pillow](https://python-pillow.org) | panorama images | MIT-CMU |

Open3D pulls in further pure-Python packages (Dash, Flask, plotly, ipywidgets
and their dependencies) that e57stitch does not use. They are MIT, BSD,
Apache-2.0, ISC or PSF licensed, apart from certifi (MPL-2.0, unmodified).

## Docker image only

The image is based on `python:3.12-slim-bookworm` (Debian 12 and CPython,
PSF licence) and adds these Debian packages, unmodified:

| package | licence |
| --- | --- |
| libgl1, libegl1 (libglvnd) | MIT-style |
| libx11-6 | MIT/X11 |
| libglib2.0-0 | LGPL-2.1-or-later |
| libusb-1.0-0 | LGPL-2.1-or-later |
| libgomp1 (GCC) | GPL-3.0 with the GCC Runtime Library Exception |
| sqlite3 | public domain |

Their licence texts are in the image under `/usr/share/doc/<package>/copyright`,
and their source code is available from Debian
(`apt-get source <package>`, or <https://sources.debian.org>).
