# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys
import os
from datetime import date

# import sphinx_rtd_theme
from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------

version = get_distribution("km3irf").version
short_version = ".".join(version.split(".")[:2])
doc_version = ".".join(version.split("+")[:1])
project = "km3irf {}".format(short_version)
copyright = "{0}, Tamas Gal and Mikhail Smirnov".format(date.today().year)
author = "Tamas Gal, Mikhail Smirnov"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "version.py"]

# AutoAPI
autoapi_type = "python"
autoapi_dirs = ["../src/km3irf"]
autoapi_options = ["members", "undoc-members", "show-module-summary"]
autoapi_include_summaries = True

# Gallery
sphinx_gallery_conf = {
    "backreferences_dir": "gen_modules",
    "default_thumb_file": "_static/default_gallery_thumbnail.png",
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_title = "km3irf {}".format(version)
html_logo = "_static/logo_km3irf_light.png"

# Specify icons from pydata_sphinx_theme
html_theme_options = {
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://git.km3net.de/km3py/km3irf",
            "icon": "fa-brands fa-gitlab",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/km3irf",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "km3irf {}".format(doc_version),
        "image_dark": "_static/logo_km3irf_dark.png",
        "alt_text": "km3irf",
    },
}
