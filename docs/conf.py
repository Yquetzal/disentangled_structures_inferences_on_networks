project = "disentangled-net"
author = "Guilhem Dupuy, Rémy Cazabet"
release = "2.0.0"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath"]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_use_param = False
nitpicky = True

html_theme = "furo"
html_title = "disentangled-net"
html_static_path = ["_static"]
