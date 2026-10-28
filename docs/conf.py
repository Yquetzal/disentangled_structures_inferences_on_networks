project = "disentangled-net"
author = "Guilhem Dupuy, Rémy Cazabet"
copyright = "2026, Guilhem Dupuy and Rémy Cazabet"
release = "2.0.0"
exclude_patterns = ["_build", "**/.DS_Store"]

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath"]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_use_param = False
nitpicky = True

html_theme = "sphinx_book_theme"
html_title = "disentangled-net"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_context = {"default_mode": "light"}
html_theme_options = {
    "repository_url": "https://github.com/Yquetzal/disentangled_structures_inferences_on_networks",
    "repository_branch": "v2",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "show_navbar_depth": 1,
    "max_navbar_depth": 2,
    "show_toc_level": 2,
}
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "project-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
