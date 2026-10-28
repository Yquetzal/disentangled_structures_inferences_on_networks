# Publishing version 2

The distribution name stays **`disentangled-net`**, with Python import
`disentangled_net`. Version 2.0.0 intentionally breaks the 0.1 API; see
{doc}`whats_new`. A GitHub fork does not create a separate PyPI project or grant
upload rights to the existing one.

## Access needed once

1. Create a PyPI account, verify its email address and enable two-factor
   authentication. Ask a current owner of `disentangled-net` to invite your PyPI
   username as **Owner** (recommended for managing publication) or **Maintainer**
   (can upload, cannot administer collaborators). Accept the invitation. The
   public metadata currently points to Guilhem Dupuy's original repository;
   check the actual owner account on the PyPI project page before requesting access.
2. In the existing PyPI project's **Manage → Publishing**, add this GitHub
   Trusted Publisher. An existing owner can configure it for the fork without
   moving or renaming the PyPI project:

   | PyPI field | Value |
   |---|---|
   | GitHub owner | `Yquetzal` |
   | Repository | `disentangled_structures_inferences_on_networks` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

3. In GitHub repository settings, create environments `pypi` and `testpypi`.
   Require a release reviewer for `pypi` where the repository plan permits it.
   The workflow itself always requires an explicit manual dispatch.
4. TestPyPI has separate accounts and projects. Configure the same publisher
   there with environment **`testpypi`**. If that project name is available,
   create a pending publisher; if someone owns it already, obtain access or
   skip TestPyPI and validate the local wheel instead.

Trusted Publishing uses short-lived credentials from GitHub Actions: there is
no PyPI password or permanent API token to store in this repository.

Official instructions: [collaborator roles and invitations](https://docs.pypi.org/project-management/managing-collaborators/),
[adding a Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/),
[PyPA GitHub Actions guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

## Local verification

From this repository's root, in a dedicated virtual environment:

```bash
python -m pip install ".[test,docs,embeddings]" build twine
python -m pytest -q
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check --strict dist/*
```

Install the wheel in another environment and run an example outside the source
directory. `dist/` contains the wheel and source archive; neither belongs in Git.
The CI repeats tests on Python 3.9, 3.12 and 3.13, including PyTorch on the latter
two, and builds the documentation and distributions. Python 3.9 also tests the
minimum NumPy 1.22.0, SciPy 1.11.4 and NetworkX 2.8 versions. Older SciPy 1.8
wheels crashed in their native eigensolver on the review machine (macOS ARM64),
so the supported lower bound is now 1.11.4.

## Put the reviewed code on your fork

Use the **`disentangled-net-v2` checkout**, whose `origin` is your fork. The old
`disentangled_structures_inferences_on_networks` directory is a separate checkout.
Review `git status` and `git diff` first: the v2 rewrite already includes staged
deletions of the old API and archives. Keep those deletions when committing the
replacement package.

```bash
git switch -c release/v2.0.0
git add -A
git diff --cached --stat
git commit -m "Prepare disentangled-net 2.0.0"
git push -u origin release/v2.0.0
```

Open a pull request into **your fork's `main`**, let CI pass, then merge it. The
workflow files must be on the default branch for manual dispatch to be available.
No upstream pull request or synchronization is required to publish from your fork.

## Publish the tagged version

After merging, tag the reviewed commit on `main` and push that tag. Ensure
`pyproject.toml`, `disentangled_net.__version__` and `docs/conf.py` agree first.

```bash
git switch main
git pull --ff-only origin main
git tag v2.0.0
git push origin v2.0.0
gh workflow run publish.yml --ref v2.0.0 -f index=testpypi
```

Inspect the run and the TestPyPI distribution. A clean installation can install
NumPy, SciPy and NetworkX from PyPI first, then the test package without fetching
dependencies from TestPyPI:

```bash
python -m pip install numpy scipy networkx
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ disentangled-net==2.0.0
```

Then explicitly start the production upload:

```bash
gh workflow run publish.yml --ref v2.0.0 -f index=pypi
```

Both runs check that the tag matches the package version, run the tests, build
the docs and validate wheel/source metadata before uploading. A commit or tag
push alone does not publish anything. Published filenames cannot be replaced;
fixes after publication require a new version, for example 2.0.1.

## Documentation and article before submission

### First Read the Docs deployment, before any PyPI release

The existing `.readthedocs.yaml` installs the code from the selected Git branch
with its `docs` extra, using Python 3.12, then builds `docs/conf.py`. It does not
install the old PyPI release and does not require PyPI publishing rights or
PyTorch. Builds fail on Sphinx warnings and unresolved references.

1. Commit and push the reviewed **`v2` branch** to your fork. Include the code,
   `pyproject.toml`, `docs/`, `.readthedocs.yaml` and license: API documentation
   imports the current library, so uploading only `docs/` is insufficient.
   There is no need to push generated `docs/_build/` files.
2. Sign in to [Read the Docs Community](https://app.readthedocs.org/) with GitHub.
   Install its GitHub App with access to
   `Yquetzal/disentangled_structures_inferences_on_networks`.
3. Choose **Add project**, select that repository and continue. Use project
   name `disentangled-net` if available, with English documentation and the
   configuration path `.readthedocs.yaml`.
4. In the project settings, set **Default branch** to `v2` for this initial
   deployment. This makes `latest` build the current library, even while the
   repository's `main` branch still contains the previous version. Leave the
   default documentation version as `latest`.
5. Trigger/check the `latest` build in **Builds**, then open **View docs**. If the
   project name is available, the expected address is
   `https://disentangled-net.readthedocs.io/en/latest/`; use the actual address
   returned by Read the Docs.
6. Subsequent pushes to `v2` rebuild `latest` automatically through the GitHub
   integration. After merging v2 into `main`, change the Read the Docs default
   branch to `main`. Later, activate release tags for versioned documentation.

After the first successful hosted build, replace the temporary GitHub
documentation URL in `pyproject.toml` and validate the article's documentation
link. The name and URL must reflect the project actually registered.

Read the Docs is a conventional choice for Python/Sphinx projects, with release
versions managed by the service. GitHub Pages is also standard and avoids an
additional hosting account, but needs a Pages build/deployment workflow and
does not provide Read the Docs' version management automatically. Both can
serve the existing Sphinx output without changing the documentation format.

References: [adding a project](https://docs.readthedocs.com/platform/stable/intro/add-project.html),
[branches and versions](https://docs.readthedocs.com/platform/stable/versions.html),
[Sphinx hosting choices](https://www.sphinx-doc.org/en/master/tutorial/deploying.html).

### Article availability

The article's current public experiment URL refers to the earlier experiments.
Publish the current `article_figures/` and `Claude/article_v2_*` sources, settings
and data instructions in an appropriate reproducibility repository, then update
the article's link and add an archival DOI. The library release does not itself
publish these experiments or make the article's current availability claims true.
