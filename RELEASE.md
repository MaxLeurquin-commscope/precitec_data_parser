# Release Guide

This document describes how to release a new version of `precitec-data-parser` to PyPI using GitHub Actions.

## Prerequisites

1. Your PyPI API token should already be set in GitHub Secrets (see Setup section below)
2. Make sure all tests pass locally before releasing
3. Update the version number before creating a release

## Step 1: Update Version Number

Update the version in two places:

### In `pyproject.toml`

```toml
[project]
name = "precitec-data-parser"
version = "0.1.1"  # Bump this
```

### In `src/precitec_data_parser/__init__.py`

```python
__version__ = "0.1.1"  # Bump this
```

## Step 2: Commit Changes

```bash
git add .
git commit -m "Bump version to 0.1.1"
git push
```

## Step 3: Create a GitHub Release

1. Go to your repository on GitHub
2. Click on **Releases** (right sidebar)
3. Click **Create a new release**
4. Fill in the details:
   - **Tag version**: `v0.1.1` (must match `version` in pyproject.toml)
   - **Release title**: `v0.1.1` or `Release 0.1.1`
   - **Description**: List what changed (bug fixes, new features, etc.)
   
   Example description:
   ```
   ## Changes
   - Fixed bug in oblique profile calculation
   - Improved error messages
   - Updated documentation
   ```

5. Click **Publish release**

## What Happens Next

GitHub Actions automatically:
- ✅ Runs tests (from `.github/workflows/tests.yml`)
- ✅ Builds your package
- ✅ Verifies the distribution
- ✅ Publishes to PyPI

You can watch the progress in the **Actions** tab on GitHub.

## Verify Release

After ~1 minute, check that your package is live:

```bash
pip install --upgrade precitec-data-parser
```

Or visit: https://pypi.org/project/precitec-data-parser/

## Setup Instructions (One-time)

### Add PyPI API Token to GitHub Secrets

1. **Create/Get your PyPI API token**:
   - Go to https://pypi.org/manage/account/
   - Scroll to "API tokens"
   - Create a new token (scoped to this project if possible)
   - Copy the token (starts with `pypi-`)

2. **Add to GitHub Secrets**:
   - Go to your GitHub repository
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - **Name**: `PYPI_API_TOKEN`
   - **Secret**: Paste your PyPI API token
   - Click "Add secret"

That's it! The workflow will now use this token to publish.

## Troubleshooting

### Release didn't publish to PyPI

Check the workflow logs:
1. Go to **Actions** tab on GitHub
2. Click the failed workflow run
3. Click **Publish to PyPI** job
4. Review the error message

Common issues:
- Version tag doesn't match `pyproject.toml` version
- API token is invalid or expired
- Package name is already taken on PyPI

### Manual Fallback

If GitHub Actions fails, you can always publish manually:

```bash
python -m build
twine upload dist/*
```

## Semantic Versioning

Use [Semantic Versioning](https://semver.org/):

- `0.1.0` → `0.1.1`: Bug fixes (patch)
- `0.1.0` → `0.2.0`: New features (minor)
- `0.1.0` → `1.0.0`: Breaking changes (major)

Example release progression:
```
v0.1.0 (initial release)
v0.1.1 (bug fix)
v0.1.2 (bug fix)
v0.2.0 (new features)
v1.0.0 (first stable release)
```
