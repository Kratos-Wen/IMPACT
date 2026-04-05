# Licensing

IMPACT uses a split license structure to distinguish software from dataset assets and documentation.

Software:
- [LICENSE](../LICENSE) applies to repository-authored code, scripts, and configuration files, unless otherwise noted

Dataset assets and documentation:
- [LICENSE-DATA](../LICENSE-DATA) applies to dataset assets in `dataset/`, repository-authored documentation in `docs/`, the top-level `README.md`, and Markdown documentation under `tasks/`, unless otherwise noted

Third-party materials:
- `third_party/` uses per-directory provenance and license notices
- bundled upstream licenses are retained where available
- IMPACT-maintained method snapshots document their external references and licensing status in the corresponding `README.md`

Precedence:
- if a file or subdirectory contains a more specific license notice, that notice takes precedence over the repository-level defaults
