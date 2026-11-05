"""A system whose package is not installed (tests/test_cli_build.py): importing it fails on a package that is not its own, which is how a
system with an extra that was not asked for fails."""

import not_installed_package_for_ghost  # noqa: F401


class Ghost:
    pass
