# Copyright 2023 M Chimiste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "1.0.1"
def __getattr__(name):
    """Keep package imports free of model loading, media setup and DB work."""
    from importlib import import_module
    if name == "TheseusInsight":
        return import_module(".theseus_insight", __name__).TheseusInsight
    if name in {"PodcastGenerator", "generate_visualizer_video"}:
        return getattr(import_module(".podcast", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# -----------------------------------------------------------------------------
# Global debug / logging configuration
# -----------------------------------------------------------------------------

import os, builtins, logging

# Enable extra verbosity only if environment variable DEBUG is truthy
_DEBUG_MODE = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

# Instead of globally monkey-patching `print`, which breaks libraries that
# introspect built-ins (Numba, TensorFlow, etc.), we simply encourage use of
# the standard `logging` module and leave `print` untouched.
#
# We do not globally redefine `print`; third-party packages relying on built-in
# introspection (e.g. Numba) keep working.  Use the standard `logging` module
# for runtime diagnostics instead.

# Set a sensible global logging level
logging.basicConfig(
    level=logging.DEBUG if _DEBUG_MODE else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Expose DEBUG_MODE for other modules (optional)
DEBUG_MODE: bool = _DEBUG_MODE
