# kinodb — Python bindings for the kinodb trajectory database
#
# The core functionality (open, Database, query) comes from the
# Rust extension module. PyTorch integration is in kinodb.torch.

from kinodb.kinodb import open, Database

__all__ = ["open", "Database"]
__version__ = "0.1.0"
