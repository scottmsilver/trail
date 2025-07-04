#!/usr/bin/env python3
"""Test route CLI with mocked problematic import"""

import sys
import os
import types

# Mock the problematic import
mock_py3dep = types.ModuleType('py3dep')
sys.modules['py3dep'] = mock_py3dep

# Now run the route CLI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after mocking
import route_cli

if __name__ == "__main__":
    # Override sys.argv
    sys.argv = ["route_cli.py", "Start: 40.6572, -111.5709", "End: 40.6472,-111.5671"]
    route_cli.main()