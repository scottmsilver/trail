#!/usr/bin/env python3
"""
Test that the CLI uses the correct terminology (no "cache" references)
"""

import subprocess
import sys

def test_help_output():
    """Test that help output doesn't contain 'cache' references"""
    cmd = [sys.executable, "../elevation.py", "--data-dir", "./test_data"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    help_text = result.stdout + result.stderr
    
    # Check for any remaining cache references
    if "cache" in help_text.lower():
        print("❌ Found 'cache' in help output:")
        for line in help_text.split('\n'):
            if "cache" in line.lower():
                print(f"   {line}")
    else:
        print("✓ No 'cache' references found in help output")
    
    # Check for correct terminology
    if "loaded" in help_text:
        print("✓ Found 'loaded' terminology in help")
    else:
        print("❌ Missing 'loaded' terminology in help")
    
    print("\nHelp output:")
    print("-" * 50)
    print(help_text)

if __name__ == "__main__":
    test_help_output()