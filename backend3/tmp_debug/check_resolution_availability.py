#!/usr/bin/env python3
"""
Check what elevation resolutions are available for a given area
"""

import py3dep
from elevation import Bounds

def check_available_resolutions(bounds: Bounds):
    """Check which resolutions are available for an area"""
    print(f"\nChecking elevation data availability for:")
    print(f"  Bounds: {bounds.south:.4f},{bounds.west:.4f} to {bounds.north:.4f},{bounds.east:.4f}")
    print("\nTesting resolutions:")
    
    resolutions = [1, 3, 5, 10, 30, 60]
    available = []
    
    for res in resolutions:
        try:
            # Try to get just the metadata without downloading
            print(f"\n  {res}m: ", end="", flush=True)
            
            # py3dep will attempt to access the data
            # This is a quick check - actual download may still fail
            dem = py3dep.get_dem(
                bounds.to_tuple(),
                resolution=res,
                crs="EPSG:4326"
            )
            
            if dem is not None and dem.size > 0:
                print(f"✓ Available (shape: {dem.shape})")
                available.append(res)
                
                # Calculate approximate file size
                pixels = dem.shape[0] * dem.shape[1]
                approx_size_mb = (pixels * 4) / (1024 * 1024)  # 4 bytes per float32
                print(f"     Approximate size: {approx_size_mb:.1f} MB")
                print(f"     Pixel resolution: ~{res}m x {res}m")
            else:
                print("✗ No data")
                
        except Exception as e:
            error_msg = str(e)
            if "resolution" in error_msg.lower():
                print(f"✗ Not available at this resolution")
            elif "no data" in error_msg.lower():
                print(f"✗ No data for this area")
            else:
                print(f"✗ Error: {error_msg[:60]}...")
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(available)} resolutions available: {available}")
    
    if available:
        print("\nRecommendations:")
        if 3 in available:
            print("  • 3m: Best quality for detailed terrain analysis")
        if 10 in available:
            print("  • 10m: Good balance of quality and file size")
        if 30 in available:
            print("  • 30m: Fastest downloads, suitable for large areas")

def main():
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python check_resolution_availability.py <south> <north> <west> <east>")
        print("\nExample locations:")
        print("  Park City, UT:     python check_resolution_availability.py 40.65 40.66 -111.51 -111.50")
        print("  Yosemite, CA:      python check_resolution_availability.py 37.7 37.8 -119.6 -119.5")
        print("  Grand Canyon, AZ:  python check_resolution_availability.py 36.0 36.1 -112.2 -112.1")
        print("  Alaska (Denali):   python check_resolution_availability.py 63.0 63.1 -151.1 -151.0")
        sys.exit(1)
    
    bounds = Bounds(
        south=float(sys.argv[1]),
        north=float(sys.argv[2]),
        west=float(sys.argv[3]),
        east=float(sys.argv[4])
    )
    
    check_available_resolutions(bounds)

if __name__ == "__main__":
    main()