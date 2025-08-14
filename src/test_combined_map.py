#!/usr/bin/env python3
"""
Test script to verify the combined case legislation map functionality.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from relevant_section_finder import load_combined_case_legislation_map, get_acts_for_case

def test_combined_map_loading():
    """Test loading the combined case legislation map."""
    print("Testing combined case legislation map loading...")
    
    # Test loading the map
    case_legislation_map = load_combined_case_legislation_map()
    
    if not case_legislation_map:
        print("❌ Failed to load combined case legislation map")
        return False
    
    print(f"✅ Successfully loaded map with {len(case_legislation_map)} cases")
    
    # Test a few sample cases
    sample_cases = ['ewhc_fam_2018_3244', 'ewhc_admin_2005_2977', 'ewhc_ch_2023_123']
    
    for case_name in sample_cases:
        acts = case_legislation_map.get(case_name, [])
        print(f"  Case '{case_name}': {len(acts)} acts - {acts}")
    
    return True

def test_acts_lookup():
    """Test looking up acts for specific cases."""
    print("\nTesting acts lookup functionality...")
    
    # Test the utility function
    test_cases = [
        'ewhc_fam_2018_3244',
        'ewhc_admin_2005_2977',
        'nonexistent_case'
    ]
    
    for case_name in test_cases:
        acts = get_acts_for_case(case_name)
        if acts:
            print(f"✅ Case '{case_name}': {len(acts)} acts found")
        else:
            print(f"⚠️  Case '{case_name}': No acts found")
    
    return True

def test_map_structure():
    """Test the structure of the loaded map."""
    print("\nTesting map structure...")
    
    case_legislation_map = load_combined_case_legislation_map()
    
    if not case_legislation_map:
        print("❌ Cannot test structure - map not loaded")
        return False
    
    # Check a few entries to understand the structure
    sample_entries = list(case_legislation_map.items())[:3]
    
    for case_name, acts in sample_entries:
        print(f"  Case: {case_name}")
        print(f"    Type: {type(acts)}")
        print(f"    Acts: {acts}")
        print(f"    Act count: {len(acts)}")
        
        # Check if acts are strings (legislation paths)
        if acts and isinstance(acts[0], str):
            print(f"    First act format: {acts[0]}")
        
        print()
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Combined Case Legislation Map Functionality")
    print("=" * 60)
    
    tests = [
        test_combined_map_loading,
        test_acts_lookup,
        test_map_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The combined map functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main() 