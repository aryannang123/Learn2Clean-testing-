#!/usr/bin/env python3
"""
Quick test to verify Learn2Clean setup is working
Run: python quick_test.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import pandas as pd
        import numpy as np
        from stable_baselines3 import PPO
        import gymnasium as gym
        import hydra
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: poetry install")
        return False

def test_data():
    """Test if datasets are available"""
    data_files = [
        "data/titanic.csv",
        "data/crimes.csv", 
        "data/crime_incidents_messy.csv"
    ]
    
    all_good = True
    for file in data_files:
        if Path(file).exists():
            print(f"✅ Dataset found: {file}")
        else:
            print(f"❌ Dataset missing: {file}")
            all_good = False
    
    return all_good

def test_learn2clean():
    """Test if Learn2Clean modules can be imported"""
    try:
        from learn2clean.loaders.csv_loader import CSVLoader
        from learn2clean.actions.dummy_add import DummyAdd
        print("✅ Learn2Clean modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Learn2Clean import error: {e}")
        return False

def main():
    print("🤖 Learn2Clean Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package imports", test_imports),
        ("Dataset availability", test_data), 
        ("Learn2Clean modules", test_learn2clean)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Learn2Clean is ready to use.")
        print("\nRun your first experiment:")
        print("poetry run python experiments/tutorials/01_titanic_csv_dummy.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("See SETUP.md for installation instructions.")

if __name__ == "__main__":
    main()