#!/usr/bin/env python3
"""
PyroGuard AI - Cirrascale LLM Demo Runner

Easy-to-use script for running PyroGuard AI with Cirrascale-powered LLM drones.

Usage:
    # Run with mock LLM (for demonstration)
    python run_cirrascale_demo.py --demo-mode
    
    # Run with actual Cirrascale connection
    python run_cirrascale_demo.py --cirrascale-endpoint https://api.cirrascale.com \
                                  --api-key YOUR_API_KEY \
                                  --llm-model models/llm/wildfire_phi3_optimized.onnx
"""

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description='PyroGuard AI - Cirrascale LLM Demo')
    parser.add_argument('--demo-mode', action='store_true',
                       help='Run in demo mode with mock LLM capabilities')
    parser.add_argument('--cirrascale-endpoint', type=str,
                       help='Cirrascale cloud endpoint')
    parser.add_argument('--api-key', type=str,
                       help='Cirrascale API key')
    parser.add_argument('--llm-model', type=str,
                       help='Path to edge-optimized LLM model')
    parser.add_argument('--grid-size', type=int, default=25,
                       help='Simulation grid size')
    parser.add_argument('--train-first', action='store_true',
                       help='Train LLM on Cirrascale before running demo')
    
    args = parser.parse_args()
    
    print("🔥 PyroGuard AI - Cirrascale LLM Integration")
    print("=" * 50)
    
    if args.train_first:
        print("🚀 Training LLM on Cirrascale first...")
        
        if not args.cirrascale_endpoint or not args.api_key:
            print("❌ Cirrascale endpoint and API key required for training")
            return
        
        # Run training script
        train_cmd = [
            sys.executable, 'training/train_llm_cirrascale.py',
            '--cirrascale-endpoint', args.cirrascale_endpoint,
            '--api-key', args.api_key,
            '--generate-data',
            '--train-all',
            '--deploy-models'
        ]
        
        print(f"Running: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)
        
        # Set model path to newly trained model
        args.llm_model = 'models/llm/situation_analyzer_optimized.onnx'
    
    # Prepare demo command
    demo_cmd = [
        sys.executable, 'demos/demo_integrated.py',
        '--drone-type', 'llm',
        '--grid-size', str(args.grid_size)
    ]
    
    if args.demo_mode:
        print("🎭 Running in demo mode (mock LLM capabilities)")
        # Create a mock model file for demo
        os.makedirs('models/llm', exist_ok=True)
        mock_model_path = 'models/llm/demo_mock_model.txt'
        with open(mock_model_path, 'w') as f:
            f.write("Mock LLM model for demonstration purposes")
        demo_cmd.extend(['--llm-model-path', mock_model_path])
    else:
        if args.cirrascale_endpoint:
            demo_cmd.extend(['--cirrascale-endpoint', args.cirrascale_endpoint])
        if args.api_key:
            demo_cmd.extend(['--cirrascale-api-key', args.api_key])
        if args.llm_model:
            demo_cmd.extend(['--llm-model-path', args.llm_model])
    
    print(f"🎮 Starting PyroGuard AI Demo...")
    print(f"Command: {' '.join(demo_cmd)}")
    print("\n🎯 Features Available:")
    print("  - LLM-powered situation analysis")
    print("  - Natural language drone commands")
    print("  - Intelligent fire suppression strategies")
    print("  - Edge AI deployment (no cloud dependency)")
    if not args.demo_mode:
        print("  - Cirrascale cloud training integration")
    
    # Run the demo using subprocess to handle paths with spaces
    subprocess.run(demo_cmd, check=True)


if __name__ == "__main__":
    main()
