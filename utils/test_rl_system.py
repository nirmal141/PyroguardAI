#!/usr/bin/env python3
"""
Test script for RL Drone System

This script runs quick tests to verify that all components work together.
"""

import sys
import os
import numpy as np
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        print(f"   ❌ PyTorch import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        print(f"   ✅ Gymnasium")
    except Exception as e:
        print(f"   ❌ Gymnasium import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"   ✅ Matplotlib")
    except Exception as e:
        print(f"   ❌ Matplotlib import failed: {e}")
        return False
    
    try:
        from drones.rl.rl_environment import RLWildfireEnvironment
        print("   ✅ RL Wildfire Environment")
    except Exception as e:
        print(f"   ❌ RL Environment import failed: {e}")
        return False
    
    try:
        from drones.rl.rl_agent import RLFirefighterDrone, create_rl_drone
        print("   ✅ RL Firefighter Drone")
    except Exception as e:
        print(f"   ❌ RL Drone import failed: {e}")
        return False
    
    print("   🎉 All imports successful!")
    return True


def test_rl_environment():
    """Test RL environment creation and basic functionality."""
    print("\n🌲 Testing RL Environment...")
    
    try:
        from drones.rl.rl_environment import RLWildfireEnvironment
        
        # Create environment
        env = RLWildfireEnvironment(grid_size=10, max_episode_steps=50)
        print("   ✅ Environment created")
        
        # Test reset
        observation, info = env.reset()
        print("   ✅ Environment reset")
        print(f"   📊 Observation keys: {list(observation.keys())}")
        print(f"   📊 Action space: {env.action_space}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("   ✅ Environment step")
        print(f"   🎯 Sample reward: {reward:.2f}")
        
        env.close()
        print("   🎉 RL Environment test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ RL Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_rl_drone():
    """Test RL drone creation and basic functionality."""
    print("\n🤖 Testing RL Drone...")
    
    try:
        from rl_wildfire_env import RLWildfireEnvironment
        from drones.rl.rl_agent import create_rl_drone
        
        # Create environment and drone
        env = RLWildfireEnvironment(grid_size=10, max_episode_steps=50)
        drone = create_rl_drone(env.observation_space, env.action_space)
        print("   ✅ RL Drone created")
        
        # Test action selection
        observation, _ = env.reset()
        action = drone.select_action(observation, training=False)
        print(f"   ✅ Action selected: {action}")
        
        # Test training step
        next_obs, reward, done, truncated, info = env.step(action)
        training_stats = drone.train_step(observation, action, reward, next_obs, done)
        print(f"   ✅ Training step completed")
        print(f"   📈 Training stats keys: {list(training_stats.keys())}")
        
        # Test statistics
        stats = drone.get_statistics()
        print(f"   ✅ Statistics retrieved: {len(stats)} metrics")
        
        env.close()
        print("   🎉 RL Drone test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ RL Drone test failed: {e}")
        traceback.print_exc()
        return False


def test_integrated_demo():
    """Test the integrated demo without visualization."""
    print("\n🎮 Testing Integrated Demo...")
    
    try:
        from demo_rl import RLIntegratedWildfireEnvironment
        
        # Test simple drone mode
        env_simple = RLIntegratedWildfireEnvironment(
            grid_size=10,
            drone_type="simple"
        )
        print("   ✅ Simple drone environment created")
        
        # Test a few steps
        state = env_simple.reset()
        for i in range(5):
            state = env_simple.step()
        print("   ✅ Simple drone simulation steps completed")
        
        # Test RL drone mode (without model)
        env_rl = RLIntegratedWildfireEnvironment(
            grid_size=10,
            drone_type="rl",
            rl_model_path="nonexistent.pth"  # Should fall back to simple
        )
        print("   ✅ RL drone environment created (fallback to simple)")
        
        env_simple.close()
        env_rl.close()
        print("   🎉 Integrated demo test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated demo test failed: {e}")
        traceback.print_exc()
        return False


def test_short_training():
    """Test a very short training run."""
    print("\n🏋️ Testing Short Training Run...")
    
    try:
        from rl_wildfire_env import RLWildfireEnvironment
        from drones.rl.rl_agent import create_rl_drone
        
        # Create environment and drone
        env = RLWildfireEnvironment(grid_size=8, max_episode_steps=20)
        drone = create_rl_drone(env.observation_space, env.action_space)
        
        print("   🚀 Starting mini training run (5 episodes)...")
        
        for episode in range(5):
            observation, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = drone.select_action(observation, training=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Train
                drone.train_step(observation, action, reward, next_observation, 
                               terminated or truncated)
                
                observation = next_observation
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            drone.end_episode(episode_reward, episode_length, 
                            info.get('is_successful', False))
            
            if episode % 2 == 0:
                print(f"     Episode {episode}: reward={episode_reward:.2f}, "
                      f"length={episode_length}, epsilon={drone.epsilon:.3f}")
        
        # Check final statistics
        stats = drone.get_statistics()
        print(f"   📊 Final stats: {stats['episodes_trained']} episodes, "
              f"buffer size: {stats['buffer_size']}")
        
        env.close()
        print("   🎉 Short training test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Short training test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🔥 PyroGuard AI - RL Drone System Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("RL Environment", test_rl_environment),
        ("RL Drone", test_rl_drone),
        ("Integrated Demo", test_integrated_demo),
        ("Short Training", test_short_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your RL drone system is ready!")
        print("\n📖 Next steps:")
        print("   1. Run training: python train_rl_drone.py --episodes 1000")
        print("   2. Test trained model: python demo_rl.py --drone-type rl --model-path path/to/model.pth")
        print("   3. Compare with simple drone: python demo_rl.py --drone-type simple")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)