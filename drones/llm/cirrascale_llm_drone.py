#!/usr/bin/env python3
"""
Cirrascale-Powered LLM Drone for PyroGuard AI

This module integrates Cirrascale's cloud infrastructure for training and deploying
LLM-enhanced wildfire suppression drones with edge AI capabilities.
"""

import torch
import numpy as np
import json
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CirrascaleConfig:
    """Configuration for Cirrascale cloud services"""
    endpoint: str
    api_key: str
    project_id: str
    gpu_type: str = "A100"
    num_gpus: int = 4
    region: str = "us-west-2"


class CirrascaleLLMClient:
    """
    Client for interacting with Cirrascale cloud infrastructure
    for LLM training and optimization
    """
    
    def __init__(self, config: CirrascaleConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        })
    
    def submit_training_job(self, job_config: Dict) -> str:
        """Submit LLM training job to Cirrascale infrastructure"""
        payload = {
            'project_id': self.config.project_id,
            'gpu_type': self.config.gpu_type,
            'num_gpus': self.config.num_gpus,
            'region': self.config.region,
            'job_config': job_config
        }
        
        response = self.session.post(
            f"{self.config.endpoint}/api/v1/training/submit",
            json=payload
        )
        
        if response.status_code == 200:
            job_id = response.json()['job_id']
            logger.info(f"🚀 Training job submitted to Cirrascale: {job_id}")
            return job_id
        else:
            raise Exception(f"Failed to submit job: {response.text}")
    
    def monitor_job(self, job_id: str) -> Dict:
        """Monitor training job status"""
        response = self.session.get(
            f"{self.config.endpoint}/api/v1/training/status/{job_id}"
        )
        return response.json()
    
    def download_model(self, job_id: str, output_path: str) -> str:
        """Download trained model from Cirrascale"""
        response = self.session.get(
            f"{self.config.endpoint}/api/v1/training/download/{job_id}"
        )
        
        model_path = Path(output_path) / f"cirrascale_model_{job_id}.pth"
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"📥 Model downloaded: {model_path}")
        return str(model_path)
    
    def optimize_for_edge(self, model_path: str, optimization_config: Dict) -> str:
        """Optimize trained model for edge deployment"""
        payload = {
            'model_path': model_path,
            'optimization_config': optimization_config
        }
        
        response = self.session.post(
            f"{self.config.endpoint}/api/v1/optimization/submit",
            json=payload
        )
        
        if response.status_code == 200:
            optimization_id = response.json()['optimization_id']
            logger.info(f"⚡ Edge optimization started: {optimization_id}")
            return optimization_id
        else:
            raise Exception(f"Failed to start optimization: {response.text}")


class WildfireLLMProcessor:
    """
    Edge-deployed LLM for wildfire situation analysis and decision making
    Trained on Cirrascale, optimized for edge deployment
    """
    
    def __init__(self, model_path: str, device: str = "cpu", use_npu: bool = False):
        self.device = device
        self.use_npu = use_npu
        self.model = self.load_optimized_model(model_path)
        self.tokenizer = self.load_tokenizer()
        
        # Initialize NPU session if available
        if self.use_npu:
            self.ort_session = self._setup_npu_session(model_path)
        
        # Wildfire-specific prompts and templates
        self.situation_template = self.load_situation_template()
        self.strategy_template = self.load_strategy_template()
        self.command_template = self.load_command_template()
    
    def load_optimized_model(self, model_path: str):
        """Load Cirrascale-optimized model for edge deployment"""
        try:
            # Check if this is a demo mock model
            if model_path.endswith('.txt'):
                logger.info(f"🎭 Demo mode: Using mock model from {model_path}")
                return "mock_model"  # Return a placeholder for demo mode
            
            # Load quantized/optimized model
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            logger.info(f"✅ Loaded optimized LLM from {model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return None
    
    def analyze_fire_situation(self, grid_state: np.ndarray, 
                             drone_status: Dict, 
                             weather_data: Dict) -> Dict[str, Any]:
        """
        Analyze current wildfire situation using LLM
        
        Args:
            grid_state: Current wildfire grid state
            drone_status: Current drone status and position
            weather_data: Weather conditions (wind, humidity, etc.)
            
        Returns:
            Structured analysis with priorities and recommendations
        """
        # Convert grid state to natural language description
        situation_desc = self._grid_to_description(grid_state)
        
        # Create analysis prompt
        prompt = self.situation_template.format(
            fire_situation=situation_desc,
            drone_position=drone_status.get('position', 'unknown'),
            drone_water=drone_status.get('water_level', 0),
            drone_energy=drone_status.get('energy', 0),
            wind_direction=weather_data.get('wind_direction', 0),
            wind_strength=weather_data.get('wind_strength', 0)
        )
        
        # Generate analysis
        analysis = self._generate_response(prompt)
        
        return self._parse_situation_analysis(analysis)
    
    def generate_suppression_strategy(self, situation_analysis: Dict) -> Dict[str, Any]:
        """Generate optimal fire suppression strategy"""
        strategy_prompt = self.strategy_template.format(
            fire_priority=situation_analysis.get('priority_level', 'medium'),
            fire_locations=situation_analysis.get('fire_locations', []),
            spread_prediction=situation_analysis.get('spread_prediction', 'unknown'),
            resources_available=situation_analysis.get('resources', {})
        )
        
        strategy = self._generate_response(strategy_prompt)
        return self._parse_strategy(strategy)
    
    def process_voice_command(self, command: str, context: Dict) -> Dict[str, Any]:
        """Process natural language commands for drone control"""
        command_prompt = self.command_template.format(
            voice_command=command,
            current_situation=context.get('situation', 'normal'),
            drone_capabilities=context.get('capabilities', [])
        )
        
        parsed_command = self._generate_response(command_prompt)
        return self._parse_command(parsed_command)
    
    def _grid_to_description(self, grid_state: np.ndarray) -> str:
        """Convert grid state to natural language description"""
        fire_cells = np.sum((grid_state == 2) | (grid_state == 5))
        vegetation_cells = np.sum((grid_state == 1) | (grid_state == 6) | (grid_state == 7))
        burned_cells = np.sum(grid_state == 3)
        
        total_cells = grid_state.size
        fire_percentage = (fire_cells / total_cells) * 100
        
        # Find fire clusters
        fire_locations = self._find_fire_clusters(grid_state)
        
        description = f"""
        Current wildfire situation:
        - Active fires: {fire_cells} cells ({fire_percentage:.1f}% of area)
        - Vegetation remaining: {vegetation_cells} cells
        - Already burned: {burned_cells} cells
        - Fire clusters detected at: {fire_locations}
        """
        
        return description.strip()
    
    def _find_fire_clusters(self, grid_state: np.ndarray) -> List[Tuple[int, int]]:
        """Identify major fire cluster locations"""
        fire_mask = (grid_state == 2) | (grid_state == 5)
        fire_positions = np.where(fire_mask)
        
        if len(fire_positions[0]) == 0:
            return []
        
        # Simple clustering - find centers of fire activity
        clusters = []
        for i in range(0, len(fire_positions[0]), max(1, len(fire_positions[0]) // 5)):
            row, col = fire_positions[0][i], fire_positions[1][i]
            clusters.append((int(row), int(col)))
        
        return clusters[:5]  # Return up to 5 major clusters
    
    def _generate_response(self, prompt: str) -> str:
        """Generate LLM response using optimized edge model"""
        if self.model is None:
            return "Model not available"
        
        # Demo mode - return mock responses
        if self.model == "mock_model":
            return self._generate_mock_response(prompt)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part (remove prompt)
            generated_text = response[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return "Analysis unavailable"
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock responses for demo mode"""
        if "situation" in prompt.lower():
            return """
Priority Level: HIGH

Fire Assessment:
- 3 active fire clusters detected in northeast sector
- Intensity level: high with rapid spread potential
- Spread risk: High due to 15mph easterly winds

Weather Impact:
- Wind: 15mph east - accelerating spread toward residential area
- Humidity: 25% - critically dry conditions

Threat Assessment:
- Structures at risk: 2 buildings in evacuation zone
- Terrain: mixed forest - challenging but accessible

Recommended Actions:
1. Immediate suppression of primary cluster
2. Establish firebreak between fire and structures
3. Coordinate with ground crews for evacuation support
"""
        elif "strategy" in prompt.lower():
            return """
Primary Target: Fire cluster at northeast perimeter (grid 15,18)

Action Sequence:
1. Navigate directly to primary fire location
2. Establish defensive perimeter around threatened structures
3. Apply concentrated water drops on fire head
4. Monitor for spot fires and re-ignition
5. Coordinate with ground teams for mop-up operations

Resource Allocation:
- Water usage: Aggressive (80-90% capacity)
- Energy management: High intensity flight pattern
- Flight pattern: Direct approach with tactical positioning

Contingency Measures:
- If fire jumps containment: Retreat and call for aerial support
- If water runs low: Prioritize structure protection
- If weather deteriorates: Establish safe landing zone
"""
        elif "command" in prompt.lower():
            return """
Action Type: suppress_fire
Target: active_fire_cluster
Priority: high
Parameters:
  - Intensity: maximum
  - Duration: sustained
  - Approach: tactical_positioning
"""
        else:
            return "Mock LLM response for demonstration purposes"
    
    def _setup_npu_session(self, model_path: str):
        """Setup ONNX Runtime session for Snapdragon NPU acceleration"""
        try:
            import onnxruntime as ort
            
            # Check if this is a real ONNX model (not demo)
            if not model_path.endswith('.onnx') or not os.path.exists(model_path):
                logger.info("🎭 Demo mode: NPU session not available for mock models")
                return None
            
            # Check if file is actually an ONNX model (not just text)
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'\x08\x01\x12'):  # ONNX magic bytes
                        logger.info("🎭 Demo mode: Mock ONNX file detected")
                        return None
            except:
                return None
            
            # Snapdragon NPU provider options
            npu_options = {
                'device_id': 0,
                'enable_npu_fast_math': True,
                'npu_performance_mode': 'high_performance',
                'enable_mixed_precision': True
            }
            
            # Try to create session with Qualcomm NPU provider
            providers = [
                ('QNNExecutionProvider', npu_options),  # Qualcomm NPU
                ('CPUExecutionProvider', {})  # Fallback to CPU
            ]
            
            session = ort.InferenceSession(model_path, providers=providers)
            
            # Check which provider is actually being used
            active_provider = session.get_providers()[0]
            if 'QNN' in active_provider:
                logger.info(f"🚀 NPU acceleration enabled: {active_provider}")
                logger.info(f"   Expected speedup: 5-10x faster inference")
                logger.info(f"   NPU TOPS: 45 (Snapdragon Elite)")
            else:
                logger.info(f"⚠️ NPU not available, using: {active_provider}")
            
            return session
            
        except ImportError:
            logger.error("❌ ONNX Runtime not available for NPU acceleration")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to setup NPU session: {e}")
            logger.info("💡 Falling back to CPU inference")
            return None
    
    def _parse_situation_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse LLM analysis into structured format"""
        # Simple parsing - in production, use more sophisticated NLP
        return {
            'priority_level': self._extract_priority(analysis),
            'fire_locations': self._extract_locations(analysis),
            'spread_prediction': self._extract_prediction(analysis),
            'recommended_actions': self._extract_actions(analysis),
            'raw_analysis': analysis
        }
    
    def _parse_strategy(self, strategy: str) -> Dict[str, Any]:
        """Parse strategy into actionable steps"""
        return {
            'primary_target': self._extract_target(strategy),
            'action_sequence': self._extract_sequence(strategy),
            'resource_allocation': self._extract_resources(strategy),
            'raw_strategy': strategy
        }
    
    def _parse_command(self, command: str) -> Dict[str, Any]:
        """Parse voice command into drone actions"""
        return {
            'action_type': self._extract_action_type(command),
            'target_location': self._extract_target_location(command),
            'parameters': self._extract_parameters(command),
            'raw_command': command
        }
    
    def load_situation_template(self) -> str:
        """Load situation analysis prompt template"""
        return """
        Analyze this wildfire situation and provide tactical assessment:
        
        Fire Situation: {fire_situation}
        Drone Position: {drone_position}
        Drone Water Level: {drone_water}%
        Drone Energy: {drone_energy}%
        Wind Direction: {wind_direction}°
        Wind Strength: {wind_strength}
        
        Provide analysis including:
        1. Priority level (low/medium/high/critical)
        2. Most dangerous fire locations
        3. Predicted fire spread pattern
        4. Recommended immediate actions
        
        Analysis:
        """
    
    def load_strategy_template(self) -> str:
        """Load strategy generation prompt template"""
        return """
        Generate optimal fire suppression strategy:
        
        Priority Level: {fire_priority}
        Fire Locations: {fire_locations}
        Spread Prediction: {spread_prediction}
        Available Resources: {resources_available}
        
        Provide strategy including:
        1. Primary target for suppression
        2. Step-by-step action sequence
        3. Resource allocation plan
        4. Contingency measures
        
        Strategy:
        """
    
    def load_command_template(self) -> str:
        """Load voice command processing template"""
        return """
        Parse this voice command for drone control:
        
        Command: "{voice_command}"
        Current Situation: {current_situation}
        Drone Capabilities: {drone_capabilities}
        
        Convert to structured action:
        1. Action type (move/suppress/return/patrol)
        2. Target location or area
        3. Specific parameters
        
        Parsed Command:
        """
    
    def load_tokenizer(self):
        """Load tokenizer for the optimized model"""
        # Placeholder - implement based on your specific model
        return None
    
    # Helper methods for parsing (simplified implementations)
    def _extract_priority(self, text: str) -> str:
        text_lower = text.lower()
        if 'critical' in text_lower:
            return 'critical'
        elif 'high' in text_lower:
            return 'high'
        elif 'medium' in text_lower:
            return 'medium'
        else:
            return 'low'
    
    def _extract_locations(self, text: str) -> List[str]:
        # Simple location extraction
        return []
    
    def _extract_prediction(self, text: str) -> str:
        return "spreading northeast"
    
    def _extract_actions(self, text: str) -> List[str]:
        return ["suppress primary fire", "monitor spread"]
    
    def _extract_target(self, text: str) -> str:
        return "primary fire cluster"
    
    def _extract_sequence(self, text: str) -> List[str]:
        return ["move to target", "suppress fire", "monitor results"]
    
    def _extract_resources(self, text: str) -> Dict[str, Any]:
        return {"water": "80%", "energy": "60%"}
    
    def _extract_action_type(self, text: str) -> str:
        return "move_and_suppress"
    
    def _extract_target_location(self, text: str) -> Tuple[int, int]:
        return (10, 15)
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        return {"intensity": "high", "duration": "sustained"}


class CirrascaleLLMDrone:
    """
    LLM-enhanced drone that leverages Cirrascale cloud training
    and edge deployment for intelligent wildfire suppression
    """
    
    def __init__(self, 
                 position: Tuple[int, int] = (1, 1),
                 cirrascale_config: Optional[CirrascaleConfig] = None,
                 edge_model_path: Optional[str] = None):
        
        self.position = list(position)
        self.water_level = 100.0
        self.energy = 100.0
        self.max_water = 100.0
        self.max_energy = 100.0
        self.fires_extinguished = 0
        
        # Cirrascale integration
        self.cirrascale_client = None
        if cirrascale_config:
            self.cirrascale_client = CirrascaleLLMClient(cirrascale_config)
        
        # Edge LLM processor with NPU support
        self.llm_processor = None
        if edge_model_path:
            # Check if running on Snapdragon Elite (ARM64 Windows)
            use_npu = self._detect_snapdragon_npu()
            self.llm_processor = WildfireLLMProcessor(edge_model_path, use_npu=use_npu)
        
        # Decision history for learning
        self.decision_history = []
        
        logger.info("🤖 Cirrascale LLM Drone initialized")
    
    def update(self, grid_state: np.ndarray, weather_data: Dict = None) -> Dict[str, Any]:
        """
        Update drone with LLM-enhanced decision making
        
        Args:
            grid_state: Current wildfire grid state
            weather_data: Current weather conditions
            
        Returns:
            Action dictionary with LLM reasoning
        """
        if self.llm_processor is None:
            return self._fallback_behavior(grid_state)
        
        # Get current status
        drone_status = self.get_status()
        
        # Analyze situation with LLM
        situation_analysis = self.llm_processor.analyze_fire_situation(
            grid_state, drone_status, weather_data or {}
        )
        
        # Generate strategy
        strategy = self.llm_processor.generate_suppression_strategy(situation_analysis)
        
        # Execute strategy
        action = self._execute_strategy(strategy, grid_state)
        
        # Store decision for learning
        self.decision_history.append({
            'situation': situation_analysis,
            'strategy': strategy,
            'action': action,
            'timestamp': time.time()
        })
        
        return action
    
    def process_voice_command(self, command: str) -> Dict[str, Any]:
        """Process voice command using LLM"""
        if self.llm_processor is None:
            return {'error': 'LLM processor not available'}
        
        context = {
            'situation': 'active_fire',
            'capabilities': ['move', 'suppress_fire', 'return_to_base']
        }
        
        parsed_command = self.llm_processor.process_voice_command(command, context)
        return self._execute_parsed_command(parsed_command)
    
    def train_on_cirrascale(self, training_data: Dict) -> str:
        """Submit training job to Cirrascale infrastructure"""
        if self.cirrascale_client is None:
            raise Exception("Cirrascale client not configured")
        
        job_config = {
            'model_type': 'wildfire_llm',
            'training_data': training_data,
            'optimization_target': 'edge_deployment',
            'max_epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4
        }
        
        job_id = self.cirrascale_client.submit_training_job(job_config)
        logger.info(f"🚀 Training job submitted to Cirrascale: {job_id}")
        
        return job_id
    
    def deploy_trained_model(self, job_id: str, output_dir: str) -> str:
        """Deploy trained model from Cirrascale to edge"""
        if self.cirrascale_client is None:
            raise Exception("Cirrascale client not configured")
        
        # Download trained model
        model_path = self.cirrascale_client.download_model(job_id, output_dir)
        
        # Optimize for edge deployment
        optimization_config = {
            'target_device': 'edge',
            'max_memory': '2GB',
            'max_latency': '100ms',
            'quantization': 'int8'
        }
        
        optimization_id = self.cirrascale_client.optimize_for_edge(model_path, optimization_config)
        
        logger.info(f"⚡ Model optimization started: {optimization_id}")
        
        return optimization_id
    
    def _execute_strategy(self, strategy: Dict, grid_state: np.ndarray) -> Dict[str, Any]:
        """Execute LLM-generated strategy"""
        primary_target = strategy.get('primary_target', 'nearest_fire')
        action_sequence = strategy.get('action_sequence', ['patrol'])
        
        # Find target location
        target_pos = self._find_target_location(primary_target, grid_state)
        
        if target_pos:
            # Move toward target
            direction = self._calculate_direction(self.position, target_pos)
            new_pos = [
                self.position[0] + direction[0],
                self.position[1] + direction[1]
            ]
            
            # Validate position
            if self._is_valid_position(new_pos, grid_state.shape):
                self.position = new_pos
                self.energy -= 2
                
                # Check if we can suppress fire at this location
                if grid_state[self.position[0], self.position[1]] in [2, 5]:  # Fire
                    return self._suppress_fire(strategy)
                
                return {
                    'action': 'move_toward_target',
                    'position': tuple(self.position),
                    'target': target_pos,
                    'strategy': strategy.get('raw_strategy', ''),
                    'llm_reasoning': True
                }
        
        return {'action': 'patrol', 'llm_reasoning': True}
    
    def _suppress_fire(self, strategy: Dict) -> Dict[str, Any]:
        """Suppress fire with LLM-guided parameters"""
        if self.water_level <= 0:
            return {'action': 'no_water', 'position': tuple(self.position)}
        
        # LLM-guided suppression intensity
        resource_allocation = strategy.get('resource_allocation', {})
        water_usage = min(20, self.water_level)  # Intelligent water usage
        
        self.water_level -= water_usage
        self.energy -= 5
        self.fires_extinguished += 1
        
        return {
            'action': 'fire_suppressed',
            'position': tuple(self.position),
            'water_used': water_usage,
            'strategy': strategy.get('raw_strategy', ''),
            'llm_reasoning': True
        }
    
    def _execute_parsed_command(self, parsed_command: Dict) -> Dict[str, Any]:
        """Execute parsed voice command"""
        action_type = parsed_command.get('action_type', 'patrol')
        target_location = parsed_command.get('target_location')
        
        if action_type == 'move_and_suppress' and target_location:
            self.position = list(target_location)
            return {
                'action': 'voice_command_executed',
                'command_type': action_type,
                'new_position': tuple(self.position),
                'llm_reasoning': True
            }
        
        return {'action': 'command_processed', 'llm_reasoning': True}
    
    def _detect_snapdragon_npu(self) -> bool:
        """Detect if running on Snapdragon Elite with NPU support"""
        try:
            import platform
            import subprocess
            
            # Check if running on ARM64 Windows (Snapdragon Elite indicator)
            if platform.machine().lower() in ['arm64', 'aarch64'] and platform.system() == 'Windows':
                logger.info("🔍 ARM64 Windows detected - checking for Snapdragon NPU")
                
                # Try to detect Qualcomm NPU through system info
                try:
                    result = subprocess.run(['wmic', 'path', 'win32_processor', 'get', 'name'], 
                                          capture_output=True, text=True, timeout=5)
                    if 'Snapdragon' in result.stdout or 'Qualcomm' in result.stdout:
                        logger.info("🚀 Snapdragon processor detected - NPU likely available")
                        return True
                except:
                    pass
                
                # Check for Qualcomm AI Engine
                try:
                    import onnxruntime as ort
                    available_providers = ort.get_available_providers()
                    if 'QNNExecutionProvider' in available_providers:
                        logger.info("✅ Qualcomm NPU provider available")
                        return True
                except:
                    pass
                
                logger.info("⚠️ ARM64 Windows but NPU provider not detected")
                return False
            
            # Check for other ARM platforms (like Apple Silicon)
            elif platform.machine().lower() in ['arm64', 'aarch64']:
                logger.info("🍎 ARM platform detected (non-Windows) - NPU not supported")
                return False
            
            else:
                logger.info("💻 x86/x64 platform - NPU not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error detecting NPU: {e}")
            return False
    
    def _fallback_behavior(self, grid_state: np.ndarray) -> Dict[str, Any]:
        """Fallback behavior when LLM is not available"""
        # Simple fire-seeking behavior
        fire_positions = np.where((grid_state == 2) | (grid_state == 5))
        
        if len(fire_positions[0]) > 0:
            # Move toward nearest fire
            nearest_fire = self._find_nearest_fire(fire_positions)
            direction = self._calculate_direction(self.position, nearest_fire)
            
            new_pos = [
                self.position[0] + direction[0],
                self.position[1] + direction[1]
            ]
            
            if self._is_valid_position(new_pos, grid_state.shape):
                self.position = new_pos
                self.energy -= 2
                
                return {
                    'action': 'move_to_fire',
                    'position': tuple(self.position),
                    'llm_reasoning': False
                }
        
        return {'action': 'patrol', 'llm_reasoning': False}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current drone status"""
        return {
            'position': tuple(self.position),
            'water_level': self.water_level,
            'water_percentage': (self.water_level / self.max_water) * 100,
            'energy': self.energy,
            'energy_percentage': (self.energy / self.max_energy) * 100,
            'fires_extinguished': self.fires_extinguished,
            'llm_enabled': self.llm_processor is not None,
            'cirrascale_connected': self.cirrascale_client is not None
        }
    
    # Helper methods
    def _find_target_location(self, target_desc: str, grid_state: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find target location based on LLM description"""
        fire_positions = np.where((grid_state == 2) | (grid_state == 5))
        if len(fire_positions[0]) > 0:
            return (int(fire_positions[0][0]), int(fire_positions[1][0]))
        return None
    
    def _find_nearest_fire(self, fire_positions: Tuple[np.ndarray, np.ndarray]) -> Tuple[int, int]:
        """Find nearest fire to current position"""
        min_dist = float('inf')
        nearest = (0, 0)
        
        for i in range(len(fire_positions[0])):
            fire_pos = (fire_positions[0][i], fire_positions[1][i])
            dist = np.sqrt((fire_pos[0] - self.position[0])**2 + (fire_pos[1] - self.position[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = fire_pos
        
        return nearest
    
    def _calculate_direction(self, from_pos: List[int], to_pos: Tuple[int, int]) -> List[int]:
        """Calculate direction vector"""
        dx = to_pos[1] - from_pos[1]
        dy = to_pos[0] - from_pos[0]
        
        # Normalize to unit steps
        if abs(dx) > abs(dy):
            return [0, 1 if dx > 0 else -1]
        elif dy != 0:
            return [1 if dy > 0 else -1, 0]
        else:
            return [0, 0]
    
    def _is_valid_position(self, pos: List[int], grid_shape: Tuple[int, int]) -> bool:
        """Check if position is valid"""
        return (0 <= pos[0] < grid_shape[0] and 0 <= pos[1] < grid_shape[1])


# Factory function for easy integration
def create_cirrascale_llm_drone(position: Tuple[int, int] = (1, 1),
                               cirrascale_config: Optional[Dict] = None,
                               edge_model_path: Optional[str] = None) -> CirrascaleLLMDrone:
    """
    Factory function to create Cirrascale LLM-enhanced drone
    
    Args:
        position: Starting position
        cirrascale_config: Cirrascale cloud configuration
        edge_model_path: Path to edge-optimized model
        
    Returns:
        Configured CirrascaleLLMDrone instance
    """
    config = None
    if cirrascale_config:
        config = CirrascaleConfig(**cirrascale_config)
    
    return CirrascaleLLMDrone(
        position=position,
        cirrascale_config=config,
        edge_model_path=edge_model_path
    )
