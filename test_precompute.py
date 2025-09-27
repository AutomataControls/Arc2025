#!/usr/bin/env python3
"""Test the precompute_patterns script with a small subset of data"""

import json
from pathlib import Path
from precompute_patterns import HailoPatternAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load just a few tasks for testing
data_dir = Path('/mnt/d/opt/ARCPrize2025/data')
training_path = data_dir / 'arc-agi_training_challenges.json'

with open(training_path, 'r') as f:
    all_tasks = json.load(f)

# Get first 5 tasks
test_tasks = dict(list(all_tasks.items())[:5])
logger.info(f"Testing with {len(test_tasks)} tasks")

# Initialize analyzer
analyzer = HailoPatternAnalyzer(hailo_device_id=0)

# Analyze each task
for task_id, task_data in test_tasks.items():
    logger.info(f"\nAnalyzing task: {task_id}")
    result = analyzer.analyze_task(task_id, task_data)
    
    # Show detected patterns
    for detector_name, detection in result['patterns'].items():
        if detection.get('confidence', 0) > 0:
            logger.info(f"  {detector_name}: {detection}")
    
    # Show transformations
    logger.info(f"  Size change: {result['transformations']['size_change']['rule']}")
    logger.info(f"  Color mapping: {result['transformations']['color_mapping']['consistent_map']}")

logger.info("\nTest complete!")