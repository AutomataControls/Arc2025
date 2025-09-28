#!/usr/bin/env python3
"""
Analysis of shape mismatch failures from evaluation log
This will help us identify new rules for GridSizePredictorV3
"""

# Shape mismatches from the evaluation log
shape_mismatches = [
    # Task ID, Predicted Shape, Actual Shape, Input Shape
    ("42f83767", (20, 24), (15, 15), None),  # Predicted larger, actual is square
    ("ce602527", (5, 5), (5, 3), None),       # Width was wrong
    ("77fdfe62", (4, 4), (2, 2), None),       # Predicted 2x larger
    ("652646ff", (20, 20), (18, 6), None),    # Completely wrong aspect ratio
    ("abbfd121", (10, 10), (7, 7), None),     # Off by 3 in both dimensions
    ("b94a9452", (4, 3), (3, 3), None),       # Close but not square
    ("be94b721", (3, 3), (4, 3), None),       # Height off by 1
    ("469497ad", (10, 10), (15, 15), None),   # 1.5x scaling missed
    ("e1baa8a4", (2, 2), (2, 3), None),       # Width off by 1
    ("2037f2c7", (25, 25), (4, 8), None),     # Massive overestimation
    ("8abad3cf", (3, 6), (4, 10), None),      # Both dimensions wrong
    ("d749d46f", (4, 13), (10, 18), None),    # Scaling relationship missed
    ("d631b094", (3, 3), (1, 1), None),       # Predicted 3x3, actual 1x1
    ("b91ae062", (6, 6), (9, 9), None),       # 1.5x scaling
    ("bbc9ae5d", (1, 6), (3, 6), None),       # Height wrong, width correct
]

# Analysis patterns
patterns_found = {
    "1.5x_scaling": ["469497ad", "b91ae062"],  # Output is 1.5x input
    "massive_reduction": ["2037f2c7", "d631b094"],  # Output much smaller than input
    "off_by_one": ["be94b721", "e1baa8a4"],  # Single dimension off by 1
    "aspect_ratio_change": ["652646ff", "2037f2c7", "8abad3cf"],  # Major aspect change
    "near_square": ["b94a9452", "ce602527"],  # Almost square but not quite
    "dimension_preserved": ["bbc9ae5d"],  # One dimension stays same
}

# New rules to implement
new_rules = [
    {
        "name": "fractional_scaling",
        "description": "Check for 1.5x, 2.5x, 0.5x scaling factors",
        "tasks": ["469497ad", "b91ae062"]
    },
    {
        "name": "extreme_reduction", 
        "description": "Output can be tiny (1x1) compared to input",
        "tasks": ["d631b094"]
    },
    {
        "name": "off_by_one_adjustment",
        "description": "Try ±1 adjustments when close to a pattern",
        "tasks": ["be94b721", "e1baa8a4", "b94a9452"]
    },
    {
        "name": "single_dimension_preservation",
        "description": "One dimension stays same, other changes",
        "tasks": ["bbc9ae5d", "ce602527"]
    },
    {
        "name": "content_density_rule",
        "description": "Output size based on density of non-background pixels",
        "tasks": ["2037f2c7", "652646ff"]
    }
]

print("SHAPE MISMATCH ANALYSIS")
print("=" * 50)
print(f"\nTotal shape mismatches: {len(shape_mismatches)}")
print(f"\nPattern categories identified: {len(patterns_found)}")

for pattern, tasks in patterns_found.items():
    print(f"\n{pattern}: {len(tasks)} tasks")
    
print(f"\n\nNew rules to implement: {len(new_rules)}")
for rule in new_rules:
    print(f"\n- {rule['name']}")
    print(f"  {rule['description']}")
    print(f"  Examples: {', '.join(rule['tasks'])}")