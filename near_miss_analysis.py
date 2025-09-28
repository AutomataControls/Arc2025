#!/usr/bin/env python3
"""
Near-miss analysis from evaluation log
Identify patterns where models were close but not exact
"""

# Extract pixel accuracy data from evaluation log
# Format: (task_id, pixel_accuracy, had_shape_mismatch)
pixel_accuracies = [
    ("6ecd11f4", 33.3, False),
    ("817e6c09", 83.7, False),
    ("bf699163", 0.0, False),
    ("db93a21d", 38.2, False),
    ("a87f7484", 22.2, False),
    ("782b5218", 52.0, False),
    ("1fad071e", 20.0, False),
    ("ce9e57f2", 70.8, False),
    ("1a2e2828", 0.0, False),
    ("42f83767", 0.0, True),   # Shape mismatch
    ("ce602527", 0.0, True),   # Shape mismatch
    ("4e45f183", 38.2, False),
    ("e133d23d", 11.1, False),
    ("a3f84088", 51.6, False),
    ("22a4bbc2", 33.8, False),
    ("1b59e163", 4.3, False),
    ("25094a63", 8.3, False),
    ("cfb2ce5a", 40.0, False),
    ("b7249182", 83.1, False),  # Very close!
    ("11e1fe23", 94.7, False),  # Very close!
    ("ae58858e", 63.9, False),
    ("1acc24af", 77.1, False),  # Close!
    ("252143c9", 75.2, False),  # Close!
    ("84ba50d3", 35.8, False),
    ("77fdfe62", 0.0, True),    # Shape mismatch
    ("2546ccf6", 58.8, False),
    ("695367ec", 42.7, False),
    ("652646ff", 0.0, True),    # Shape mismatch
    ("f8cc533f", 4.4, False),
    ("aee291af", 25.0, False),
    ("1190bc91", 29.0, False),
    ("963f59bc", 91.0, False),  # Very close!
    ("ac0c5833", 93.3, False),  # Very close!
    ("abbfd121", 0.0, True),    # Shape mismatch
    ("ac3e2b04", 81.2, False),  # Close!
    ("2dc579da", 11.1, False),
    ("9720b24f", 55.6, False),
    ("e9c9d9a1", 35.0, False),
    ("e8dc4411", 6.2, False),
    ("984d8a3e", 27.0, False),
    ("5587a8d0", 0.0, False),
    ("b60334d2", 60.5, False),
    ("99306f82", 66.7, False),
    ("b94a9452", 0.0, True),    # Shape mismatch
    ("dc433765", 95.9, False),  # Very close!
    ("17cae0c1", 0.0, False),
    ("be94b721", 0.0, True),    # Shape mismatch
    ("36d67576", 82.2, False),  # Close!
    ("3618c87e", 72.0, False),  # Close!
    ("31aa019c", 88.0, False),  # Very close!
    ("845d6e51", 84.8, False),  # Very close!
    ("42918530", 54.5, False),
    ("758abdf0", 17.2, False),
    ("d5c634a2", 16.7, False),
    ("28bf18c6", 33.3, False),
    ("9a4bb226", 0.0, False),
    ("396d80d7", 6.6, False),
    ("673ef223", 81.5, False),  # Close!
    ("469497ad", 0.0, True),    # Shape mismatch
    ("cdecee7f", 22.2, False),
    ("27a77e38", 36.0, False),
    ("3979b1a8", 31.0, False),
    ("e1baa8a4", 0.0, True),    # Shape mismatch
    ("a1aa0c1e", 40.0, False),
    ("239be575", 100.0, False), # EXACT MATCH! (fixed by heuristics)
    ("8e5a5113", 0.0, False),
    ("668eec9a", 20.0, False),
    ("e5062a87", 66.0, False),
    ("dc1df850", 76.6, False),  # Close!
    ("5a5a2103", 42.4, False),
    ("8fff9e47", 11.1, False),
    ("9caf5b84", 0.0, False),
    ("f83cb3f6", 92.9, False),  # Very close!
    ("332202d5", 0.0, False),
    ("8d510a79", 65.0, False),
    ("88a62173", 25.0, False),
    ("2037f2c7", 0.0, True),    # Shape mismatch
    ("8abad3cf", 0.0, True),    # Shape mismatch
    ("0520fde7", 66.7, False),
    ("91714a58", 94.1, False),  # Very close!
    ("fc10701f", 8.3, False),
    ("2b9ef948", 0.0, False),
    ("6f8cd79b", 30.0, False),
    ("92e50de0", 55.6, False),
    ("67e8384a", 66.7, False),
    ("d749d46f", 0.0, True),    # Shape mismatch
    ("b8825c91", 38.3, False),
    ("d631b094", 0.0, True),    # Shape mismatch
    ("b7cb93ac", 0.0, False),
    ("d687bc17", 95.1, False),  # Very close!
    ("ea959feb", 13.8, False),
    ("a48eeaf7", 92.0, False),  # Very close!
    ("a416fc5b", 0.0, False),
    ("54d82841", 76.0, False),  # Close!
    ("9ba4a9aa", 0.0, False),
    ("b91ae062", 0.0, True),    # Shape mismatch
    ("7d419a02", 52.7, False),
    ("c8f0f002", 8.3, False),
    ("bbc9ae5d", 0.0, True),    # Shape mismatch
    ("bc4146bd", 31.2, False),
]

# Categorize by accuracy ranges
very_close = []  # 80-99%
close = []       # 60-79%
moderate = []    # 40-59%
poor = []        # 20-39%
very_poor = []   # 1-19%
zero = []        # 0%

for task_id, acc, shape_mismatch in pixel_accuracies:
    if shape_mismatch:
        continue  # Skip shape mismatches for now
    
    if 80 <= acc < 100:
        very_close.append((task_id, acc))
    elif 60 <= acc < 80:
        close.append((task_id, acc))
    elif 40 <= acc < 60:
        moderate.append((task_id, acc))
    elif 20 <= acc < 40:
        poor.append((task_id, acc))
    elif 0 < acc < 20:
        very_poor.append((task_id, acc))
    elif acc == 0:
        zero.append((task_id, acc))

print("NEAR-MISS ANALYSIS")
print("=" * 60)
print(f"\nTotal tasks analyzed: {len(pixel_accuracies)}")
print(f"Shape mismatches: {sum(1 for _, _, sm in pixel_accuracies if sm)}")
print(f"Exact match (100%): 1 (239be575 - fixed by heuristics!)")

print(f"\n🎯 VERY CLOSE (80-99% accuracy): {len(very_close)} tasks")
print("These are prime candidates for heuristic improvements!")
for task_id, acc in sorted(very_close, key=lambda x: x[1], reverse=True):
    print(f"  {task_id}: {acc:.1f}%")

print(f"\n🔸 CLOSE (60-79% accuracy): {len(close)} tasks")
for task_id, acc in sorted(close, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {task_id}: {acc:.1f}%")

print(f"\n📊 Summary:")
print(f"  Very close (80-99%): {len(very_close)} tasks")
print(f"  Close (60-79%): {len(close)} tasks")
print(f"  Moderate (40-59%): {len(moderate)} tasks")
print(f"  Poor (20-39%): {len(poor)} tasks")
print(f"  Very poor (1-19%): {len(very_poor)} tasks")
print(f"  Zero (0%): {len(zero)} tasks")

# Key insights
print("\n🔍 KEY INSIGHTS:")
print("\n1. HIGH-VALUE TARGETS (80%+ accuracy):")
print("   These 17 tasks are SO CLOSE to being correct!")
print("   Common issues likely include:")
print("   - Missing single pixels (Object Integrity)")
print("   - Wrong color for 1-2 cells (Color Palette)")
print("   - Minor symmetry issues (Symmetry Solver)")
print("   - Small edge/border errors")

print("\n2. HEURISTIC SUCCESS STORY:")
print("   Task 239be575: 100% accuracy AFTER heuristics")
print("   This proves the heuristic pipeline can fix neural outputs!")

print("\n3. RECOMMENDED HEURISTIC ENHANCEMENTS:")
print("   a) Single Pixel Fixer: Detect isolated missing/extra pixels")
print("   b) Color Consistency: Fix cells that break color patterns")
print("   c) Edge Completer: Complete partial edges/borders")
print("   d) Pattern Extender: Extend incomplete repeating patterns")
print("   e) Object Completer: Fill holes in solid objects")

# Specific patterns to look for
print("\n4. SPECIFIC PATTERNS IN HIGH-ACCURACY FAILURES:")
very_close_sorted = sorted(very_close, key=lambda x: x[1], reverse=True)
print(f"\n   Top 5 nearest misses:")
for i, (task_id, acc) in enumerate(very_close_sorted[:5]):
    print(f"   {i+1}. {task_id}: {acc:.1f}% - Likely missing <{100-acc:.0f}% of pixels")
    if acc > 95:
        print(f"      → This is probably 1-2 pixels wrong!")
    elif acc > 90:
        print(f"      → Likely a small object or edge issue")
    elif acc > 85:
        print(f"      → Could be a color swap or small region")