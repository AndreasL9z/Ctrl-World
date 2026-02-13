import tensorflow_datasets as tfds
import random
from tqdm import tqdm
import os
from oxe_configs import OXE_DATASET_CONFIGS
import json
import numpy as np
import h5py
from PIL import Image
import re
import gc
import signal
import sys
from llm_instruction_verb_filter import instruction_matches_prompt  # Updated interface

SAVE_H5_NAME = "droid_pick_eval_positive_400_with_action.h5"  # Name of the saved h5 file
DEBUG = False  # Whether to use DROID_100
MAX_SAMPLES = 400 #float("inf")
TRAIN_SPLIT = "train"  # "test", droid training set does not exist

# Prevent TFDS from occupying all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

POSSIBLE_LANG_INSTRUCTION_KEYS = [
    "language_instruction",
    "language_instruction_2", 
    "language_instruction_3",
]

# Specify the list of instructions to filter
TARGET_INSTRUCTIONS = [
    "Pick the marker and put it in the cup",
    "Pick up the marker from the table and put it in the bowl", 
    "Pick the marker and put it in the bowl",
    "Pick an object from the cup",
    "Pick up the bottle with the black label and put it on the counter",
    "Pick up the bottle on the counter and put it on the stove",
    "Pick up the straw on the table and put it inside the can.",
    "Pick the fork from the cup and put it on the table",
    "Pick up the green cube and put it in the bowl",
    "Pick up the orange cup from the table and put it in the black bowl.",
    "Pick the cloth and shove it in the box",
    "Pick up the clothes from the table and put them in the box.",
    "Pick up the rectangular object on the table and put it on the bridge like object",
    "Pick the blue object and move it to the bottom on the desk",
    "Pick up the silver bowl and move it to the right.",
    "Pick up the yellow bell pepper and put it on the table",
    "Pick the marker in the cup and put it on the table",
    "Pick up the pen and put it in the cup",
    "Pick up the marker and put it in the bowl",
    "Pick the marker in the cup and put it on the table"
    # "Put the marker in the pot"
]

# Filter mode configuration
FILTER_MODE = "keyword"  # "exact" or "keyword" or "both"
# exact: Only match exact instructions in TARGET_INSTRUCTIONS
# keyword: Only match instructions containing keywords
# both: Match exact instructions in TARGET_INSTRUCTIONS or instructions containing keywords

# Feature switch configuration
ENABLE_FILTERING = True  # Whether to enable filtering function
ENABLE_INCLUDE_KEYWORDS = True  # Whether to enable include keywords filtering function
ENABLE_EXCLUDE_KEYWORDS = False  # Whether to enable exclude keywords function
# New: LLM verb filtering switch
ENABLE_LLM_VERB_FILTER = True  # Whether to enable filtering through LLM instruction judgment
# Select prompt type: "wipe" or "pick_place"
LLM_PROMPT_KEY = "pick_place"

# Keywords list (used when FILTER_MODE is "keyword" or "both")
KEYWORDS = ["pick"]  # Can add more keywords, such as ["pick", "grab", "take"]

# Exclude keywords list (instructions containing these keywords will be excluded)
EXCLUDE_KEYWORDS = [] #["pick", "take", "put", "move", "place"]  # Exclude instructions containing "pick"

# Convert target instructions to lowercase for comparison
TARGET_INSTRUCTIONS_LOWER = [instr.lower().strip(" .,!?-_") for instr in TARGET_INSTRUCTIONS]

# Convert keywords to lowercase
KEYWORDS_LOWER = [kw.lower() for kw in KEYWORDS]

# Convert exclude keywords to lowercase
EXCLUDE_KEYWORDS_LOWER = [kw.lower() for kw in EXCLUDE_KEYWORDS]

# Global variable for safe exit
h5_file = None

def signal_handler(sig, frame):
    """Handle interrupt signals to ensure h5 file is properly closed"""
    print('\nReceived interrupt signal, safely closing file...')
    if h5_file is not None:
        try:
            h5_file.flush()
            h5_file.close()
            print("File safely closed")
        except:
            pass
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def force_gc():
    """Force garbage collection"""
    gc.collect()

def matches_exact_instruction(instruction):
    """
    Check if instruction exactly matches the target instruction list
    
    Args:
        instruction (str): Language instruction to check
        
    Returns:
        bool: Returns True if instruction exactly matches target instruction, otherwise False
    """
    if not instruction or instruction.strip() == "":
        return False
        
    # Clean instruction text
    cleaned_instruction = instruction.lower().strip(" .,!?-_")
    
    # Check if it matches any target instruction
    return cleaned_instruction in TARGET_INSTRUCTIONS_LOWER

def matches_keyword_instruction(instruction):
    """
    Check if instruction contains any keywords (supports fuzzy matching)
    
    Args:
        instruction (str): Language instruction to check
        
    Returns:
        bool: Returns True if instruction contains any keyword, otherwise False
    """
    if not instruction or instruction.strip() == "":
        return False
        
    # Clean instruction text
    cleaned_instruction = instruction.lower().strip(" .,!?-_")
    
    # Check if it contains any keywords (fuzzy matching)
    for keyword in KEYWORDS_LOWER:
        # Use regular expressions for more precise word boundary matching
        import re
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, cleaned_instruction):
            return True
    
    return False

def matches_exclude_keywords(instruction):
    """
    Check if instruction contains any exclude keywords (supports fuzzy matching)
    
    Args:
        instruction (str): Language instruction to check
        
    Returns:
        bool: Returns True if instruction contains any exclude keyword, otherwise False
    """
    if not instruction or instruction.strip() == "":
        return False
        
    # Clean instruction text (use same normalization logic as keyword matching)
    cleaned_instruction = instruction.lower().strip(" .,!?-_")
    
    # Check if it contains any exclude keywords (fuzzy matching)
    for keyword in EXCLUDE_KEYWORDS_LOWER:
        # Use regular expressions for more precise word boundary matching
        import re
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, cleaned_instruction):
            return True
    
    return False

def matches_target_instruction(instruction):
    """
    Check if instruction meets filtering criteria according to configured FILTER_MODE
    
    Args:
        instruction (str): Language instruction to check
        
    Returns:
        bool: Returns True if instruction meets filtering criteria, otherwise False
    """
    if not instruction or instruction.strip() == "":
        return False
    
    # If filtering is disabled, return True directly (keep all instructions)
    if not ENABLE_FILTERING:
        return True
    
    # Decide whether to check exclude keywords based on switch
    if ENABLE_EXCLUDE_KEYWORDS and matches_exclude_keywords(instruction):
        return False
    
    # Decide whether to check include keywords based on switch
    if ENABLE_INCLUDE_KEYWORDS:
        if FILTER_MODE == "exact":
            return matches_exact_instruction(instruction)
        elif FILTER_MODE == "keyword":
            return matches_keyword_instruction(instruction)
        elif FILTER_MODE == "both":
            return matches_exact_instruction(instruction) or matches_keyword_instruction(instruction)
        else:
            raise ValueError(f"Unknown filter mode: {FILTER_MODE}. Please use 'exact', 'keyword', or 'both'")
    else:
        # If include keyword filtering is not enabled, only perform exclude filtering, keep others
        return True

def process_task_name(task):
    """
    Process and clean task name
    
    Args:
        task (str): Original task string
        
    Returns:
        str: Cleaned task string
    """
    if not task:
        return ""
    
    # Capitalize first letter
    task = task.capitalize()
    # Remove trailing punctuation and clean whitespace
    task = task.strip(" .,!?-_")
    return task

dataset_name = "droid"
if DEBUG:
    dataset_name += "_100"

# Statistics tracking
tasks_seen = dict()
filtered_tasks = dict()
total_episodes_processed = 0
total_episodes_kept = 0
total_samples = 0
# Add independent episode counter for each task group
task_episode_counters = {}

# Print filtering configuration
print("="*50)
print("Filtering Configuration")
print("="*50)
print(f"Filtering function: {'Enabled' if ENABLE_FILTERING else 'Disabled'}")
if ENABLE_FILTERING:
    print(f"Include keywords filtering: {'Enabled' if ENABLE_INCLUDE_KEYWORDS else 'Disabled'}")
    if ENABLE_INCLUDE_KEYWORDS:
        print(f"Filter mode: {FILTER_MODE}")
        if FILTER_MODE in ["exact", "both"]:
            print(f"Number of target instructions: {len(TARGET_INSTRUCTIONS)}")
        if FILTER_MODE in ["keyword", "both"]:
            print(f"Keywords: {KEYWORDS}")
    print(f"Exclude keywords filtering: {'Enabled' if ENABLE_EXCLUDE_KEYWORDS else 'Disabled'}")
    if ENABLE_EXCLUDE_KEYWORDS:
        print(f"Exclude keywords: {EXCLUDE_KEYWORDS}")
else:
    print("⚠️  Filtering function is disabled, all instructions will be retained")
print("="*50)

try:
    with h5py.File(SAVE_H5_NAME, "w") as f:
        h5_file = f  # Set global variable for signal handling
        
        dataset = tfds.load(
            dataset_name, data_dir="gs://gresearch/robotics", split=TRAIN_SPLIT
        )
        
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name.split("_")[0]][
            "image_obs_keys"
        ]  # Dictionary mapping img_keys to image names in OXE

        # Get all three views: primary, secondary, wrist
        views = [img_key_to_name["primary"], img_key_to_name["secondary"], img_key_to_name["wrist"]]
        
        len_of_dataset = min(dataset.cardinality().numpy(), MAX_SAMPLES)
        
        i = 0
        # Convert to iterator to catch exceptions from failed episode loading
        iter_dataset = iter(dataset)
        
        while True:
            try:
                episode = next(iter_dataset)
                total_episodes_processed += 1
                
                # Perform garbage collection every 50 episodes
                if total_episodes_processed % 50 == 0:
                    force_gc()
                    f.flush()  # Flush to disk
                
                # Print progress
                if total_episodes_processed % 100 == 0:
                    print(f"Processing episode {total_episodes_processed} of {len_of_dataset}, dataset: {dataset_name}")
                    print(f"Episodes retained: {total_episodes_kept}")
                    if total_episodes_processed > 0:
                        print(f"Filtering success rate: {total_episodes_kept/total_episodes_processed:.2%}")
                
                # Break if enough samples have been saved
                if total_episodes_kept >= MAX_SAMPLES:
                    break
                    
                # Extract task (language instruction)
                task = None
                try:
                    for _, step in enumerate(episode["steps"]):
                        if task is None or task == "":
                            for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
                                if key in step:
                                    task = step[key].numpy().decode()
                                    break
                        if task:  # Exit early once task is found
                            break
                except Exception as e:
                    print(f"Failed to extract task: {e}")
                    continue

                if task is None or task == "":
                    continue

                # Clean task name
                original_task = task
                task = process_task_name(task)

                # Track all seen tasks
                if task not in tasks_seen:
                    tasks_seen[task] = 0
                tasks_seen[task] += 1
                
                # Apply filtering logic
                if not matches_target_instruction(task):
                    continue

                # -------- LLM instruction filtering (after keyword filtering) --------
                if ENABLE_LLM_VERB_FILTER:
                    try:
                        if not instruction_matches_prompt(task, prompt_key=LLM_PROMPT_KEY):
                            continue  # Does not meet LLM rules, skip this instruction
                    except Exception as e:
                        print(f"LLM instruction filtering failed: {e}, skip this instruction")
                        continue
                    
                # Track filtered tasks
                if task not in filtered_tasks:
                    filtered_tasks[task] = 0
                filtered_tasks[task] += 1
                
                total_episodes_kept += 1
                i += 1
                
                print(f"--- Retained episode {i}: '{task}' ---")
                
                # Extract episode images for each view
                episode_images_list = [[] for _ in range(len(views))]
                # Extract joint positions, gripper positions, action data and all action_dict fields
                joint_positions = []
                gripper_positions = []
                actions = []

                # action_dict fields
                act_cartesian_pos = []
                act_cartesian_vel = []
                act_gripper_pos   = []
                act_gripper_vel   = []
                act_joint_pos     = []
                act_joint_vel     = []
 
                try:
                    for step_idx, step in enumerate(episode["steps"]):
                        # Extract images from all views
                        for j, img_key in enumerate(views):
                            episode_images_list[j].append(step["observation"][img_key].numpy())
                        
                        # Extract joint positions and gripper positions
                        joint_positions.append(step["observation"]["joint_position"].numpy())
                        gripper_positions.append(step["observation"]["gripper_position"].numpy())
                        # Extract action data
                        actions.append(step["action"].numpy())

                        # Extract all action_dict fields
                        adict = step["action_dict"]
                        act_cartesian_pos.append(adict["cartesian_position"].numpy())
                        act_cartesian_vel.append(adict["cartesian_velocity"].numpy())
                        act_gripper_pos.append(adict["gripper_position"].numpy())
                        act_gripper_vel.append(adict["gripper_velocity"].numpy())
                        act_joint_pos.append(adict["joint_position"].numpy())
                        act_joint_vel.append(adict["joint_velocity"].numpy())
                except Exception as e:
                    print(f"Failed to extract episode data: {e}")
                    continue

                # Ensure data integrity
                if len(joint_positions) == 0:
                    print(f"Episode {i} data is empty, skipping")
                    continue

                try:
                    # Process each view
                    for view_idx, episode_images in enumerate(episode_images_list):
                        # Create task group if it doesn't exist
                        if task not in f:
                            f.create_group(task)
                            # Save task instruction as attribute
                            f[task].attrs['instruction'] = task
                            f[task].attrs['original_instruction'] = original_task
                            # Initialize episode counter for task group
                            task_episode_counters[task] = 0

                        # Get task group
                        task_group = f[task]
                        
                        # Increment episode counter for task group (only for first view)
                        if view_idx == 0:
                            task_episode_counters[task] += 1
                            current_task_episode = task_episode_counters[task]
                        
                        # Create unique episode identifier (using numbering within task group)
                        episode_name = f"ep_{current_task_episode}_view_{view_idx}"
                        
                        # Convert to numpy array for storage (no frame sampling)
                        episode_images_array = np.array(episode_images)
                        
                        # Save episode images
                        task_group.create_dataset(
                            episode_name,
                            data=episode_images_array,
                            compression="gzip",
                            compression_opts=6,
                        )
                        
                        # Save metadata
                        task_group[episode_name].attrs['num_frames'] = len(episode_images)
                        task_group[episode_name].attrs['view'] = view_idx
                        task_group[episode_name].attrs['view_name'] = views[view_idx]

                    # Save joint positions, gripper positions and action data (only once per episode)
                    task_group = f[task]
                    joint_episode_name = f"ep_{current_task_episode}_joint_positions"
                    gripper_episode_name = f"ep_{current_task_episode}_gripper_positions"
                    action_episode_name = f"ep_{current_task_episode}_actions"
                    
                    # Save joint positions
                    task_group.create_dataset(
                        joint_episode_name,
                        data=np.array(joint_positions),
                        compression="gzip",
                        compression_opts=6,
                    )
                    
                    # Save gripper positions
                    task_group.create_dataset(
                        gripper_episode_name,
                        data=np.array(gripper_positions),
                        compression="gzip",
                        compression_opts=6,
                    )
                    
                    # Save action data
                    task_group.create_dataset(
                        action_episode_name,
                        data=np.array(actions),
                        compression="gzip",
                        compression_opts=6,
                    )

                    # Save action_dict fields
                    def _save(name, arr, dtype):
                        task_group.create_dataset(
                            name,
                            data=np.array(arr, dtype=dtype),
                            compression="gzip",
                            compression_opts=6,
                        )

                    _save(f"ep_{current_task_episode}_act_cartesian_pos", act_cartesian_pos, np.float64)
                    _save(f"ep_{current_task_episode}_act_cartesian_vel", act_cartesian_vel, np.float64)
                    _save(f"ep_{current_task_episode}_act_gripper_pos", act_gripper_pos, np.float64)
                    _save(f"ep_{current_task_episode}_act_gripper_vel", act_gripper_vel, np.float64)
                    _save(f"ep_{current_task_episode}_act_joint_pos", act_joint_pos, np.float64)
                    _save(f"ep_{current_task_episode}_act_joint_vel", act_joint_vel, np.float64)

                    # Add metadata
                    for ds_name, dtype_label in [
                        ("act_cartesian_pos", "cartesian_position"),
                        ("act_cartesian_vel", "cartesian_velocity"),
                        ("act_gripper_pos", "gripper_position"),
                        ("act_gripper_vel", "gripper_velocity"),
                        ("act_joint_pos", "joint_position"),
                        ("act_joint_vel", "joint_velocity"),
                    ]:
                        full_ds_name = f"ep_{current_task_episode}_{ds_name}"
                        task_group[full_ds_name].attrs["data_type"] = dtype_label
                        task_group[full_ds_name].attrs["num_timesteps"] = len(act_cartesian_pos)

                    total_samples += 1
                    
                    # Flush after saving each episode to ensure data is written to disk
                    f.flush()
                    
                except Exception as e:
                    print(f"Failed to save episode {i} data: {e}")
                    # Continue processing next episode even if saving fails
                    continue
                
            except StopIteration:
                print("Dataset processing completed")
                break
            except Exception as e:
                print(f"Failed to load episode {total_episodes_processed}, error: {e}")
                # Perform garbage collection when error occurs
                force_gc()
                continue

        # Final flush to ensure all data is written
        print("Completing final write...")
        f.flush()
        
except Exception as e:
    print(f"Serious error occurred during overall processing: {e}")
    print("Program will try to save processed data...")
finally:
    # Ensure file handle is released
    h5_file = None
    force_gc()

# Print final statistics
print("\n" + "="*50)
print("Processing Complete")
print("="*50)
print(f"\nFiltering function: {'Enabled' if ENABLE_FILTERING else 'Disabled'}")
if ENABLE_FILTERING:
    print(f"Include keywords filtering: {'Enabled' if ENABLE_INCLUDE_KEYWORDS else 'Disabled'}")
    if ENABLE_INCLUDE_KEYWORDS:
        print(f"Filter mode: {FILTER_MODE}")
        if FILTER_MODE in ["exact", "both"]:
            print(f"\nTarget instruction list ({len(TARGET_INSTRUCTIONS)}):")
            for i, instr in enumerate(TARGET_INSTRUCTIONS):
                print(f"  {i+1}. {instr}")

        if FILTER_MODE in ["keyword", "both"]:
            print(f"\nKeywords list:")
            for i, keyword in enumerate(KEYWORDS):
                print(f"  {i+1}. {keyword}")

    print(f"Exclude keywords filtering: {'Enabled' if ENABLE_EXCLUDE_KEYWORDS else 'Disabled'}")
    if ENABLE_EXCLUDE_KEYWORDS:
        print(f"\nExclude keywords list:")
        for i, keyword in enumerate(EXCLUDE_KEYWORDS):
            print(f"  {i+1}. {keyword}")
else:
    print("⚠️  Filtering function is disabled, all instructions are retained")
print(f"Total episodes processed: {total_episodes_processed}")
print(f"Total episodes retained: {total_episodes_kept}")
if total_episodes_processed > 0:
    print(f"Filtering success rate: {total_episodes_kept/total_episodes_processed:.2%}")
print(f"Total samples saved: {total_samples}")

print(f"\nAll seen tasks ({len(tasks_seen)}):")
for task, count in sorted(tasks_seen.items(), key=lambda x: x[1], reverse=True):
    print(f"  {task}: {count}")

print(f"\nTasks retained after filtering ({len(filtered_tasks)}):")
for task, count in sorted(filtered_tasks.items(), key=lambda x: x[1], reverse=True):
    print(f"  {task}: {count}")

print(f"\nView configuration:")
print(f"  Primary: {views[0] if len(views) > 0 else 'N/A'}")
print(f"  Secondary: {views[1] if len(views) > 1 else 'N/A'}")
print(f"  Wrist: {views[2] if len(views) > 2 else 'N/A'}")

print(f"\nData save content:")
print(f"  - Image data (3 views, per step)")
print(f"  - Joint position data (per step)")
print(f"  - Gripper position data (per step)")
print(f"  - Action data (per step)")
print(f"  - Complete time series (no frame sampling)")

# Verify if file is readable
try:
    with h5py.File(SAVE_H5_NAME, "r") as test_f:
        print(f"\n✅ File verification successful: {SAVE_H5_NAME} can be read normally")
        print(f"Contained task groups: {list(test_f.keys())}")
except Exception as e:
    print(f"\n❌ File verification failed: {e}")