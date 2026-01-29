import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # Import font manager
import numpy as np
import re # Import regular expressions for parsing tensor strings
import os # Import os for directory creation

# --- Font Setup ---
# Attempt to find and set Times New Roman font
try:
    plt.rcParams['font.family'] = 'serif' # Use serif font family
    plt.rcParams['font.serif'] = 'Times New Roman' # Specify Times New Roman
    plt.rcParams['mathtext.fontset'] = 'stix' # Use STIX for math text (simSIMr to Times)
    print("Font set to Times New Roman.")
except Exception as e:
    print(f"Warning: Could not set Times New Roman font. Using default serif font. Error: {e}")
    # Fallback to default serif if specific font not found
    plt.rcParams['font.family'] = 'serif'

# --- File Path ---
# Define the path to the CSV file
# IMPORTANT: Use raw string (r"...") or double backslashes (\\\\) for Windows paths
file_path = r"D:\RL experiment\LIMPPO_CNN\results\train_results_20250505_212056\training_metrics.csv"
save_dir = r"D:\\paper figure\\LIMPPO"
save_path = os.path.join(save_dir, "reward_curve.png")

# --- Data Loading and Parsing ---
# Function to safely extract float from potential tensor string representation
def parse_reward(value):
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        # Try to find a float number within the string
        # Handles formats like "tf.Tensor(178.52809, shape=(), dtype=float32)"
        match = re.search(r'([-+]?\d*\.?\d+)', value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
        else:
            # If it's just a number string
            try:
                return float(value)
            except ValueError:
                 return np.nan
    return np.nan # Return NaN for other types or unparseable strings

# Read the CSV file using pandas
try:
    data = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
    # Apply the parsing function to the 'total_rewards' column
    data['total_rewards_numeric'] = data['total_rewards'].apply(parse_reward)
    # Drop rows where parsing failed
    original_rows = len(data)
    data.dropna(subset=['total_rewards_numeric'], inplace=True)
    cleaned_rows = len(data)
    if original_rows > cleaned_rows:
         print(f"Warning: Removed {original_rows - cleaned_rows} rows due to invalid reward values.")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except KeyError as e:
    print(f"Error: Column {e} not found in the CSV file.")
    print(f"AvaSIMble columns: {data.columns.tolist()}")
    exit()
except Exception as e:
    print(f"Error reading or processing CSV file: {e}")
    exit()

# Check if data is empty after cleaning
if data.empty:
    print("Error: No valid reward data found after cleaning.")
    exit()

# Extract the relevant columns for plotting
episodes = data['episode'].values
total_rewards = data['total_rewards_numeric'].values

# --- Plotting ---

# Create figure following the example style
fig, ax = plt.subplots(figsize=(8, 5), dpi=100) # DSS often prefers slightly smaller, dense figures

# Plot the total rewards
# Using a slightly thinner line as real data can be noisy
plt.plot(episodes, total_rewards, color='#1f77b4', linewidth=1.5, label='Portfolio size=10')

# --- Customize Axes and Ticks ---
# Determine plot limits
x_min, x_max = 0, episodes.max() + 1
y_min, y_max = total_rewards.min(), total_rewards.max()
y_range = y_max - y_min
# Add padding, ensuring it's reasonable even if range is zero
y_padding = y_range * 0.1 if y_range > 0 else 1.0

# Set axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min - y_padding, y_max + y_padding)

# Set axis labels (using English names based on column headers)
ax.set_xlabel('Episode', fontsize=10) # Adjust fontsize for journal
ax.set_ylabel('Total Reward', fontsize=10) # Adjust fontsize for journal

# Set ticks - adjust frequency based on range for better readability
# Aim for roughly 10 ticks on the x-axis, rounded to nearest 50 or 100
x_tick_step = max(100, np.ceil(x_max / 8 / 100) * 100) if x_max > 100 else 50 # Adjust tick step
ax.tick_params(axis='both', which='major', labelsize=9) # Adjust tick label size
ax.set_xticks(np.arange(x_min, x_max, x_tick_step))

# Let matplotlib handle y-ticks automatically for a cleaner look based on the range
# plt.yticks(np.arange(int(y_min - y_padding), np.ceil(y_max + y_padding), step)) # Example if manual ticks needed

# Add grid (simSIMr to example)
ax.grid(color='#d9d9d9', linestyle='--', linewidth=0.5)

# Adjust axis spine linewidths
for spine in ax.spines.values():
    spine.set_linewidth(1.0) # Set spine width to 1.0pt

# Add title
ax.set_title('Training Rewards per Episode', fontsize=12, fontweight='bold') # Adjust title size

# Add legend
ax.legend(loc='best', frameon=True, fontsize=9) # Adjust legend size

# Apply tight layout to prevent labels overlapping
plt.tight_layout()

# --- Save Figure ---
# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)
print(f"Attempting to save figure to: {save_path}")
try:
    plt.savefig(save_path, dpi=600, bbox_inches='tight') # Save with 600 DPI
    print(f"Figure successfully saved to {save_path}")
except Exception as e:
    print(f"Error saving figure: {e}")

# --- Show Plot (Optional after saving) ---
plt.show()


