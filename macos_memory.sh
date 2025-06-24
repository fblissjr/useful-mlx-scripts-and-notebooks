#!/usr/bin/env bash

# --- Configuration ---
declare -A PRESETS
PRESETS["legacy_default"]="85 75" # Wired Limit: 85%, LWM: 75% (Original script's behavior)
PRESETS["conservative"]="80"      # Wired Limit: 80%, LWM: High (Ample RAM for OS)
PRESETS["moderate"]="90"         # Wired Limit: 90%, LWM: High (Recommended for general MLX use)
PRESETS["aggressive"]="95"       # Wired Limit: 95%, LWM: High (Pushes GPU memory usage)
PRESETS["mlx_tuned"]="92"         # Wired Limit: 92%, LWM: High (Optimized for MLX heavy tasks)

HIGH_LWM_MB_VALUE=999999     # A very large number to effectively set LWM very high.

# --- Helper Functions ---
print_usage() {
  echo "mlx_memory.sh - Manages macOS iGPU wired memory limits for MLX and other GPU tasks."
  echo ""
  echo "IMPORTANT:"
  echo "  - These settings are temporary and will reset upon reboot."
  echo "  - To persist settings across reboots, add them to /etc/sysctl.conf (e.g., iogpu.wired_limit_mb=XXXX)."
  echo "  - Incorrect values can potentially affect system stability. Use with caution."
  echo "  - This script requires 'sudo' to apply changes. Without sudo, it only displays calculations."
  echo ""
  echo "USAGE:"
  echo "  sudo $0 <command|preset|percentages>"
  echo ""
  echo "COMMANDS:"
  echo "  help             Displays this help message."
  echo "  reset            Resets iogpu.wired_limit_mb and iogpu.wired_lwm_mb to system defaults (by setting them to 0)."
  echo "                   Example: sudo $0 reset"
  echo ""
  echo "PRESETS:"
  echo "  Apply predefined settings for 'iogpu.wired_limit_mb' (as a percentage of total RAM)"
  echo "  and sets 'iogpu.wired_lwm_mb' to a very high value ($HIGH_LWM_MB_VALUE MB) unless specified otherwise."
  echo ""
  echo "  Available presets:"
  for preset_key in "${!PRESETS[@]}"; do
    local details="${PRESETS[$preset_key]}"
    local limit_pct desc_lwm
    if [[ "$preset_key" == "legacy_default" ]]; then
        limit_pct="${details% *}%"
        desc_lwm="LWM ${details#* }%"
    else
        limit_pct="${details}%"
        desc_lwm="LWM set to $HIGH_LWM_MB_VALUE MB (effectively very high)"
    fi
    printf "    %-18s Sets Limit to %s, %s\n" "$preset_key" "$limit_pct" "$desc_lwm"
  done
  echo "  Example: sudo $0 moderate"
  echo ""
  echo "CUSTOM PERCENTAGES:"
  echo "  $0 <limit_percent> [lwm_percent]"
  echo ""
  echo "  <limit_percent>  Required. Custom percentage (0-100) of total RAM for iogpu.wired_limit_mb."
  echo "                   Example: sudo $0 90"
  echo "                     (This will set limit to 90% and LWM to $HIGH_LWM_MB_VALUE MB)"
  echo ""
  echo "  [lwm_percent]    Optional. Custom percentage (0-100) of total RAM for iogpu.wired_lwm_mb."
  echo "                   If not provided, iogpu.wired_lwm_mb is set to $HIGH_LWM_MB_VALUE MB."
  echo "                   Example: sudo $0 88 78"
  echo "                     (Sets limit to 88% and LWM to 78%)"
  echo ""
  echo "EXAMPLES:"
  echo "  $0 help                       # Display this help message"
  echo "  sudo $0 moderate              # Apply the 'moderate' preset"
  echo "  sudo $0 90                    # Set wired limit to 90%, LWM to a high default"
  echo "  sudo $0 85 75                 # Set wired limit to 85%, LWM to 75%"
  echo "  sudo $0 reset                 # Reset to system default values"
  echo "  $0 92                       # Show calculation for 92% limit (no sudo = no changes)"
  echo ""
}

# --- Main Logic ---

# If no arguments, or 'help' is passed, show usage and exit.
if [[ $# -eq 0 || "$1" == "help" || "$1" == "--help" || "$1" == "-h" ]]; then
  print_usage
  exit 0
fi

# Get total memory in MB
TOTAL_MEM_MB=$(($(sysctl -n hw.memsize) / 1024 / 1024))
if [[ -z "$TOTAL_MEM_MB" || "$TOTAL_MEM_MB" -le 0 ]]; then
    echo "Error: Could not determine total system memory."
    exit 1
fi

ARG1="$1"
ARG2="$2" # Might be empty

WIRED_LIMIT_PERCENT=""
WIRED_LWM_PERCENT="" # Percentage based LWM, if provided
WIRED_LWM_MB_VALUE="$HIGH_LWM_MB_VALUE" # Default to high fixed value for LWM

# Handle reset command
if [[ "$ARG1" == "reset" ]]; then
  echo "Resetting iogpu.wired_limit_mb and iogpu.wired_lwm_mb to system defaults..."
  if [[ $EUID -eq 0 ]]; then
    sudo sysctl -w iogpu.wired_limit_mb=0
    sudo sysctl -w iogpu.wired_lwm_mb=0 # Resetting LWM also
    echo "Limits reset to system defaults."
  else
    echo "Current user is not root. To apply changes, run with sudo:"
    echo "  sudo $0 reset"
    echo "The commands would be:"
    echo "  sudo sysctl -w iogpu.wired_limit_mb=0"
    echo "  sudo sysctl -w iogpu.wired_lwm_mb=0"
  fi
  exit 0
fi

# Determine parameters based on input
if [[ -n "${PRESETS[$ARG1]}" ]]; then # Is it a preset name?
  SELECTED_PRESET_VALUES="${PRESETS[$ARG1]}"
  if [[ "$ARG1" == "legacy_default" ]]; then
    WIRED_LIMIT_PERCENT="${SELECTED_PRESET_VALUES% *}"
    WIRED_LWM_PERCENT="${SELECTED_PRESET_VALUES#* }"
  else
    WIRED_LIMIT_PERCENT="$SELECTED_PRESET_VALUES"
    # LWM will use HIGH_LWM_MB_VALUE by default for other presets
  fi
  echo "Applying preset: $ARG1"
elif [[ "$ARG1" =~ ^[0-9]+$ ]]; then # Is it a numeric percentage?
  WIRED_LIMIT_PERCENT=$ARG1
  if [[ -n "$ARG2" && "$ARG2" =~ ^[0-9]+$ ]]; then # Is a second numeric percentage provided for LWM?
    WIRED_LWM_PERCENT=$ARG2
    echo "Applying custom percentages: Limit $WIRED_LIMIT_PERCENT%, LWM $WIRED_LWM_PERCENT%"
  else
    # LWM will use HIGH_LWM_MB_VALUE if only limit_percent is provided or ARG2 is not a number
    if [[ -n "$ARG2" ]]; then # ARG2 was there but not a number
        echo "Warning: Second argument '$ARG2' is not a valid percentage. Using default high LWM."
    fi
    echo "Applying custom percentage: Limit $WIRED_LIMIT_PERCENT%, LWM will be set to $HIGH_LWM_MB_VALUE MB."
  fi
else
  echo "Error: Invalid input '$ARG1'."
  echo "Run '$0 help' for usage instructions."
  exit 1
fi

# Validate WIRED_LIMIT_PERCENT
if ! [[ "$WIRED_LIMIT_PERCENT" =~ ^[0-9]+$ ]] || [[ $WIRED_LIMIT_PERCENT -lt 0 || $WIRED_LIMIT_PERCENT -gt 100 ]]; then
  echo "Error: Limit percentage ($WIRED_LIMIT_PERCENT) must be a number between 0 and 100."
  echo "Run '$0 help' for usage instructions."
  exit 1
fi

# Calculate wired limit in MB
WIRED_LIMIT_MB=$(($TOTAL_MEM_MB * $WIRED_LIMIT_PERCENT / 100))

# Calculate LWM if WIRED_LWM_PERCENT was explicitly set
if [[ -n "$WIRED_LWM_PERCENT" ]]; then
  if ! [[ "$WIRED_LWM_PERCENT" =~ ^[0-9]+$ ]] || [[ $WIRED_LWM_PERCENT -lt 0 || $WIRED_LWM_PERCENT -gt 100 ]]; then
    echo "Error: LWM percentage ($WIRED_LWM_PERCENT) must be a number between 0 and 100."
    echo "Run '$0 help' for usage instructions."
    exit 1
  fi
  WIRED_LWM_MB_VALUE=$(($TOTAL_MEM_MB * $WIRED_LWM_PERCENT / 100))
  LWM_DISPLAY_PERCENT="($WIRED_LWM_PERCENT%)"
else
  # WIRED_LWM_MB_VALUE is already set to HIGH_LWM_MB_VALUE
  LWM_DISPLAY_PERCENT="" # No percentage to display if using fixed high value
fi


# Display the calculated values
echo ""
echo "--- System & Calculated Memory Values ---"
echo "Total physical memory: $TOTAL_MEM_MB MB"
echo "Target Maximum GPU wired limit (iogpu.wired_limit_mb): $WIRED_LIMIT_MB MB ($WIRED_LIMIT_PERCENT%)"
echo "Target GPU wired low water mark (iogpu.wired_lwm_mb): $WIRED_LWM_MB_VALUE MB $LWM_DISPLAY_PERCENT"
echo ""

# Confirm before applying if sudo is used
APPLY_CHANGES=false
if [[ $EUID -eq 0 ]]; then
  # Automatically proceed if sudo, as user has explicitly run with privileges.
  # A confirmation step can be re-added if desired, but often for CLI tools run with sudo, action is expected.
  # For safety, let's keep a confirmation.
  read -p "Apply these settings with sysctl? (yes/No): " confirmation
  if [[ "$confirmation" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    APPLY_CHANGES=true
  else
    echo "Settings not applied."
    exit 0
  fi
else
  echo "Current user is not root. To apply these settings, run with sudo:"
  if [[ -n "$ARG2" ]]; then
    echo "  sudo $0 $ARG1 $ARG2"
  else
    echo "  sudo $0 $ARG1"
  fi
  echo "The sysctl commands to achieve this would be:"
fi

# Apply the values with sysctl or display commands
if [[ "$APPLY_CHANGES" == true ]]; then
  echo "Applying settings..."
  if sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB && \
     sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB_VALUE; then
    echo "Settings successfully applied."
    echo "Note: These settings will reset on reboot."
  else
    echo "Error: Failed to apply sysctl settings. Check permissions and values."
    exit 1
  fi
else
  # Display the commands even if not running as sudo, or if user said no to confirmation.
  echo "  sudo sysctl -w iogpu.wired_limit_mb=$WIRED_LIMIT_MB"
  echo "  sudo sysctl -w iogpu.wired_lwm_mb=$WIRED_LWM_MB_VALUE"
fi

exit 0
