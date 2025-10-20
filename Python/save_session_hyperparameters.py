import os
import json
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def save_session_hyperparameters(
    config, 
    training_state=None,
    hyperparams_dir: str = "./hyperparameters",
    resumed: bool = False,
    resumed_from: Optional[str] = None
) -> str:
    """
    Save hyperparameters to a separate JSON file at the start of a training session.
    
    Args:
        config: TrainingConfig object containing all hyperparameters
        training_state: Optional TrainingState object (for resumed sessions)
        hyperparams_dir: Directory to save hyperparameter files
        resumed: Whether this is a resumed session
        resumed_from: Path to checkpoint resumed from (if applicable)
        
    Returns:
        Path to the saved hyperparameters file
    """
    # Create directory if it doesn't exist
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include experiment name in filename if available
    if hasattr(config, 'experiment_name') and config.experiment_name:
        filename = f"hyperparams_{config.experiment_name}_{timestamp}.json"
    else:
        filename = f"hyperparams_{timestamp}.json"
    
    filepath = os.path.join(hyperparams_dir, filename)
    
    # Build hyperparameter dictionary
    hyperparams = {
        # Session metadata
        "session_info": {
            "timestamp": datetime.now().isoformat(),
            "resumed": resumed,
            "resumed_from": resumed_from,
            "starting_episode": training_state.episode if training_state else 0,
            "starting_timesteps": training_state.timesteps if training_state else 0,
        },
        
        # Experiment identification
        "experiment": {
            "name": getattr(config, 'experiment_name', None),
            "tags": getattr(config, 'experiment_tags', []),
            "notes": getattr(config, 'experiment_notes', None),
            "random_seed": getattr(config, 'random_seed', None),
        },
        
        # All configuration parameters
        "config": config.to_dict(),
        
        # Metadata
        "metadata": {
            "config_source": getattr(config, '_config_file', 'default'),
            "save_location": filepath,
        }
    }
    
    # Save to JSON file with nice formatting
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=2, sort_keys=False)
    
    logger.info(f"✓ Saved session hyperparameters to {filepath}")
    
    return filepath