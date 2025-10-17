"""
Optimized filesystem utilities for hierarchical federated learning.
Consolidates metrics storage while preserving signals and models structure.
"""
from pathlib import Path
from .metrics_storage import get_metrics_storage

def get_signals_dir(project_root: Path) -> Path:
    """Get or create the global signals directory (for cloud start/complete signals)."""
    signals_dir = project_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    return signals_dir

def get_models_dir(project_root: Path) -> Path:
    """Get or create the models directory for storing .pkl files."""
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def get_round_signals_dir(project_root: Path, global_round: int) -> Path:
    """Get directory for round-specific signals (completion signals only)."""
    round_signals_dir = project_root / "signals" / f"round_{global_round}"
    round_signals_dir.mkdir(parents=True, exist_ok=True)
    return round_signals_dir

def get_server_signal_path(project_root: Path, global_round: int, server_id: int) -> Path:
    """Get path for server completion signal."""
    signals_dir = get_round_signals_dir(project_root, global_round)
    return signals_dir / f"server_{server_id}_completion.signal"

def get_model_path(project_root: Path, server_id: int, global_round: int) -> Path:
    """Get path for server model file."""
    models_dir = get_models_dir(project_root)
    return models_dir / f"model_s{server_id}_g{global_round}.pkl"

def get_global_model_path(project_root: Path, global_round: int) -> Path:
    """Get path for global aggregated model."""
    models_dir = get_models_dir(project_root)
    return models_dir / f"global_model_round_{global_round}.pkl"

def save_server_metrics_optimized(project_root: Path, global_round: int, server_id: int, 
                                 server_metrics: dict, client_metrics: list, comm_metrics: list):
    """Save all server-related metrics using consolidated storage."""
    storage = get_metrics_storage(project_root)
    
    # Save server metrics
    storage.save_server_metrics(global_round, server_id, server_metrics)
    
    # Save client metrics
    for client_data in client_metrics:
        client_id = client_data.get('client_id', f'client_{client_data.get("partition_id", "unknown")}')
        storage.save_client_metrics(global_round, server_id, client_id, client_data)
    
    # Save communication metrics
    for comm_data in comm_metrics:
        client_id = comm_data.get('client_id', f'client_{comm_data.get("partition_id", "unknown")}')
        storage.save_communication_metrics(global_round, server_id, client_id, comm_data)

def save_global_metrics_optimized(project_root: Path, global_round: int, global_metrics: dict):
    """Save global metrics using consolidated storage."""
    storage = get_metrics_storage(project_root)
    storage.save_global_metrics(global_round, global_metrics)

def create_completion_signal(project_root: Path, global_round: int, server_id: int, message: str = None):
    """Create server completion signal."""
    signal_path = get_server_signal_path(project_root, global_round, server_id)
    timestamp = __import__('datetime').datetime.now().isoformat()
    content = message or f"Server {server_id} completed round {global_round} at {timestamp}"
    signal_path.write_text(content)
    return signal_path

def check_all_servers_completed(project_root: Path, global_round: int, num_servers: int) -> bool:
    """Check if all servers have completed the current round."""
    for server_id in range(num_servers):
        signal_path = get_server_signal_path(project_root, global_round, server_id)
        if not signal_path.exists():
            return False
    return True

def cleanup_old_signals(project_root: Path, keep_last_n_rounds: int = 5):
    """Clean up old signal files to prevent accumulation."""
    signals_dir = project_root / "signals"
    if not signals_dir.exists():
        return
    
    # Get all round signal directories
    round_dirs = sorted([d for d in signals_dir.iterdir() if d.is_dir() and d.name.startswith("round_")],
                       key=lambda x: int(x.name.split("_")[1]))
    
    # Keep only the last N rounds
    if len(round_dirs) > keep_last_n_rounds:
        for old_dir in round_dirs[:-keep_last_n_rounds]:
            import shutil
            shutil.rmtree(old_dir)
            print(f"🧹 Cleaned up old signals: {old_dir.name}")

def cleanup_old_models(project_root: Path, keep_last_n_rounds: int = 10):
    """Clean up old model files to prevent disk space issues."""
    models_dir = get_models_dir(project_root)
    if not models_dir.exists():
        return
    
    # Get all model files and sort by round number
    model_files = []
    for model_file in models_dir.glob("*.pkl"):
        try:
            # Extract round number from filename
            if "_g" in model_file.stem:
                round_num = int(model_file.stem.split("_g")[1])
                model_files.append((round_num, model_file))
        except (ValueError, IndexError):
            continue
    
    # Sort by round number and keep only recent ones
    model_files.sort(key=lambda x: x[0])
    if len(model_files) > keep_last_n_rounds * 4:  # Assuming 3 servers + 1 global model per round
        cutoff_round = model_files[-keep_last_n_rounds * 4][0]
        for round_num, model_file in model_files:
            if round_num < cutoff_round:
                model_file.unlink()
                print(f"🧹 Cleaned up old model: {model_file.name}")

# Legacy compatibility functions (redirect to optimized versions)
def leaf_server_dir(project_root: Path, server_id: int, global_round: int = 0) -> Path:
    """LEGACY: Returns signals directory for backward compatibility."""
    return get_round_signals_dir(project_root, global_round)

def global_dir(project_root: Path, global_round: int = 0) -> Path:
    """LEGACY: Returns signals directory for backward compatibility."""
    return get_round_signals_dir(project_root, global_round)
