"""
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) Implementation

This module implements SCAFFOLD control variates to correct client and server drift
in non-IID federated learning scenarios, addressing the fundamental cause of 
performance collapse after round 30+.

Reference: Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


class SCAFFOLDControlVariates:
    """
    Manages SCAFFOLD control variates for drift correction in federated learning.
    
    Key Concepts:
    - c_i: Client control variate (captures local drift)
    - c_server: Server control variate (captures global drift)  
    - Corrected gradient = local_grad - c_i + c_server
    """
    
    def __init__(self, model: nn.Module):
        """Initialize control variates to zero for all model parameters."""
        self.client_control = self._init_control_variate(model)
        self.server_control = self._init_control_variate(model)
        
    def _init_control_variate(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Initialize control variate as zero tensors matching model parameters."""
        control = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                control[name] = torch.zeros_like(param.data)
        return control
        
    def get_client_control(self) -> Dict[str, torch.Tensor]:
        """Get client control variate c_i."""
        return self.client_control
        
    def get_server_control(self) -> Dict[str, torch.Tensor]:
        """Get server control variate c_server."""
        return self.server_control
        
    def update_client_control(self, 
                            local_model: nn.Module,
                            global_model: nn.Module,
                            learning_rate: float,
                            local_epochs: int) -> None:
        """
        Update client control variate after local training.
        
        Formula: c_i^{new} = c_i^{old} - c_server + (1/K*lr) * (global_model - local_model)
        where K is the number of local steps.
        """
        with torch.no_grad():
            for name, local_param in local_model.named_parameters():
                if name in self.client_control:
                    global_param = dict(global_model.named_parameters())[name]
                    
                    # Compute model difference
                    model_diff = global_param.data - local_param.data
                    
                    # Update control variate
                    self.client_control[name] = (
                        self.client_control[name] 
                        - self.server_control[name]
                        + model_diff / (local_epochs * learning_rate)
                    )
                    
        logger.debug(f"Updated client control variate for {len(self.client_control)} parameters")
        
    def update_server_control(self, 
                            client_controls: List[Dict[str, torch.Tensor]],
                            client_weights: List[float]) -> None:
        """
        Update server control variate by aggregating client controls.
        
        Formula: c_server^{new} = sum(w_i * c_i) where w_i are client weights
        """
        if not client_controls:
            return
            
        # Weighted average of client control variates
        with torch.no_grad():
            for param_name in self.server_control:
                weighted_sum = torch.zeros_like(self.server_control[param_name])
                
                for client_control, weight in zip(client_controls, client_weights):
                    if param_name in client_control:
                        weighted_sum += weight * client_control[param_name]
                        
                self.server_control[param_name] = weighted_sum
                
        logger.debug(f"Updated server control variate from {len(client_controls)} clients")
        
    def apply_scaffold_correction(self,
                                model: nn.Module,
                                learning_rate: float) -> None:
        """
        Apply SCAFFOLD correction to model gradients during training.
        
        This should be called during the training loop to correct gradients:
        corrected_grad = original_grad - c_i + c_server
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.client_control:
                    # Apply correction: grad = grad - c_i + c_server
                    correction = -self.client_control[name] + self.server_control[name]
                    param.grad.data += correction
                    
        logger.debug("Applied SCAFFOLD gradient correction")


def create_scaffold_manager(model: nn.Module) -> SCAFFOLDControlVariates:
    """Factory function to create SCAFFOLD control variate manager."""
    return SCAFFOLDControlVariates(model)


def aggregate_scaffold_controls(client_controls: List[Dict[str, torch.Tensor]], 
                              client_weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Aggregate client control variates into server control variate.
    
    Args:
        client_controls: List of client control variates
        client_weights: Weights for aggregation (typically based on data size)
        
    Returns:
        Aggregated server control variate
    """
    if not client_controls:
        return {}
        
    # Initialize with zeros
    server_control = {}
    for param_name in client_controls[0]:
        server_control[param_name] = torch.zeros_like(client_controls[0][param_name])
        
    # Weighted aggregation
    total_weight = sum(client_weights)
    for client_control, weight in zip(client_controls, client_weights):
        normalized_weight = weight / total_weight
        for param_name in server_control:
            if param_name in client_control:
                server_control[param_name] += normalized_weight * client_control[param_name]
                
    return server_control


def scaffold_enabled_from_config(config: Dict) -> bool:
    """Check if SCAFFOLD is enabled in configuration."""
    return config.get("scaffold_enabled", False)
