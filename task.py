# fedge/task.py - CIFAR-10 implementation (torchvision-only, no HuggingFace)
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import toml

logger = logging.getLogger(__name__)

# Config helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _get_hier_cfg() -> Dict:
    return toml.load(PROJECT_ROOT / "pyproject.toml")["tool"]["flwr"]["hierarchy"]

# Standard CIFAR-10 normalization - FIXED: Use correct values
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)  # FIXED: Corrected from (0.2470, 0.2435, 0.2616)

try:
    from torchvision.transforms import (
        Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
    )
    _HAS_TV = True
except Exception:
    _HAS_TV = False

def _default_transform():
    if not _HAS_TV:
        # minimal transforms if torchvision isn't present
        def _to_tensor(img: Image.Image):
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - np.array(_CIFAR10_MEAN)) / np.array(_CIFAR10_STD)
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
            return torch.from_numpy(arr)
        return _to_tensor

    return Compose([
        ToTensor(),
        Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    ])

def _train_transform():
    if not _HAS_TV:
        return _default_transform()  # fallback to basic transform
    
    return Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])

# SCAFFOLD's exact 6-layer CNN for CIFAR-10
class Net(nn.Module):
    """
    Compact CIFAR-10 CNN: 3 conv blocks (32→64→128), BN+ReLU, maxpool,
    global-avg-pool, classifier. ~0.9M params; solid baseline for CIFAR-10.
    """
    def __init__(self, in_ch: int = 3, img_h: int = 32, img_w: int = 32, n_class: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 32x32 -> 16x16
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 16x16 -> 8x8
            nn.Dropout(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 8x8 -> 4x4
            nn.Dropout(0.1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, 128, 1, 1)
        self.head = nn.Linear(128, n_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# ──────────────────────── Torchvision CIFAR-10 loader ──────────────────────
try:
    from torchvision.datasets import CIFAR10 as TVCIFAR10
    import shutil
except Exception as _tv_exc:
    TVCIFAR10 = None  # Will raise when used

# ──────────────────────── Public loaders ──────────────────────
@dataclass
class Cifar10Data:
    train: Dataset
    test: Dataset
    # convenient raw labels for partitioning
    train_labels: np.ndarray
    test_labels: np.ndarray

def _cleanup_cifar10(root: Path) -> None:
    """Clean up corrupted CIFAR-10 download files."""
    bad_tgz = root / "cifar-10-python.tar.gz"
    if bad_tgz.exists():
        bad_tgz.unlink()
    # Remove extracted directory if partially corrupted
    extracted = root / "cifar-10-batches-py"
    if extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)

def _manual_cifar10_download(root: Path) -> bool:
    """Manual CIFAR-10 download with robust retry logic."""
    import urllib.request
    import hashlib
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    target_file = root / "cifar-10-python.tar.gz"
    expected_md5 = "c58f30108f718f92721af3b95e74349a"
    
    for attempt in range(3):
        try:
            logger.info(f"Manual CIFAR-10 download attempt {attempt + 1}/3")
            
            # Download with timeout and chunk reading
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(target_file, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({progress:.1f}%)")
            
            # Verify MD5
            with open(target_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash == expected_md5:
                logger.info("Manual CIFAR-10 download successful and verified")
                return True
            else:
                logger.warning(f"MD5 mismatch: expected {expected_md5}, got {file_hash}")
                target_file.unlink()
                
        except Exception as e:
            logger.warning(f"Manual download attempt {attempt + 1} failed: {e}")
            if target_file.exists():
                target_file.unlink()
            time.sleep(2)  # Wait before retry
    
    return False

def load_cifar10_hf(seed: int = 42,
                    transform_train=None,
                    transform_test=None) -> Cifar10Data:
    """
    CIFAR-10 loader using torchvision only (no HuggingFace).
    Auto-heals corrupted downloads by cleaning cache and retrying once.
    Returns train/test datasets and raw label arrays.
    """
    if TVCIFAR10 is None:
        raise ImportError("torchvision is required for CIFAR-10 loading")

    t_train = transform_train or _train_transform()
    t_test = transform_test or _default_transform()

    root = PROJECT_ROOT / "data"
    root.mkdir(parents=True, exist_ok=True)
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            train_ds = TVCIFAR10(root=str(root), train=True, download=True, transform=t_train)
            test_ds = TVCIFAR10(root=str(root), train=False, download=True, transform=t_test)
            break  # Success
        except RuntimeError as e:
            msg = str(e).lower()
            if "file not found or corrupted" in msg or "md5" in msg:
                logger.warning(f"CIFAR-10 download corrupted (attempt {attempt + 1}/{max_attempts}): {e}")
                _cleanup_cifar10(root)
                
                # Try manual download on final attempt
                if attempt == max_attempts - 1:
                    logger.info("Attempting manual CIFAR-10 download as final fallback")
                    if _manual_cifar10_download(root):
                        # Try torchvision again with manually downloaded file
                        train_ds = TVCIFAR10(root=str(root), train=True, download=False, transform=t_train)
                        test_ds = TVCIFAR10(root=str(root), train=False, download=False, transform=t_test)
                        logger.info("CIFAR-10 manual download successful")
                        break
                    else:
                        raise RuntimeError("All CIFAR-10 download attempts failed")
                else:
                    time.sleep(5)  # Wait before retry
            else:
                raise

    # torchvision CIFAR10 exposes labels in .targets
    train_labels = np.array(getattr(train_ds, "targets", []), dtype=np.int64)
    test_labels = np.array(getattr(test_ds, "targets", []), dtype=np.int64)

    return Cifar10Data(train=train_ds, test=test_ds,
                       train_labels=train_labels, test_labels=test_labels)

# ──────────────────────── Utility: build dataloaders ──────────────────────
def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

def subset(dataset: Dataset, indices: Sequence[int]) -> Dataset:
    return Subset(dataset, indices=list(indices))

# Legacy compatibility functions
def get_cifar10_dataset(train: bool = True):
    """Legacy compatibility - load CIFAR-10 dataset (torchvision)."""
    data = load_cifar10_hf(seed=42)
    return data.train if train else data.test

def get_cifar10_test_loader(batch_size: int = 32) -> DataLoader:
    """Get CIFAR-10 test DataLoader for global evaluation (torchvision)."""
    data = load_cifar10_hf(seed=42)
    return make_loader(data.test, batch_size=batch_size, shuffle=False)

# Legacy compatibility wrapper
class CifarDataset(Dataset):
    """Legacy wrapper to expose labels as numpy array for any torch Dataset."""

    def __init__(self, base_dataset: Dataset, labels: Optional[Sequence[int]] = None):
        self.base_dataset = base_dataset
        if labels is None and hasattr(base_dataset, "targets"):
            labels = getattr(base_dataset, "targets")
        self.labels = np.array(labels if labels is not None else [], dtype=np.int64)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        return self.base_dataset[idx]


def _to_plain_ints(arr: Sequence) -> List[int]:
    return [int(x if not hasattr(x, "item") else x.item()) for x in arr]


# Global partition cache to prevent repeated JSON loading
_PARTITION_CACHE = {}
_PARTITION_FILE_CACHE = None

def _load_partition_indices(server_id: int, partition_id: int) -> List[int]:
    """Load pre-created partition indices from JSON file with caching."""
    import json
    import os
    
    global _PARTITION_FILE_CACHE
    
    parts_json = os.getenv("PARTITIONS_JSON")
    if not parts_json:
        parts_json = str(PROJECT_ROOT / "rounds" / "partitions.json")
    
    # Cache the entire partition file to avoid repeated JSON parsing
    if _PARTITION_FILE_CACHE is None:
        logger.debug(f"📂 Loading partition file {parts_json} (first time only)")
        try:
            with open(parts_json, "r") as f:
                _PARTITION_FILE_CACHE = json.load(f)
            logger.info(f"✅ Cached partition file with {len(_PARTITION_FILE_CACHE)} servers")
        except FileNotFoundError:
            logger.error(f"❌ Partition file not found: {parts_json}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading partition file {parts_json}: {e}")
            return []
    
    # Check partition cache first
    cache_key = f"s{server_id}_c{partition_id}"
    if cache_key in _PARTITION_CACHE:
        logger.debug(f"🎯 Using cached partition for server={server_id}, client={partition_id}")
        return _PARTITION_CACHE[cache_key]
    
    try:
        indices = _PARTITION_FILE_CACHE[str(server_id)][str(partition_id)]
        _PARTITION_CACHE[cache_key] = indices
        # Reduced verbosity: only log first partition load per server
        if partition_id == 0:
            logger.info(f"✅ Loading partitions for server {server_id} ({len(indices)} samples for client 0)")
        return indices
        
    except KeyError as e:
        logger.error(f"❌ Partition key not found: server={server_id}, client={partition_id}")
        logger.error(f"Available servers: {list(_PARTITION_FILE_CACHE.keys()) if _PARTITION_FILE_CACHE else 'unknown'}")
        return []
    except Exception as e:
        logger.error(f"❌ Unexpected error loading partition {server_id}/{partition_id}: {e}")
        return []


# Remove _balanced_sampler - not needed for CIFAR-10


def load_data(
    dataset_flag: str,
    partition_id: int,
    num_partitions: int,
    *,
    batch_size: int = 16,
    indices: Optional[List[int]] = None,
    server_id: Optional[int] = None,
):
    dataset_flag = dataset_flag.lower()
    if dataset_flag != "cifar10":
        raise ValueError("dataset_flag must be 'cifar10'")

    # Load CIFAR-10 datasets using torchvision
    data = load_cifar10_hf(seed=42)
    
    if indices is not None:
        idx_train = indices
    else:
        idx_train = _load_partition_indices(server_id or 0, partition_id)
    
    # For test set, map training indices to valid test indices proportionally
    train_size = len(data.train)
    test_size = len(data.test)
    
    if train_size == 0 or test_size == 0:
        logger.error(f"Empty dataset: train_size={train_size}, test_size={test_size}")
        idx_test = []
    else:
        # Map each training index to corresponding test index proportionally
        idx_test = [min(int(idx * test_size / train_size), test_size - 1) for idx in idx_train]
        # Remove duplicates while preserving order
        seen = set()
        idx_test = [x for x in idx_test if not (x in seen or seen.add(x))]
        
        logger.debug(f"Mapped {len(idx_train)} train indices to {len(idx_test)} test indices")

    # Validate indices are within bounds
    if idx_train and max(idx_train) >= train_size:
        logger.error(f"Train index out of bounds: max={max(idx_train)}, train_size={train_size}")
        idx_train = [idx for idx in idx_train if idx < train_size]
        logger.warning(f"Filtered to {len(idx_train)} valid train indices")
    
    if idx_test and max(idx_test) >= test_size:
        logger.error(f"Test index out of bounds: max={max(idx_test)}, test_size={test_size}")
        idx_test = [idx for idx in idx_test if idx < test_size]
        logger.warning(f"Filtered to {len(idx_test)} valid test indices")

    train_subset = subset(data.train, idx_train)
    test_subset = subset(data.test, idx_test)

    trainloader = make_loader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = make_loader(test_subset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, 10  # CIFAR-10 has 10 classes


# ─────────────────────── Training / eval utils ─────────────────────────────
def _make_scheduler(opt: torch.optim.Optimizer, sched_type: str, lr: float):
    sched_type = sched_type.lower()
    if sched_type == "cosine":
        total_gr = int(os.getenv("TOTAL_GLOBAL_ROUNDS", "150"))
        return CosineAnnealingLR(opt, T_max=total_gr, eta_min=lr * 0.01)
    gamma = float(os.getenv("LR_GAMMA", "0.95"))
    return StepLR(opt, step_size=1, gamma=gamma)


def train(
    net: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    *,
    lr: Optional[float] = None,
    momentum: Optional[float] = None,
    weight_decay: Optional[float] = None,
    gamma: Optional[float] = None,  # kept for API compatibility
    clip_norm: Optional[float] = None,
    prox_mu: float = 0.0,
    ref_weights: Optional[List[np.ndarray]] = None,
    global_round: int = 0,
    scaffold_enabled: bool = False,
):
    net.to(device)

    # Read required training parameters from TOML - no fallbacks
    if lr is None or momentum is None or weight_decay is None or clip_norm is None:
        cfg = toml.load(PROJECT_ROOT / "pyproject.toml")
        hierarchy_config = cfg["tool"]["flwr"]["hierarchy"]
        
        if lr is None:
            lr = hierarchy_config["lr_init"]
        if momentum is None:
            momentum = hierarchy_config["momentum"]
        if weight_decay is None:
            weight_decay = hierarchy_config["weight_decay"]
        if clip_norm is None:
            clip_norm = hierarchy_config["clip_norm"]
    
    wd = weight_decay
    clip_val = clip_norm

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    sched = (
        None
        if global_round < int(os.getenv("WARMUP_ROUNDS", "5"))  # Keep warmup default for now
        else _make_scheduler(opt, os.getenv("SCHEDULER_TYPE", "step"), lr)  # Keep scheduler default
    )

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for stability
    
    # Convert reference weights to tensors with proper validation
    ref_tensors = None
    if prox_mu > 0 and ref_weights:
        state_dict_keys = list(net.state_dict().keys())
        if len(ref_weights) == len(state_dict_keys):
            ref_tensors = []
            for key, ref_w in zip(state_dict_keys, ref_weights):
                current_param = net.state_dict()[key]
                ref_tensor = torch.tensor(ref_w, dtype=current_param.dtype, device=device)
                if ref_tensor.shape == current_param.shape:
                    ref_tensors.append(ref_tensor)
                else:
                    ref_tensors = None
                    break

    running_loss, total_batches = 0.0, 0
    net.train()
    
    for epoch in range(epochs):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.flatten().long().to(device)
            opt.zero_grad()
            
            # Forward pass with mixed precision for stability
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = net(imgs)
                loss = ce(logits, labels)
                
                # FedProx regularization term with parameter normalization
                if prox_mu > 0 and ref_tensors is not None:
                    model_params = list(net.parameters())
                    if len(ref_tensors) == len(model_params):
                        prox_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)
                        for p, w0 in zip(model_params, ref_tensors):
                            prox_loss += torch.sum((p - w0) ** 2)
                        
                        # Normalize by parameter count to prevent scale blow-ups
                        num_params = sum(p.numel() for p in model_params)
                        normalized_prox_loss = prox_loss / max(1, num_params)
                        loss = loss + (prox_mu / 2.0) * normalized_prox_loss

            loss.backward()
            
            # ENHANCED: Check for NaN gradients before SCAFFOLD correction
            nan_grads = any(torch.isnan(p.grad).any() for p in net.parameters() if p.grad is not None)
            if nan_grads:
                logger.warning(f"NaN gradients detected in epoch {epoch}, batch {batch_idx}")
                opt.zero_grad()
                continue
            
            if scaffold_enabled and hasattr(net, "_scaffold_manager"):
                net._scaffold_manager.apply_scaffold_correction(net, opt.param_groups[0]["lr"])

            # ENHANCED: Gradient clipping with diagnostic logging
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            if grad_norm > clip_val:
                logger.debug(f"Gradient clipped: norm={grad_norm:.4f} -> {clip_val}")
            
            opt.step()
            
            # ENHANCED: Check for NaN parameters after optimization step
            nan_params = any(torch.isnan(p).any() for p in net.parameters())
            if nan_params:
                logger.error(f"NaN parameters detected after optimization step in epoch {epoch}")
                return float('nan')

            running_loss += loss.item()
            total_batches += 1

    if sched:
        sched.step()
    return float(running_loss / max(total_batches, 1))


@torch.no_grad()
def test(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Enhanced evaluation with proper error handling and diagnostic logging."""
    net.to(device).eval()
    
    # Use float64 for precise accumulation
    loss_sum = 0.0
    correct = 0
    total = 0
    
    # Track min/max logits for diagnostic purposes
    min_logit = float('inf')
    max_logit = float('-inf')
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.flatten().long().to(device)
        
        # Forward pass
        outputs = net(imgs)
        
        # Diagnostic: Check for NaN/inf in logits
        logit_min = outputs.min().item()
        logit_max = outputs.max().item()
        min_logit = min(min_logit, logit_min)
        max_logit = max(max_logit, logit_max)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logger.warning(f"NaN/inf detected in logits at batch {batch_idx}: min={logit_min:.4f}, max={logit_max:.4f}")
        
        # Use F.cross_entropy with reduction="sum" for proper accumulation
        batch_loss = F.cross_entropy(outputs, labels, reduction="sum").item()
        loss_sum += batch_loss
        
        # Accuracy calculation
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Guard against division by zero
    if total == 0:
        logger.error("No samples processed during evaluation!")
        return float('nan'), 0.0
    
    # Log diagnostic information
    logger.debug(f"Eval diagnostics: logit_range=[{min_logit:.4f}, {max_logit:.4f}], total_samples={total}")
    
    avg_loss = loss_sum / total  # Per-sample loss
    accuracy = correct / total
    
    return avg_loss, accuracy


def get_weights(net: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in net.state_dict().values()]


def set_weights(net: nn.Module, weights: Sequence[np.ndarray]):
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), weights)}))


def get_transform(train: bool = True) -> Compose:
    """Return transform for CIFAR-10."""
    return _train_transform() if train else _default_transform()
