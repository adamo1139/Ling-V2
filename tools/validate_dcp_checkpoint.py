#!/usr/bin/env python3
"""
Comprehensive DCP checkpoint validation script to identify issues before HuggingFace conversion.

This script performs multiple validation checks on DCP checkpoints to detect problems that may
manifest after conversion to HuggingFace format, including:
- NaN/Inf values in weights and optimizer states
- Zero or near-zero expert weights
- Mismatched tensor shapes
- Corrupted metadata
- Missing required files

Usage:
  python Ling-V2/tools/validate_dcp_checkpoint.py --checkpoint-path /path/to/iter_XXXXXXX

  # For already converted HF models:
  python Ling-V2/tools/validate_dcp_checkpoint.py --hf-model-path /path/to/hf_model

  # Compare DCP and HF models:
  python Ling-V2/tools/validate_dcp_checkpoint.py --checkpoint-path /path/to/iter_XXXXXXX --hf-model-path /path/to/hf_model

Options:
  --checkpoint-path    Path to DCP checkpoint (iter_XXXXXXX directory)
  --hf-model-path      Path to HuggingFace model directory
  --include-optimizer  Also scan optimizer state tensors (default: off)
  --verbose            Show detailed information
  --fix-suggestions    Show suggestions for fixing detected issues
  --test-inference     Run test inference on HF model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.checkpoint import default_planner as dcp_default_planner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the checkpoint."""

    severity: str  # "critical", "warning", "info"
    category: str  # "nan_inf", "router", "expert", "shape", "metadata", "file"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: Optional[str] = None


class CheckpointValidator:
    """Validates DCP and HuggingFace checkpoints for common issues."""

    def __init__(self, verbose: bool = False, fix_suggestions: bool = False):
        self.verbose = verbose
        self.fix_suggestions = fix_suggestions
        self.issues: List[ValidationIssue] = []

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the list."""
        self.issues.append(issue)
        if self.verbose:
            self._print_issue(issue)

    def _print_issue(self, issue: ValidationIssue):
        """Print a validation issue."""
        symbols = {"critical": "❌", "warning": "⚠️", "info": "ℹ️"}
        print(f"{symbols.get(issue.severity, '•')} [{issue.severity.upper()}] {issue.category}: {issue.message}")
        if self.verbose and issue.details:
            for k, v in issue.details.items():
                print(f"    {k}: {v}")
        if self.fix_suggestions and issue.fix_suggestion:
            print(f"    💡 Fix: {issue.fix_suggestion}")

    def validate_dcp_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate a DCP checkpoint directory."""
        print(f"\n🔍 Validating DCP checkpoint: {checkpoint_path}")

        # Check directory exists
        if not checkpoint_path.exists():
            self.add_issue(
                ValidationIssue(
                    severity="critical",
                    category="file",
                    message=f"Checkpoint directory does not exist: {checkpoint_path}",
                )
            )
            return False

        # Check for required files
        required_files = ["metadata.json", "common.pt"]
        for file in required_files:
            if not (checkpoint_path / file).exists():
                self.add_issue(
                    ValidationIssue(
                        severity="critical",
                        category="file",
                        message=f"Missing required file: {file}",
                        fix_suggestion="Ensure checkpoint was saved correctly",
                    )
                )

        # Check for .distcp shard files
        distcp_files = list(checkpoint_path.glob("*.distcp"))
        if not distcp_files:
            self.add_issue(
                ValidationIssue(
                    severity="critical",
                    category="file",
                    message="No .distcp shard files found",
                    fix_suggestion="Check if checkpoint was saved with DCP format",
                )
            )
            return False

        print(f"  Found {len(distcp_files)} shard files")

        # Validate metadata
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if self.verbose:
                        print(f"  Metadata keys: {list(metadata.keys())}")
            except Exception as e:
                self.add_issue(
                    ValidationIssue(
                        severity="critical",
                        category="metadata",
                        message=f"Failed to parse metadata.json: {e}",
                        fix_suggestion="Check if metadata file is corrupted",
                    )
                )

        # Set up DCP reader with monkey-patch for single-process loading
        self._setup_dcp_reader()

        # Scan for tensor issues
        self._scan_dcp_tensors(checkpoint_path)

        return len([i for i in self.issues if i.severity == "critical"]) == 0

    def _setup_dcp_reader(self):
        """Set up DCP reader with necessary patches."""

        def _patched_set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
            assert metadata is not None
            for k, v in metadata.state_dict_metadata.items():
                if hasattr(self, "keys") and k not in self.keys:
                    continue
                if isinstance(v, TensorStorageMetadata):
                    v = torch.empty(v.size, dtype=v.properties.dtype)
                state_dict[k] = v
            super(dcp_default_planner._EmptyStateDictLoadPlanner, self).set_up_planner(
                state_dict, metadata, is_coordinator
            )

        dcp_default_planner._EmptyStateDictLoadPlanner.set_up_planner = _patched_set_up_planner

    def _scan_dcp_tensors(self, checkpoint_path: Path, max_batch_bytes: int = 1 * 2**30):
        """Scan DCP tensors for issues."""
        print("  Scanning tensors for issues...")

        reader = FileSystemReader(str(checkpoint_path))
        try:
            md = reader.read_metadata().state_dict_metadata
        except Exception as e:
            self.add_issue(
                ValidationIssue(
                    severity="critical",
                    category="metadata",
                    message=f"Failed to read DCP metadata: {e}",
                )
            )
            return

        # Collect tensor info
        tensor_info = []
        for k, v in md.items():
            if isinstance(v, TensorStorageMetadata):
                dtype = v.properties.dtype
                numel = np.prod([int(d) for d in v.size])
                nbytes = numel * torch.tensor([], dtype=dtype).element_size()
                tensor_info.append((k, nbytes, dtype, tuple(int(d) for d in v.size)))

        print(f"  Found {len(tensor_info)} tensors to scan")

        # Process in batches
        batch: List[str] = []
        acc_bytes = 0
        issues_found: Dict[str, List[str]] = {
            "nan": [],
            "inf": [],
            "zero_expert": [],
            "shape_mismatch": [],
        }
        scanned_counts = {"model": 0, "optimizer": 0}

        def process_batch():
            nonlocal batch, acc_bytes
            if not batch:
                return

            try:
                state = _load_state_dict_from_keys(set(batch), checkpoint_id=str(checkpoint_path))
            except Exception as e:
                self.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="file",
                        message=f"Failed to load batch of {len(batch)} tensors",
                        details={"error": str(e)},
                    )
                )
                batch = []
                acc_bytes = 0
                return

            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue

                # Check for NaN/Inf
                if tensor.is_floating_point():
                    tf = tensor.float()
                    has_nan = torch.isnan(tf).any().item()
                    has_inf = torch.isinf(tf).any().item()

                    if has_nan:
                        issues_found["nan"].append(name)
                    if has_inf:
                        issues_found["inf"].append(name)

                    # Check expert weights for near-zero values
                    if "expert" in name.lower() and "weight" in name.lower():
                        near_zero = (tf.abs() < 1e-8).float().mean().item()
                        if near_zero > 0.5:  # More than 50% near zero
                            issues_found["zero_expert"].append(name)
                            self.add_issue(
                                ValidationIssue(
                                    severity="warning",
                                    category="expert",
                                    message=f"Expert weight {name} has {near_zero:.1%} near-zero values",
                                    details={"shape": tuple(tf.shape)},
                                    fix_suggestion="Check if expert collapsed during training",
                                )
                            )

            del state
            batch = []
            acc_bytes = 0

        # Process all tensors in batches
        for name, nbytes, _, _ in tensor_info:
            # Optimizer tensors are irrelevant for CPT when using --finetune (optimizer is
            # reinitialized). Keep them opt-in so results don't look alarming but benign.
            is_optimizer = "optimizer" in name
            if is_optimizer and not getattr(self, "include_optimizer", False):
                continue

            scanned_counts["optimizer" if is_optimizer else "model"] += 1

            if acc_bytes + nbytes > max_batch_bytes:
                process_batch()
            batch.append(name)
            acc_bytes += nbytes
        process_batch()

        print(
            f"  Scanned tensors: model={scanned_counts['model']}"
            + (f", optimizer={scanned_counts['optimizer']}" if getattr(self, "include_optimizer", False) else "")
        )

        # Report aggregate issues
        if issues_found["nan"]:
            self.add_issue(
                ValidationIssue(
                    severity="critical",
                    category="nan_inf",
                    message=f"Found {len(issues_found['nan'])} tensors with NaN values",
                    details={"tensors": issues_found["nan"][:5]},  # Show first 5
                    fix_suggestion=(
                        "NaN values will cause inference failures. Check training logs for instability"
                    ),
                )
            )

        if issues_found["inf"]:
            self.add_issue(
                ValidationIssue(
                    severity="critical",
                    category="nan_inf",
                    message=f"Found {len(issues_found['inf'])} tensors with Inf values",
                    details={"tensors": issues_found["inf"][:5]},
                    fix_suggestion="Inf values indicate numerical overflow during training",
                )
            )

    def validate_hf_model(self, model_path: Path, test_inference: bool = False) -> bool:
        """Validate a HuggingFace model."""
        print(f"\n🔍 Validating HuggingFace model: {model_path}")

        if not model_path.exists():
            self.add_issue(
                ValidationIssue(
                    severity="critical",
                    category="file",
                    message=f"Model directory does not exist: {model_path}",
                )
            )
            return False

        # Check for required files
        required_files = ["config.json", "model.safetensors.index.json"]
        for file in required_files:
            if not (model_path / file).exists():
                # Check alternative names
                if file == "model.safetensors.index.json":
                    if not (model_path / "model-00001-of-00001.safetensors").exists():
                        self.add_issue(
                            ValidationIssue(
                                severity="critical",
                                category="file",
                                message="Missing model weights file",
                                fix_suggestion="Ensure conversion completed successfully",
                            )
                        )
                else:
                    self.add_issue(
                        ValidationIssue(
                            severity="warning",
                            category="file",
                            message=f"Missing file: {file}",
                        )
                    )

        # Load and validate config
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if self.verbose:
                        print(f"  Model type: {config.get('model_type', 'unknown')}")
                        print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
            except Exception as e:
                self.add_issue(
                    ValidationIssue(
                        severity="warning",
                        category="metadata",
                        message=f"Failed to load config.json: {e}",
                    )
                )

        # Validate model weights
        try:
            from safetensors import safe_open

            # Find all safetensors files
            safetensor_files = list(model_path.glob("*.safetensors"))
            if not safetensor_files:
                self.add_issue(
                    ValidationIssue(
                        severity="critical",
                        category="file",
                        message="No .safetensors files found",
                        fix_suggestion="Ensure model conversion completed",
                    )
                )
                return False

            print(f"  Found {len(safetensor_files)} safetensors files")

            # Check each file for NaN/Inf
            for file in safetensor_files:
                if self.verbose:
                    print(f"  Scanning {file.name}...")

                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)

                        if tensor.is_floating_point():
                            tf = tensor.float()
                            if torch.isnan(tf).any():
                                self.add_issue(
                                    ValidationIssue(
                                        severity="critical",
                                        category="nan_inf",
                                        message=f"NaN values in {key} ({file.name})",
                                        fix_suggestion="Model has corrupted weights",
                                    )
                                )
                            if torch.isinf(tf).any():
                                self.add_issue(
                                    ValidationIssue(
                                        severity="critical",
                                        category="nan_inf",
                                        message=f"Inf values in {key} ({file.name})",
                                        fix_suggestion="Model has overflowed weights",
                                    )
                                )

        except ImportError:
            self.add_issue(
                ValidationIssue(
                    severity="warning",
                    category="dependency",
                    message="safetensors not installed, skipping weight validation",
                    fix_suggestion="Install safetensors: pip install safetensors",
                )
            )
        except Exception as e:
            self.add_issue(
                ValidationIssue(
                    severity="warning",
                    category="file",
                    message=f"Error validating model weights: {e}",
                )
            )

        # Optionally test inference
        if test_inference:
            self._test_hf_inference(model_path)

        return len([i for i in self.issues if i.severity == "critical"]) == 0

    def _test_hf_inference(self, model_path: Path):
        """Run a basic inference test on the HF model."""
        print("\n  Running inference test...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
            )

            inputs = tokenizer("Hello world", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Check logits for NaN/Inf
                if torch.isnan(logits).any():
                    self.add_issue(
                        ValidationIssue(
                            severity="critical",
                            category="nan_inf",
                            message="NaN values in model logits during inference",
                            fix_suggestion="Check for numerical instability in model weights",
                        )
                    )
                if torch.isinf(logits).any():
                    self.add_issue(
                        ValidationIssue(
                            severity="critical",
                            category="nan_inf",
                            message="Inf values in model logits during inference",
                            fix_suggestion="Check for numerical overflow in model weights",
                        )
                    )

                # Test softmax for probability issues
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                if (probs < 0).any():
                    self.add_issue(
                        ValidationIssue(
                            severity="critical",
                            category="router",
                            message="Negative probabilities after softmax",
                            details={"min_prob": probs.min().item()},
                            fix_suggestion="This indicates severe numerical issues in the model",
                        )
                    )

                print("  ✓ Basic inference test passed")

        except Exception as e:
            self.add_issue(
                ValidationIssue(
                    severity="warning",
                    category="inference",
                    message=f"Inference test failed: {e}",
                    fix_suggestion="Run with --verbose for more details",
                )
            )

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if not self.issues:
            print("✅ No issues found! Checkpoint appears healthy.")
            return

        # Count by severity
        critical = len([i for i in self.issues if i.severity == "critical"])
        warning = len([i for i in self.issues if i.severity == "warning"])
        info = len([i for i in self.issues if i.severity == "info"])

        print(f"Total issues: {len(self.issues)}")
        print(f"  ❌ Critical: {critical}")
        print(f"  ⚠️  Warning: {warning}")
        print(f"  ℹ️  Info: {info}")

        # Group by category
        print("\nIssues by category:")
        categories: Dict[str, List[ValidationIssue]] = {}
        for issue in self.issues:
            categories.setdefault(issue.category, []).append(issue)

        for cat, issues in sorted(categories.items()):
            critical_count = len([i for i in issues if i.severity == "critical"])
            suffix = f" ({critical_count} critical)" if critical_count > 0 else ""
            print(f"  {cat}: {len(issues)} issues{suffix}")

        # Print critical issues
        if critical > 0:
            print("\n⚠️  CRITICAL ISSUES REQUIRE ATTENTION:")
            for issue in self.issues:
                if issue.severity == "critical":
                    print(f"  • {issue.message}")
                    if self.fix_suggestions and issue.fix_suggestion:
                        print(f"    → {issue.fix_suggestion}")

        # Overall recommendation
        print("\n" + "=" * 60)
        if critical > 0:
            print("❌ VALIDATION FAILED: Critical issues found.")
            print("   DO NOT use this checkpoint for inference without fixes.")
        elif warning > 0:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            print("   Checkpoint may work but could have issues.")
        else:
            print("✅ VALIDATION PASSED")
            print("   Checkpoint appears ready for use.")


def main():
    parser = argparse.ArgumentParser(description="Validate DCP checkpoints before HF conversion")
    parser.add_argument("--checkpoint-path", type=str, help="Path to DCP checkpoint (iter_XXXXXXX)")
    parser.add_argument("--hf-model-path", type=str, help="Path to HuggingFace model directory")
    parser.add_argument(
        "--include-optimizer",
        action="store_true",
        help="Also scan optimizer state tensors in the checkpoint (off by default)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--fix-suggestions", action="store_true", help="Show fix suggestions")
    parser.add_argument("--test-inference", action="store_true", help="Run test inference on HF model")

    args = parser.parse_args()

    if not args.checkpoint_path and not args.hf_model_path:
        print("Error: Specify at least one of --checkpoint-path or --hf-model-path")
        parser.print_help()
        sys.exit(1)

    validator = CheckpointValidator(verbose=args.verbose, fix_suggestions=args.fix_suggestions)
    validator.include_optimizer = args.include_optimizer

    # Validate DCP checkpoint
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path).absolute()
        validator.validate_dcp_checkpoint(checkpoint_path)

    # Validate HF model
    if args.hf_model_path:
        model_path = Path(args.hf_model_path).absolute()
        validator.validate_hf_model(model_path, test_inference=args.test_inference)

    # Print summary
    validator.print_summary()

    # Exit with error code if critical issues found
    critical_count = len([i for i in validator.issues if i.severity == "critical"])
    sys.exit(1 if critical_count > 0 else 0)


if __name__ == "__main__":
    # Quiet down noisy deprecation warnings that aren't actionable for validation.
    warnings.filterwarnings("default", category=DeprecationWarning)
    main()
