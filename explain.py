"""
Explainability analysis for dementia detection models.
Uses SHAP and Integrated Gradients to explain model predictions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig
import hydra
import json
import warnings
warnings.filterwarnings('ignore')

# Import explainability tools
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from captum.attr import IntegratedGradients, LayerGradientXActivation
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: Captum not available. Install with: pip install captum")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from train import DementiaDetectionModule
from src.data.dataset import DementiaDataset

class ModelExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize model explainer."""
        self.model = model.to(device)
        self.model.eval()
        self.cfg = cfg
        self.device = device
        
        print(f"[EXPLAIN] Initializing explainer with method: {cfg.training.explain.method}")
        print(f"[EXPLAIN] Using device: {device}")
        
        # Set up explainability methods
        self.method = cfg.training.explain.method
        
        if self.method == 'shap' and SHAP_AVAILABLE:
            # SHAP requires background data - we'll set it up when we have real data
            self.explainer = None
            print("[EXPLAIN] SHAP explainer will be initialized with background data")
        elif self.method == 'integrated_gradients' and CAPTUM_AVAILABLE:
            # Integrated Gradients for the full model
            self.explainer = IntegratedGradients(self.model)
            print("[EXPLAIN] Integrated Gradients explainer initialized")
        else:
            print(f"[EXPLAIN] Warning: Explainability method '{self.method}' not available")
            self.explainer = None
    
    def initialize_shap_explainer(self, background_data: Dict[str, torch.Tensor]) -> None:
        """Initialize SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            return
        
        print("[EXPLAIN] Initializing SHAP with background data...")
        # Use a subset of background data
        self.background_data = {
            k: v[:self.cfg.training.explain.num_background_samples].to(self.device)
            for k, v in background_data.items()
            if isinstance(v, torch.Tensor)
        }
        
        # Create a wrapper function for SHAP
        def model_predict(batch_dict):
            with torch.no_grad():
                return self.model(batch_dict).detach().cpu().numpy()
        
        self.shap_predict = model_predict
        print("[EXPLAIN] SHAP explainer initialized")
    
    def explain_sample(
        self,
        sample: Dict[str, torch.Tensor],
        target_class: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Get feature attributions for a single sample.
        
        Args:
            sample: Dictionary with input tensors
            target_class: Class to explain (0=no dementia, 1=dementia)
            
        Returns:
            Dictionary with attribution values for each feature type
        """
        # Move sample to device and ensure correct shapes
        sample_gpu = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                # Remove extra dimensions and add batch dimension if needed
                if v.dim() == 0:  # Scalar
                    sample_gpu[k] = v
                elif k == 'waveform':
                    # Waveform should be (batch, samples) = (1, samples)
                    v_squeezed = v.squeeze()  # Remove all extra dims
                    sample_gpu[k] = v_squeezed.unsqueeze(0).to(self.device)
                elif k == 'attention_mask':
                    # Attention mask should be (batch, samples)
                    v_squeezed = v.squeeze()
                    sample_gpu[k] = v_squeezed.unsqueeze(0).to(self.device)
                else:
                    # Other tensors: add batch dim if needed
                    if v.dim() == 1:
                        sample_gpu[k] = v.unsqueeze(0).to(self.device)
                    else:
                        sample_gpu[k] = v.to(self.device)
            else:
                sample_gpu[k] = v
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(sample_gpu)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        print(f"[EXPLAIN] Prediction: class={pred_class}, confidence={confidence:.3f}")
        
        attributions = {}
        
        # Always compute attributions regardless of method
        if 'waveform' in sample_gpu and CAPTUM_AVAILABLE:
            try:
                print(f"[EXPLAIN] Computing waveform attributions...")
                
                # Create a wrapper function that IntegratedGradients can use
                def model_forward_wrapper(waveform_input):
                    """Wrapper that takes only waveform and uses stored batch data."""
                    batch_copy = sample_gpu.copy()
                    batch_copy['waveform'] = waveform_input
                    return self.model(batch_copy)
                
                # Create baseline (zeros)
                baseline_waveform = torch.zeros_like(sample_gpu['waveform'])
                
                # Use Integrated Gradients with wrapper - use fewer steps to save memory
                from captum.attr import IntegratedGradients
                ig_explainer = IntegratedGradients(model_forward_wrapper)
                
                waveform_attr = ig_explainer.attribute(
                    sample_gpu['waveform'],
                    baseline_waveform,
                    target=target_class,
                    n_steps=10,  # Reduced from 50 to save memory
                    internal_batch_size=1  # Process one step at a time
                )
                attributions['waveform'] = waveform_attr.squeeze().detach().cpu().numpy()
                print(f"[EXPLAIN] Waveform attributions shape: {attributions['waveform'].shape}")
                
            except Exception as e:
                print(f"[EXPLAIN] IntegratedGradients failed (likely out of memory): {str(e)[:100]}")
                # Use simple gradient as fallback - much less memory
                try:
                    print(f"[EXPLAIN] Trying gradient fallback...")
                    waveform_input = sample_gpu['waveform'].clone().detach().requires_grad_(True)
                    sample_gpu['waveform'] = waveform_input
                    logits = self.model(sample_gpu)
                    logits[0, target_class].backward()
                    attributions['waveform'] = waveform_input.grad.squeeze().detach().cpu().numpy()
                    print(f"[EXPLAIN] Using gradient fallback, shape: {attributions['waveform'].shape}")
                except Exception as e2:
                    print(f"[EXPLAIN] Gradient fallback also failed: {e2}")
                    # Last resort: use absolute waveform values
                    attributions['waveform'] = np.abs(sample['waveform'].squeeze().numpy())
                    print(f"[EXPLAIN] Using absolute values fallback")
        
        if 'text_embeddings' in sample_gpu:
            try:
                print(f"[EXPLAIN] Computing text embedding importance...")
                # For text embeddings, use absolute values as importance
                text_attr = sample_gpu['text_embeddings'].squeeze().abs().detach().cpu().numpy()
                attributions['text_embeddings'] = text_attr
                print(f"[EXPLAIN] Text attributions shape: {attributions['text_embeddings'].shape}")
            except Exception as e:
                print(f"[EXPLAIN] Error with text embeddings: {e}")
        
        attributions['prediction'] = {
            'class': pred_class,
            'confidence': confidence,
            'probabilities': probs.squeeze().detach().cpu().numpy()
        }
        
        return attributions
    
    def plot_waveform_attributions(
        self,
        waveform: np.ndarray,
        attributions: np.ndarray,
        sample_rate: int,
        prediction: Dict,
        subject: str,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot waveform with attribution overlay."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Time axis
        time = np.arange(len(waveform)) / sample_rate
        
        # Plot waveform
        ax1.plot(time, waveform, alpha=0.7, linewidth=0.5)
        ax1.set_title(f'Audio Waveform - Subject: {subject}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Plot attributions (importance over time)
        # Downsample attributions to match waveform if needed
        if len(attributions) != len(waveform):
            # Average attributions in windows
            window_size = len(waveform) // 100
            if window_size > 0:
                attr_downsampled = np.array([
                    attributions[i:i+window_size].mean()
                    for i in range(0, len(attributions), window_size)
                ])
                time_attr = np.linspace(0, time[-1], len(attr_downsampled))
            else:
                attr_downsampled = attributions[:len(waveform)]
                time_attr = time[:len(attr_downsampled)]
        else:
            attr_downsampled = attributions
            time_attr = time
        
        # Normalize attributions for visualization
        attr_norm = (attr_downsampled - attr_downsampled.min()) / (attr_downsampled.max() - attr_downsampled.min() + 1e-8)
        
        ax2.fill_between(time_attr, 0, attr_norm, alpha=0.6, color='orange')
        ax2.set_title(f'Feature Importance (Prediction: {"Dementia" if prediction["class"] == 1 else "No Dementia"}, Confidence: {prediction["confidence"]:.2%})')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Importance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[EXPLAIN] Saved plot to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_text_feature_importance(
        self,
        text_embeddings: np.ndarray,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot text embedding importance."""
        plt.figure(figsize=(12, 4))
        
        # Plot top dimensions by importance
        top_k = min(50, len(text_embeddings))
        top_indices = np.argsort(np.abs(text_embeddings))[-top_k:]
        
        plt.bar(range(top_k), text_embeddings[top_indices])
        plt.title(f'Top {top_k} Text Embedding Dimensions by Importance')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[EXPLAIN] Saved plot to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_summary(
        self,
        predictions: List[Dict],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot summary of predictions across samples."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract data
        true_labels = [p['true_label'] for p in predictions]
        pred_labels = [p['prediction']['class'] for p in predictions]
        confidences = [p['prediction']['confidence'] for p in predictions]
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Dementia', 'Dementia'],
                   yticklabels=['No Dementia', 'Dementia'])
        ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy_score(true_labels, pred_labels):.2%}')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Confidence distribution
        ax2.hist([confidences[i] for i in range(len(confidences)) if pred_labels[i] == 0],
                alpha=0.5, label='No Dementia', bins=20)
        ax2.hist([confidences[i] for i in range(len(confidences)) if pred_labels[i] == 1],
                alpha=0.5, label='Dementia', bins=20)
        ax2.set_title('Prediction Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[EXPLAIN] Saved summary plot to {save_path}")
            plt.close()
        else:
            plt.show()

def analyze_model(
    cfg: DictConfig,
    checkpoint_path: str,
    num_samples: int = 10,
    output_dir: str = 'explain_output'
) -> None:
    """
    Run explainability analysis on trained model.
    
    Args:
        cfg: Hydra config
        checkpoint_path: Path to model checkpoint
        num_samples: Number of test samples to analyze
        output_dir: Directory to save explanation outputs
    """
    print("="*80)
    print("EXPLAINABILITY ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[EXPLAIN] Output directory: {output_path}")
    
    # Load model from checkpoint
    print(f"[EXPLAIN] Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = DementiaDetectionModule(cfg)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        # Some checkpoints may store the raw state_dict directly.
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model.load_state_dict(state_dict)
    model.eval()
    print("[EXPLAIN] Model loaded successfully")
    
    # Create explainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    explainer = ModelExplainer(model, cfg, device=device)
    
    # Load test dataset
    print("[EXPLAIN] Loading test dataset...")
    with open(cfg.data.paths.splits) as f:
        splits = json.load(f)
    
    test_dataset = DementiaDataset(cfg, splits['test'], mode='test')
    print(f"[EXPLAIN] Test dataset size: {len(test_dataset)} samples")
    
    # Analyze samples
    predictions = []
    num_to_analyze = min(num_samples, len(test_dataset))
    
    print(f"\n[EXPLAIN] Analyzing {num_to_analyze} samples...")
    print("="*80)
    
    for i in range(num_to_analyze):
        print(f"\n[EXPLAIN] Sample {i+1}/{num_to_analyze}")
        sample = test_dataset[i]
        subject = sample['subject']
        true_label = sample['label'].item()
        
        print(f"[EXPLAIN] Subject: {subject}, True label: {true_label} ({'Dementia' if true_label == 1 else 'No Dementia'})")
        
        # Get attributions
        attributions = explainer.explain_sample(sample, target_class=true_label)
        
        # Plot waveform attributions
        if 'waveform' in attributions:
            waveform = sample['waveform'].squeeze().numpy()
            explainer.plot_waveform_attributions(
                waveform,
                attributions['waveform'],
                cfg.data.preprocessing.sample_rate,
                attributions['prediction'],
                subject,
                save_path=output_path / f'sample_{i:03d}_{subject}_waveform.png'
            )
        
        # Plot text feature importance
        if 'text_embeddings' in attributions:
            explainer.plot_text_feature_importance(
                attributions['text_embeddings'],
                save_path=output_path / f'sample_{i:03d}_{subject}_text.png'
            )
        
        # Store prediction info
        predictions.append({
            'subject': subject,
            'true_label': true_label,
            'prediction': attributions['prediction']
        })
        
        print(f"[EXPLAIN] Completed analysis for {subject}")
    
    # Plot overall summary
    print("\n[EXPLAIN] Generating summary plots...")
    explainer.plot_prediction_summary(
        predictions,
        save_path=output_path / 'prediction_summary.png'
    )
    
    # Save predictions to CSV
    pred_df = pd.DataFrame([
        {
            'subject': p['subject'],
            'true_label': p['true_label'],
            'predicted_label': p['prediction']['class'],
            'confidence': p['prediction']['confidence'],
            'prob_no_dementia': p['prediction']['probabilities'][0],
            'prob_dementia': p['prediction']['probabilities'][1],
            'correct': p['true_label'] == p['prediction']['class']
        }
        for p in predictions
    ])
    pred_df.to_csv(output_path / 'predictions.csv', index=False)
    print(f"[EXPLAIN] Saved predictions to {output_path / 'predictions.csv'}")
    
    # Print summary statistics
    accuracy = (pred_df['correct'].sum() / len(pred_df)) * 100
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Samples analyzed: {len(predictions)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Mean confidence: {pred_df['confidence'].mean():.2%}")
    print(f"\nPrediction distribution:")
    print(pred_df['predicted_label'].value_counts())
    
    # Log to wandb if enabled
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        print("\n[EXPLAIN] Logging to W&B...")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags + ['explainability']
        )
        
        # Log summary
        wandb.log({
            'explain/accuracy': accuracy,
            'explain/num_samples': len(predictions),
            'explain/mean_confidence': pred_df['confidence'].mean()
        })
        
        # Log plots
        for img in output_path.glob('*.png'):
            wandb.log({f'explain/{img.stem}': wandb.Image(str(img))})
        
        wandb.finish()
        print("[EXPLAIN] W&B logging complete")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - Results saved to: {output_path}")
    print("="*80)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for explainability analysis.
    
    Usage:
        python explain.py +checkpoint_path=checkpoints/best_model.ckpt +num_samples=10
    """
    import sys
    
    # Get checkpoint path from command line or use latest
    checkpoint_path = None
    num_samples = 10
    output_dir = 'explain_output'
    
    for arg in sys.argv[1:]:
        if 'checkpoint_path=' in arg:
            checkpoint_path = arg.split('=')[1]
        elif 'model_checkpoint=' in arg:
            # Accept alias used in some notebooks/commands.
            checkpoint_path = arg.split('=')[1]
        elif 'num_samples=' in arg:
            num_samples = int(arg.split('=')[1])
        elif 'output_dir=' in arg:
            output_dir = arg.split('=')[1]
    
    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.ckpt'))
            if checkpoints:
                checkpoint_path = str(sorted(checkpoints)[-1])
                print(f"[EXPLAIN] Using latest checkpoint: {checkpoint_path}")
            else:
                print("[EXPLAIN] ERROR: No checkpoints found in checkpoints/")
                print("[EXPLAIN] Please specify checkpoint_path=<path>")
                sys.exit(1)
        else:
            print("[EXPLAIN] ERROR: checkpoints/ directory not found")
            print("[EXPLAIN] Please train a model first or specify checkpoint_path=<path>")
            sys.exit(1)

    checkpoint_path = str(Path(checkpoint_path).expanduser())
    if not Path(checkpoint_path).exists():
        print(f"[EXPLAIN] ERROR: Checkpoint not found: {checkpoint_path}")
        print("[EXPLAIN] Please verify the path or use a file inside checkpoints/")
        sys.exit(1)
    
    # Run analysis
    try:
        analyze_model(
            cfg,
            checkpoint_path=checkpoint_path,
            num_samples=num_samples,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\n[EXPLAIN] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()