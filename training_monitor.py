#!/usr/bin/env python3
"""
ARC Prize 2025 Training Monitor
Tracks model performance toward 85% accuracy target for $700k grand prize
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

class TrainingMonitor:
    def __init__(self, target_accuracy: float = 0.85, checkpoint_dir: str = "./checkpoints"):
        self.target_accuracy = target_accuracy
        self.checkpoint_dir = checkpoint_dir
        self.milestones = [0.70, 0.75, 0.80, 0.85, 0.90]  # Accuracy milestones
        self.milestone_rewards = {
            0.70: "Good progress!",
            0.75: "Halfway to grand prize!",
            0.80: "Getting close to $700k!",
            0.85: "🎉 GRAND PRIZE THRESHOLD REACHED! $700k unlocked!",
            0.90: "🚀 EXCEPTIONAL! Top placement secured!"
        }
        self.model_histories = {}
        self.milestone_achieved = {model: set() for model in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']}
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def update_model_progress(self, model_name: str, epoch: int, train_acc: float, 
                            val_acc: float, train_loss: float, val_loss: float):
        """Update model training progress"""
        if model_name not in self.model_histories:
            self.model_histories[model_name] = {
                'epochs': [],
                'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': [],
                'best_val_acc': 0,
                'best_epoch': 0,
                'start_time': time.time()
            }
        
        history = self.model_histories[model_name]
        history['epochs'].append(epoch)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Update best accuracy
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
        
        # Check milestones
        for milestone in self.milestones:
            if val_acc >= milestone and milestone not in self.milestone_achieved[model_name]:
                self.milestone_achieved[model_name].add(milestone)
                self._trigger_milestone_alert(model_name, milestone, val_acc)
    
    def _trigger_milestone_alert(self, model_name: str, milestone: float, current_acc: float):
        """Alert when milestone is reached"""
        message = f"""
🎯 MILESTONE ACHIEVED: {model_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy: {current_acc*100:.2f}% (Milestone: {milestone*100}%)
{self.milestone_rewards[milestone]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        print(message)
        
        # Save checkpoint for milestone
        checkpoint_path = f"{self.checkpoint_dir}/{model_name}_{int(milestone*100)}pct_checkpoint.json"
        self.save_checkpoint(model_name, checkpoint_path)
        
    def estimate_time_to_target(self, model_name: str) -> Tuple[float, str]:
        """Estimate time to reach 85% accuracy based on current progress"""
        if model_name not in self.model_histories:
            return -1, "No data"
        
        history = self.model_histories[model_name]
        if len(history['val_acc']) < 3:
            return -1, "Insufficient data"
        
        # Use exponential smoothing for prediction
        current_acc = history['val_acc'][-1]
        if current_acc >= self.target_accuracy:
            return 0, "Target achieved!"
        
        # Calculate average improvement per epoch
        recent_epochs = min(5, len(history['val_acc']) - 1)
        recent_improvement = (history['val_acc'][-1] - history['val_acc'][-recent_epochs-1]) / recent_epochs
        
        if recent_improvement <= 0:
            return -1, "No improvement"
        
        epochs_needed = (self.target_accuracy - current_acc) / recent_improvement
        time_per_epoch = (time.time() - history['start_time']) / len(history['epochs'])
        estimated_seconds = epochs_needed * time_per_epoch
        
        return estimated_seconds, self._format_time(estimated_seconds)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time"""
        if seconds < 0:
            return "Unknown"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def detect_overfitting(self, model_name: str, window: int = 5) -> bool:
        """Detect if model is overfitting"""
        if model_name not in self.model_histories:
            return False
        
        history = self.model_histories[model_name]
        if len(history['val_acc']) < window + 1:
            return False
        
        # Check if validation accuracy is decreasing while training accuracy increases
        recent_val = history['val_acc'][-window:]
        recent_train = history['train_acc'][-window:]
        
        val_trend = np.polyfit(range(window), recent_val, 1)[0]
        train_trend = np.polyfit(range(window), recent_train, 1)[0]
        
        # Overfitting if val decreasing and train increasing
        return val_trend < -0.001 and train_trend > 0.001
    
    def get_recommendations(self, model_name: str) -> List[str]:
        """Get training recommendations based on current progress"""
        recommendations = []
        
        if model_name not in self.model_histories:
            return ["No training data available"]
        
        history = self.model_histories[model_name]
        current_acc = history['val_acc'][-1] if history['val_acc'] else 0
        
        # Check progress rate
        if len(history['val_acc']) > 5:
            recent_improvement = history['val_acc'][-1] - history['val_acc'][-6]
            if recent_improvement < 0.01:  # Less than 1% improvement in 5 epochs
                recommendations.append("⚠️ Slow progress detected. Consider:")
                recommendations.append("  - Adjusting learning rate")
                recommendations.append("  - Trying different optimizer")
                recommendations.append("  - Increasing model complexity")
        
        # Check overfitting
        if self.detect_overfitting(model_name):
            recommendations.append("⚠️ Overfitting detected! Consider:")
            recommendations.append("  - Adding more regularization")
            recommendations.append("  - Increasing dropout")
            recommendations.append("  - Data augmentation")
        
        # Distance to target
        if current_acc < 0.70:
            recommendations.append(f"📊 Current: {current_acc*100:.1f}% | Need: {(0.85-current_acc)*100:.1f}% more for grand prize")
        elif current_acc < 0.85:
            recommendations.append(f"🎯 Getting close! Only {(0.85-current_acc)*100:.1f}% to grand prize!")
        else:
            recommendations.append("🏆 GRAND PRIZE THRESHOLD ACHIEVED!")
        
        return recommendations
    
    def generate_progress_report(self) -> str:
        """Generate comprehensive progress report"""
        report = f"""
═══════════════════════════════════════════════════════════════════
            ARC PRIZE 2025 TRAINING PROGRESS REPORT
                Target: 85% for $700,000 Grand Prize
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════

"""
        
        # Sort models by best accuracy
        model_performances = []
        for model_name, history in self.model_histories.items():
            if history['val_acc']:
                best_acc = history['best_val_acc']
                current_acc = history['val_acc'][-1]
                model_performances.append((model_name, best_acc, current_acc))
        
        model_performances.sort(key=lambda x: x[1], reverse=True)
        
        # Model summaries
        for rank, (model_name, best_acc, current_acc) in enumerate(model_performances, 1):
            history = self.model_histories[model_name]
            est_time, time_str = self.estimate_time_to_target(model_name)
            
            status = "🏆 TARGET ACHIEVED!" if best_acc >= 0.85 else f"📈 {time_str} to target"
            
            report += f"""
{rank}. {model_name.upper()}
   Best Accuracy: {best_acc*100:.2f}% (epoch {history['best_epoch']})
   Current: {current_acc*100:.2f}%
   Status: {status}
   Milestones: {', '.join([f"{int(m*100)}%" for m in sorted(self.milestone_achieved[model_name])])}
"""
        
        # Overall progress
        report += f"""
───────────────────────────────────────────────────────────────────
OVERALL PROGRESS TO GRAND PRIZE
───────────────────────────────────────────────────────────────────
"""
        
        if model_performances:
            best_model = model_performances[0]
            progress_pct = (best_model[1] / 0.85) * 100
            report += f"""
Best Model: {best_model[0].upper()} at {best_model[1]*100:.2f}%
Progress to $700k: {progress_pct:.1f}%
[{'█' * int(progress_pct/5)}{'-' * (20 - int(progress_pct/5))}] {min(progress_pct, 100):.0f}%
"""
        
        # Recommendations
        report += f"""
───────────────────────────────────────────────────────────────────
RECOMMENDATIONS
───────────────────────────────────────────────────────────────────
"""
        
        for model_name in model_performances[:3]:  # Top 3 models
            recs = self.get_recommendations(model_name[0])
            if recs:
                report += f"\n{model_name[0].upper()}:\n"
                for rec in recs:
                    report += f"{rec}\n"
        
        return report
    
    def save_checkpoint(self, model_name: str, filepath: str):
        """Save model checkpoint data"""
        if model_name in self.model_histories:
            checkpoint = {
                'model_name': model_name,
                'history': self.model_histories[model_name],
                'milestones': list(self.milestone_achieved[model_name]),
                'timestamp': datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
    
    def create_live_dashboard(self, output_path: str = "training_dashboard.html"):
        """Create interactive dashboard with all models"""
        fig = go.Figure()
        
        # Add traces for each model
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for i, (model_name, history) in enumerate(self.model_histories.items()):
            if history['val_acc']:
                # Validation accuracy
                fig.add_trace(go.Scatter(
                    x=history['epochs'],
                    y=[acc * 100 for acc in history['val_acc']],
                    mode='lines+markers',
                    name=f'{model_name.upper()} Val',
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6)
                ))
                
                # Training accuracy (thinner, dashed)
                fig.add_trace(go.Scatter(
                    x=history['epochs'],
                    y=[acc * 100 for acc in history['train_acc']],
                    mode='lines',
                    name=f'{model_name.upper()} Train',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.5
                ))
        
        # Add target line at 85%
        fig.add_hline(y=85, line_dash="dash", line_color="red", line_width=2,
                      annotation_text="$700k Grand Prize Target (85%)")
        
        # Add milestone lines
        for milestone in [70, 75, 80]:
            fig.add_hline(y=milestone, line_dash="dot", line_color="gray", 
                         opacity=0.5, line_width=1)
        
        fig.update_layout(
            title={
                'text': 'ARC Prize 2025 - Live Training Progress',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            hovermode='x unified',
            width=1200,
            height=700,
            yaxis=dict(range=[0, 100]),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.write_html(output_path)
        return output_path

# Example usage in training loop
def integrate_with_training(monitor: TrainingMonitor, trainer, model_name: str, 
                          epoch: int, train_loss: float, train_acc: float, 
                          val_loss: float, val_acc: float):
    """Call this after each epoch in your training loop"""
    
    # Update monitor
    monitor.update_model_progress(
        model_name=model_name,
        epoch=epoch,
        train_acc=train_acc,
        val_acc=val_acc,
        train_loss=train_loss,
        val_loss=val_loss
    )
    
    # Print progress every 5 epochs
    if epoch % 5 == 0:
        print(f"\n{monitor.generate_progress_report()}")
    
    # Save dashboard
    monitor.create_live_dashboard()
    
    # Check if we should stop early (achieved target)
    if val_acc >= 0.85:
        print(f"\n🎉 {model_name.upper()} ACHIEVED GRAND PRIZE TARGET! 🎉")
        return True
    
    return False

# Colab integration script
if __name__ == "__main__":
    print("""
ARC Prize 2025 Training Monitor
═══════════════════════════════════════════════════════
Target: 85% accuracy to unlock $700,000 grand prize

To use in your training script:

    from training_monitor import TrainingMonitor, integrate_with_training
    
    monitor = TrainingMonitor(target_accuracy=0.85)
    
    # In your training loop:
    should_stop = integrate_with_training(
        monitor, trainer, model_name, epoch, 
        train_loss, train_acc, val_loss, val_acc
    )

Dashboard will be saved to: training_dashboard.html
""")