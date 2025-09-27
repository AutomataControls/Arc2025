"""
ARC Prize 2025 - Colab Training Monitor Integration
Add this to your training script for real-time progress tracking
"""

import json
import numpy as np
from datetime import datetime
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ColabTrainingMonitor:
    def __init__(self, models=['minerva', 'atlas', 'iris', 'chronos', 'prometheus']):
        self.models = models
        self.histories = {m: {'epochs': [], 'train_acc': [], 'val_acc': [], 
                             'train_loss': [], 'val_loss': []} for m in models}
        self.best_scores = {m: 0 for m in models}
        self.start_time = datetime.now()
        self.target = 0.85
        self.milestones_hit = {m: set() for m in models}
        
    def update(self, model_name, epoch, metrics):
        """Update with new epoch metrics"""
        h = self.histories[model_name]
        h['epochs'].append(epoch)
        h['train_acc'].append(metrics['train_acc'])
        h['val_acc'].append(metrics['val_acc'])
        h['train_loss'].append(metrics['train_loss'])
        h['val_loss'].append(metrics['val_loss'])
        
        # Update best
        if metrics['val_acc'] > self.best_scores[model_name]:
            self.best_scores[model_name] = metrics['val_acc']
            
        # Check milestones
        for milestone in [0.70, 0.75, 0.80, 0.85, 0.90]:
            if metrics['val_acc'] >= milestone and milestone not in self.milestones_hit[model_name]:
                self.milestones_hit[model_name].add(milestone)
                self._milestone_alert(model_name, milestone)
    
    def _milestone_alert(self, model, milestone):
        """Show milestone achievement"""
        alerts = {
            0.70: "✓ Good progress - 70% reached!",
            0.75: "⚡ Excellent - 75% achieved!",
            0.80: "🔥 Amazing - 80% accuracy!",
            0.85: "🏆 GRAND PRIZE THRESHOLD - 85% ACHIEVED! $700K UNLOCKED!",
            0.90: "🚀 EXCEPTIONAL - 90% accuracy!"
        }
        print(f"\n{'='*60}")
        print(f"🎯 {model.upper()} - {alerts[milestone]}")
        print(f"{'='*60}\n")
    
    def show_dashboard(self):
        """Display live dashboard in Colab"""
        clear_output(wait=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracies', 'Training Progress to 85%', 
                           'Best Scores Ranking', 'Loss Curves'),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]]
        )
        
        # 1. Accuracy curves
        colors = {'minerva': '#9B59B6', 'atlas': '#3498DB', 'iris': '#E74C3C', 
                 'chronos': '#F39C12', 'prometheus': '#27AE60'}
        
        for model in self.models:
            if self.histories[model]['epochs']:
                # Val accuracy
                fig.add_trace(go.Scatter(
                    x=self.histories[model]['epochs'],
                    y=[a*100 for a in self.histories[model]['val_acc']],
                    mode='lines+markers',
                    name=f'{model.upper()}',
                    line=dict(color=colors[model], width=3),
                    marker=dict(size=4)
                ), row=1, col=1)
        
        # Target line
        if any(self.histories[m]['epochs'] for m in self.models):
            max_epoch = max(max(self.histories[m]['epochs']) for m in self.models if self.histories[m]['epochs'])
            fig.add_trace(go.Scatter(
                x=[0, max_epoch],
                y=[85, 85],
                mode='lines',
                name='$700k Target',
                line=dict(color='red', width=2, dash='dash')
            ), row=1, col=1)
        
        # 2. Best scores bar chart
        best_models = sorted([(m, self.best_scores[m]*100) for m in self.models], 
                           key=lambda x: x[1], reverse=True)
        
        fig.add_trace(go.Bar(
            x=[m[0].upper() for m in best_models],
            y=[m[1] for m in best_models],
            marker_color=[colors[m[0]] for m in best_models],
            text=[f'{m[1]:.1f}%' for m in best_models],
            textposition='auto',
        ), row=2, col=1)
        
        # 3. Progress gauge
        best_overall = max(self.best_scores.values())
        progress = (best_overall / self.target) * 100
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = best_overall * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': 85, 'suffix': '%'},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen" if best_overall >= 0.85 else "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            },
            title = {'text': f"Best: {max(best_models, key=lambda x: x[1])[0].upper()}"}
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"ARC Prize 2025 Training Monitor - {datetime.now().strftime('%H:%M:%S')}",
            title_font_size=20
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Best Accuracy (%)", row=2, col=1)
        
        # Show
        fig.show()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print text summary below dashboard"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        # Time elapsed
        elapsed = datetime.now() - self.start_time
        print(f"Time Elapsed: {elapsed}")
        
        # Model rankings
        rankings = sorted([(m, self.best_scores[m]) for m in self.models], 
                         key=lambda x: x[1], reverse=True)
        
        print("\nModel Rankings:")
        for i, (model, score) in enumerate(rankings, 1):
            status = "🏆 PRIZE ELIGIBLE" if score >= 0.85 else f"{(0.85-score)*100:.1f}% to go"
            print(f"{i}. {model.upper()}: {score*100:.2f}% - {status}")
        
        # Estimate time to 85%
        if rankings[0][1] < 0.85:
            self._estimate_completion(rankings[0][0])
    
    def _estimate_completion(self, best_model):
        """Estimate time to reach 85%"""
        history = self.histories[best_model]
        if len(history['val_acc']) < 5:
            return
        
        # Simple linear estimate from last 5 epochs
        recent_accs = history['val_acc'][-5:]
        improvement_rate = (recent_accs[-1] - recent_accs[0]) / 4
        
        if improvement_rate > 0:
            epochs_needed = (0.85 - history['val_acc'][-1]) / improvement_rate
            print(f"\nEstimated epochs to 85%: {int(epochs_needed)}")
        else:
            print("\n⚠️ No recent improvement - consider adjusting hyperparameters")

# Easy integration function
def setup_colab_monitor():
    """Setup monitor for Colab - call this once"""
    monitor = ColabTrainingMonitor()
    
    # CSS for better display
    display(HTML("""
    <style>
    .milestone-alert {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        text-align: center;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    </style>
    """))
    
    return monitor

# Modified training loop integration
def update_monitor_in_loop(monitor, trainer, model_name, epoch, history):
    """Call this in your existing training loop"""
    monitor.update(model_name, epoch, {
        'train_acc': history['train_acc'][-1],
        'val_acc': history['val_acc'][-1],
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1]
    })
    
    # Show dashboard every 3 epochs
    if epoch % 3 == 0:
        monitor.show_dashboard()
    
    # Return True if target achieved
    return history['val_acc'][-1] >= 0.85

# Example integration:
"""
# Add to your Colab training script:

# At the start:
from colab_monitor_integration import setup_colab_monitor, update_monitor_in_loop
monitor = setup_colab_monitor()

# In your training loop (after each epoch):
target_reached = update_monitor_in_loop(monitor, trainer, model_name, epoch, trainer.history)
if target_reached:
    print("🎉 GRAND PRIZE TARGET ACHIEVED! 🎉")
    break
"""