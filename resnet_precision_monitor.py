import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
import struct
import pandas as pd
from IPython.display import display, HTML, clear_output
import time

class FloatAnalyzer:
    """Float 정밀도별 값의 분포와 특성을 분석하는 클래스"""
    
    @staticmethod
    def analyze_float_properties(tensor, dtype_name):
        """텐서의 float 특성을 분석"""
        if tensor.numel() == 0:
            return {}
        
        # 기본 통계
        stats = {
            'dtype': dtype_name,
            'shape': tuple(tensor.shape),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'nan_count': int(torch.isnan(tensor).sum()),
            'inf_count': int(torch.isinf(tensor).sum()),
            'zero_count': int((tensor == 0).sum()),
            'total_elements': tensor.numel()
        }
        
        # 범위별 분포
        abs_tensor = torch.abs(tensor[torch.isfinite(tensor)])
        if abs_tensor.numel() > 0:
            stats.update({
                'abs_min': float(abs_tensor.min()),
                'abs_max': float(abs_tensor.max()),
                'abs_mean': float(abs_tensor.mean()),
            })
            
            # 지수별 분포 (대략적인 범위)
            log_abs = torch.log10(abs_tensor + 1e-45)  # 매우 작은 값 방지
            stats['log_range'] = {
                'very_small': int((log_abs < -6).sum()),   # < 1e-6
                'small': int(((log_abs >= -6) & (log_abs < -3)).sum()),  # 1e-6 to 1e-3
                'normal': int(((log_abs >= -3) & (log_abs < 3)).sum()),  # 1e-3 to 1e3
                'large': int(((log_abs >= 3) & (log_abs < 6)).sum()),    # 1e3 to 1e6
                'very_large': int((log_abs >= 6).sum())    # > 1e6
            }
        
        return stats

    @staticmethod
    def get_float_limits():
        """각 float 타입의 한계값 반환"""
        return {
            'fp32': {
                'max': 3.4028235e+38,
                'min': -3.4028235e+38,
                'eps': 1.1920929e-07,
                'tiny': 1.1754944e-38
            },
            'fp16': {
                'max': 65504.0,
                'min': -65504.0,
                'eps': 0.0009765625,
                'tiny': 6.103515625e-05
            },
            'bf16': {
                'max': 3.3895314e+38,
                'min': -3.3895314e+38,
                'eps': 0.0078125,
                'tiny': 1.1754944e-38
            }
        }

class PrecisionMonitorHook:
    """모델의 forward/backward pass 중 정밀도를 모니터링하는 훅"""
    
    def __init__(self, monitor_gradients=True):
        self.monitor_gradients = monitor_gradients
        self.forward_stats = defaultdict(list)
        self.backward_stats = defaultdict(list)
        self.analyzer = FloatAnalyzer()
        
    def forward_hook(self, module, input, output):
        """Forward pass 모니터링"""
        module_name = module.__class__.__name__
        
        # Input 분석
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor) and inp.dtype.is_floating_point:
                    stats = self.analyzer.analyze_float_properties(inp, str(inp.dtype))
                    stats['module'] = f"{module_name}_input_{i}"
                    stats['pass_type'] = 'forward'
                    self.forward_stats[f"{module_name}_input_{i}"].append(stats)
        
        # Output 분석
        if isinstance(output, torch.Tensor) and output.dtype.is_floating_point:
            stats = self.analyzer.analyze_float_properties(output, str(output.dtype))
            stats['module'] = f"{module_name}_output"
            stats['pass_type'] = 'forward'
            self.forward_stats[f"{module_name}_output"].append(stats)
    
    def backward_hook(self, module, grad_input, grad_output):
        """Backward pass 모니터링"""
        if not self.monitor_gradients:
            return
            
        module_name = module.__class__.__name__
        
        # Gradient output 분석
        if grad_output is not None:
            if isinstance(grad_output, tuple):
                for i, grad in enumerate(grad_output):
                    if isinstance(grad, torch.Tensor) and grad is not None:
                        stats = self.analyzer.analyze_float_properties(grad, str(grad.dtype))
                        stats['module'] = f"{module_name}_grad_out_{i}"
                        stats['pass_type'] = 'backward'
                        self.backward_stats[f"{module_name}_grad_out_{i}"].append(stats)
    
    def register_hooks(self, model):
        """모델에 훅 등록"""
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules만
                hook_f = module.register_forward_hook(self.forward_hook)
                hooks.append(hook_f)
                if self.monitor_gradients:
                    hook_b = module.register_backward_hook(self.backward_hook)
                    hooks.append(hook_b)
        return hooks

class MultiPrecisionTrainer:
    """다중 정밀도로 ResNet을 훈련하고 모니터링하는 클래스"""
    
    def __init__(self, model_name='resnet18', num_classes=10):
        self.model_name = model_name
        self.num_classes = num_classes
        self.monitors = {}
        self.training_logs = defaultdict(list)
        
    def create_model(self, precision='fp32'):
        """지정된 정밀도로 모델 생성"""
        if self.model_name == 'resnet18':
            model = models.resnet18(num_classes=self.num_classes)
        elif self.model_name == 'resnet50':
            model = models.resnet50(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # 정밀도 설정
        if precision == 'fp16':
            model = model.half()
        elif precision == 'bf16':
            model = model.to(torch.bfloat16)
        
        return model
    
    def prepare_data(self, batch_size=32):
        """CIFAR-10 데이터 로더 준비"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = datasets.CIFAR10(root='/content/data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='/content/data', train=False, download=True, transform=transform_test)
        
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    def train_epoch(self, model, trainloader, criterion, optimizer, precision, device, monitor_frequency=10):
        """한 에폭 훈련 및 모니터링"""
        model.train()
        
        # 모니터 설정
        if precision not in self.monitors:
            self.monitors[precision] = PrecisionMonitorHook()
            hooks = self.monitors[precision].register_hooks(model)
        
        running_loss = 0.0
        batch_stats = []
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 정밀도 맞춤
            if precision == 'fp16':
                inputs = inputs.half()
            elif precision == 'bf16':
                inputs = inputs.to(torch.bfloat16)
                
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Loss 통계 수집
            loss_stats = {
                'batch': batch_idx,
                'loss_value': float(loss.item()) if not torch.isnan(loss) else float('nan'),
                'loss_dtype': str(loss.dtype),
                'is_nan': bool(torch.isnan(loss)),
                'is_inf': bool(torch.isinf(loss))
            }
            batch_stats.append(loss_stats)
            
            # Backward pass
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                
                # Gradient clipping (fp16에서 유용)
                if precision == 'fp16':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            else:
                print(f"Warning: NaN or Inf loss detected at batch {batch_idx}")
            
            running_loss += loss.item() if not torch.isnan(loss) else 0.0
            
            # 주기적 모니터링 출력
            if batch_idx % monitor_frequency == 0:
                self.print_batch_stats(precision, batch_idx, loss_stats)
                
            # 메모리 절약을 위해 일부 배치만 처리
            if batch_idx >= 50:  # 처음 50 배치만
                break
        
        self.training_logs[precision].extend(batch_stats)
        return running_loss / min(len(trainloader), 50)
    
    def print_batch_stats(self, precision, batch_idx, loss_stats):
        """배치별 통계 출력"""
        print(f"[{precision}] Batch {batch_idx:3d} | Loss: {loss_stats['loss_value']:.6f} | "
              f"NaN: {loss_stats['is_nan']} | Inf: {loss_stats['is_inf']}")
    
    def analyze_precision_comparison(self):
        """정밀도별 비교 분석"""
        print("=" * 80)
        print("PRECISION COMPARISON ANALYSIS")
        print("=" * 80)
        
        for precision in self.monitors:
            monitor = self.monitors[precision]
            print(f"\n{precision.upper()} Analysis:")
            print("-" * 40)
            
            # Forward pass 통계
            print("Forward Pass Issues:")
            for module_name, stats_list in monitor.forward_stats.items():
                if stats_list:
                    latest_stats = stats_list[-1]
                    if latest_stats['nan_count'] > 0 or latest_stats['inf_count'] > 0:
                        print(f"  {module_name}: NaN={latest_stats['nan_count']}, "
                              f"Inf={latest_stats['inf_count']}")
            
            # Backward pass 통계
            if monitor.backward_stats:
                print("Backward Pass Issues:")
                for module_name, stats_list in monitor.backward_stats.items():
                    if stats_list:
                        latest_stats = stats_list[-1]
                        if latest_stats['nan_count'] > 0 or latest_stats['inf_count'] > 0:
                            print(f"  {module_name}: NaN={latest_stats['nan_count']}, "
                                  f"Inf={latest_stats['inf_count']}")
    
    def plot_precision_analysis(self):
        """정밀도 분석 결과 시각화"""
        if not self.training_logs:
            print("No training logs available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Precision Training Analysis', fontsize=16)
        
        # 1. Loss 비교
        ax1 = axes[0, 0]
        for precision in self.training_logs:
            logs = self.training_logs[precision]
            batches = [log['batch'] for log in logs if not np.isnan(log['loss_value'])]
            losses = [log['loss_value'] for log in logs if not np.isnan(log['loss_value'])]
            if batches and losses:
                ax1.plot(batches, losses, label=f'{precision}', marker='o', markersize=3)
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Comparison Across Precisions')
        ax1.legend()
        ax1.grid(True)
        
        # 2. NaN 발생 패턴
        ax2 = axes[0, 1]
        nan_counts = {}
        for precision in self.training_logs:
            logs = self.training_logs[precision]
            nan_counts[precision] = sum(1 for log in logs if log['is_nan'])
        
        precisions = list(nan_counts.keys())
        counts = list(nan_counts.values())
        ax2.bar(precisions, counts, alpha=0.7)
        ax2.set_ylabel('NaN Count')
        ax2.set_title('NaN Occurrences by Precision')
        
        # 3. Float 범위 분포 (예시)
        ax3 = axes[1, 0]
        limits = FloatAnalyzer.get_float_limits()
        precisions = ['fp32', 'fp16']
        max_vals = [limits[p]['max'] for p in precisions]
        ax3.bar(precisions, max_vals, alpha=0.7)
        ax3.set_yscale('log')
        ax3.set_ylabel('Maximum Value (log scale)')
        ax3.set_title('Float Type Limits')
        
        # 4. 훈련 안정성 지표
        ax4 = axes[1, 1]
        stability_scores = {}
        for precision in self.training_logs:
            logs = self.training_logs[precision]
            valid_losses = [log['loss_value'] for log in logs 
                          if not np.isnan(log['loss_value']) and not np.isinf(log['loss_value'])]
            if valid_losses:
                stability_scores[precision] = np.std(valid_losses) / np.mean(valid_losses)
            else:
                stability_scores[precision] = float('inf')
        
        precisions = list(stability_scores.keys())
        scores = [stability_scores[p] for p in precisions]
        ax4.bar(precisions, scores, alpha=0.7)
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('Training Stability (lower is better)')
        
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self):
        """상세한 분석 리포트 생성"""
        print("=" * 100)
        print("DETAILED FLOAT PRECISION ANALYSIS REPORT")
        print("=" * 100)
        
        limits = FloatAnalyzer.get_float_limits()
        
        for precision_name, limit_info in limits.items():
            print(f"\n{precision_name.upper()} CHARACTERISTICS:")
            print("-" * 50)
            print(f"Maximum value: {limit_info['max']:.2e}")
            print(f"Minimum value: {limit_info['min']:.2e}")
            print(f"Machine epsilon: {limit_info['eps']:.2e}")
            print(f"Smallest normal: {limit_info['tiny']:.2e}")
            
            if precision_name in self.training_logs:
                logs = self.training_logs[precision_name]
                nan_count = sum(1 for log in logs if log['is_nan'])
                inf_count = sum(1 for log in logs if log['is_inf'])
                total_batches = len(logs)
                
                print(f"\nTraining Statistics:")
                print(f"Total batches processed: {total_batches}")
                print(f"NaN occurrences: {nan_count} ({nan_count/total_batches*100:.2f}%)")
                print(f"Inf occurrences: {inf_count} ({inf_count/total_batches*100:.2f}%)")

def main():
    """메인 실행 함수"""
    print("Multi-Precision ResNet Training Monitor")
    print("=" * 50)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 트레이너 초기화
    trainer = MultiPrecisionTrainer(model_name='resnet18', num_classes=10)
    
    # 데이터 준비
    print("Preparing data...")
    trainloader, testloader = trainer.prepare_data(batch_size=32)
    
    # 다양한 정밀도로 훈련 테스트
    precisions = ['fp32', 'fp16']
    
    for precision in precisions:
        print(f"\n{'='*20} Testing {precision.upper()} {'='*20}")
        
        # 모델 및 최적화기 생성
        model = trainer.create_model(precision).to(device)
        
        if precision == 'fp16':
            model = model.half()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        try:
            # 한 에폭만 훈련 (데모용)
            avg_loss = trainer.train_epoch(
                model, trainloader, criterion, optimizer, 
                precision, device, monitor_frequency=5
            )
            print(f"{precision} average loss: {avg_loss:.6f}")
            
        except Exception as e:
            print(f"Error during {precision} training: {str(e)}")
            continue
    
    # 분석 결과 출력
    trainer.analyze_precision_comparison()
    trainer.plot_precision_analysis()
    trainer.generate_detailed_report()

# 실행 예제
if __name__ == "__main__":
    main()
