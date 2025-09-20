import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import struct
from collections import Counter, defaultdict
import pandas as pd
from IPython.display import display, HTML

class FloatBitAnalyzer:
    """Float의 비트 수준 분석을 위한 클래스"""
    
    @staticmethod
    def float32_to_bits(f):
        """Float32를 비트 표현으로 변환"""
        bits = struct.unpack('>I', struct.pack('>f', f))[0]
        return format(bits, '032b')
    
    @staticmethod
    def float16_to_bits(f):
        """Float16을 비트 표현으로 변환"""
        # PyTorch tensor를 통해 변환
        tensor = torch.tensor(f, dtype=torch.float16)
        bits = tensor.view(torch.int16).item()
        if bits < 0:
            bits = bits + 2**16
        return format(bits, '016b')
    
    @staticmethod
    def parse_float32_bits(bits_str):
        """Float32 비트 문자열을 sign, exponent, mantissa로 분해"""
        sign = int(bits_str[0])
        exponent = int(bits_str[1:9], 2)
        mantissa = int(bits_str[9:], 2)
        return sign, exponent, mantissa
    
    @staticmethod
    def parse_float16_bits(bits_str):
        """Float16 비트 문자열을 sign, exponent, mantissa로 분해"""
        sign = int(bits_str[0])
        exponent = int(bits_str[1:6], 2)
        mantissa = int(bits_str[6:], 2)
        return sign, exponent, mantissa
    
    @classmethod
    def analyze_tensor_bits(cls, tensor, precision='fp32'):
        """텐서의 모든 값에 대한 비트 수준 분석"""
        if tensor.numel() == 0:
            return {}
        
        # 유한한 값만 분석
        finite_tensor = tensor[torch.isfinite(tensor)]
        if finite_tensor.numel() == 0:
            return {'error': 'No finite values to analyze'}
        
        values = finite_tensor.cpu().numpy().flatten()
        
        bit_analysis = {
            'total_values': len(values),
            'exponent_dist': Counter(),
            'mantissa_patterns': Counter(),
            'sign_dist': Counter(),
            'zero_mantissa_count': 0,
            'denormal_count': 0,
            'special_values': {'nan': 0, 'inf': 0, 'zero': 0}
        }
        
        for val in values:
            if np.isnan(val):
                bit_analysis['special_values']['nan'] += 1
                continue
            elif np.isinf(val):
                bit_analysis['special_values']['inf'] += 1
                continue
            elif val == 0.0:
                bit_analysis['special_values']['zero'] += 1
                continue
            
            if precision == 'fp32':
                bits = cls.float32_to_bits(float(val))
                sign, exponent, mantissa = cls.parse_float32_bits(bits)
                bias = 127
            else:  # fp16
                bits = cls.float16_to_bits(float(val))
                sign, exponent, mantissa = cls.parse_float16_bits(bits)
                bias = 15
            
            bit_analysis['sign_dist'][sign] += 1
            bit_analysis['exponent_dist'][exponent] += 1
            
            if mantissa == 0:
                bit_analysis['zero_mantissa_count'] += 1
            
            # Denormal (subnormal) 수 확인
            if exponent == 0 and mantissa != 0:
                bit_analysis['denormal_count'] += 1
            
            # 지수의 실제 값 (bias 제거)
            if exponent != 0:
                actual_exp = exponent - bias
                bit_analysis['mantissa_patterns'][f'exp_{actual_exp}'] += 1
        
        return bit_analysis

class PrecisionRangeExplorer:
    """Float 정밀도의 범위와 한계를 탐구하는 클래스"""
    
    def __init__(self):
        self.test_values = self._generate_test_values()
    
    def _generate_test_values(self):
        """다양한 범위의 테스트 값 생성"""
        values = []
        
        # 정상적인 범위의 값들
        values.extend(np.logspace(-10, 10, 100))
        values.extend(-np.logspace(-10, 10, 100))
        
        # 극한값들
        values.extend([1e-45, 1e-40, 1e-35, 1e-30])  # 매우 작은 값
        values.extend([1e30, 1e35, 1e38, 1e39])      # 매우 큰 값
        
        # 경계값들
        values.extend([65504, 65505, 65520])  # FP16 한계 근처
        values.extend([0.0, -0.0])            # Zero values
        values.extend([float('inf'), float('-inf'), float('nan')])
        
        return np.array(values, dtype=np.float32)
    
    def test_precision_conversion(self):
        """정밀도 변환 테스트"""
        results = {
            'original': [],
            'fp32': [],
            'fp16': [],
            'fp32_recovered': [],
            'conversion_error': [],
            'precision_loss': []
        }
        
        for val in self.test_values:
            if not np.isfinite(val):
                continue
                
            original = float(val)
            
            # FP32 변환
            fp32_tensor = torch.tensor(original, dtype=torch.float32)
            fp32_val = float(fp32_tensor)
            
            # FP16 변환
            try:
                fp16_tensor = torch.tensor(original, dtype=torch.float16)
                fp16_val = float(fp16_tensor)
                
                # FP16에서 다시 FP32로
                fp32_recovered = float(fp16_tensor.float())
                
                conversion_error = abs(original - fp16_val)
                precision_loss = abs(fp32_val - fp32_recovered)
                
                results['original'].append(original)
                results['fp32'].append(fp32_val)
                results['fp16'].append(fp16_val)
                results['fp32_recovered'].append(fp32_recovered)
                results['conversion_error'].append(conversion_error)
                results['precision_loss'].append(precision_loss)
                
            except (RuntimeError, OverflowError):
                # FP16 범위를 벗어난 경우
                continue
        
        return pd.DataFrame(results)
    
    def analyze_gradient_overflow_risk(self, model):
        """모델의 그래디언트 오버플로우 위험 분석"""
        overflow_risks = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # 통계 계산
                risk_factors = {
                    'max_abs_grad': float(torch.max(torch.abs(grad))),
                    'grad_norm': float(torch.norm(grad)),
                    'large_grad_ratio': float((torch.abs(grad) > 1000).sum()) / grad.numel(),
                    'small_grad_ratio': float((torch.abs(grad) < 1e-6).sum()) / grad.numel(),
                    'fp16_overflow_risk': float(torch.max(torch.abs(grad))) > 65000
                }
                
                overflow_risks[name] = risk_factors
        
        return overflow_risks
    
    def plot_precision_analysis(self, df):
        """정밀도 분석 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Float Precision Analysis', fontsize=16)
        
        # 1. 원본 vs FP16 값 비교
        ax1 = axes[0, 0]
        valid_idx = (np.isfinite(df['original']) & np.isfinite(df['fp16']))
        ax1.scatter(df['original'][valid_idx], df['fp16'][valid_idx], alpha=0.6, s=1)
        ax1.plot([-1e10, 1e10], [-1e10, 1e10], 'r--', alpha=0.7)
        ax1.set_xlabel('Original (FP32)')
        ax1.set_ylabel('FP16')
        ax1.set_title('FP32 vs FP16 Values')
        ax1.set_xscale('symlog')
        ax1.set_yscale('symlog')
        
        # 2. 변환 오차 분포
        ax2 = axes[0, 1]
        valid_errors = df['conversion_error'][df['conversion_error'] > 0]
        if len(valid_errors) > 0:
            ax2.hist(np.log10(valid_errors), bins=50, alpha=0.7)
            ax2.set_xlabel('Log10(Conversion Error)')
            ax2.set_ylabel('Count')
            ax2.set_title('Conversion Error Distribution')
        
        # 3. 정밀도 손실 분포
        ax3 = axes[0, 2]
        valid_loss = df['precision_loss'][df['precision_loss'] > 0]
        if len(valid_loss) > 0:
            ax3.hist(np.log10(valid_loss), bins=50, alpha=0.7)
            ax3.set_xlabel('Log10(Precision Loss)')
            ax3.set_ylabel('Count')
            ax3.set_title('Precision Loss Distribution')
        
        # 4. 값의 크기별 오차율
        ax4 = axes[1, 0]
        abs_original = np.abs(df['original'])
        relative_error = df['conversion_error'] / (abs_original + 1e-45)
        valid_idx = np.isfinite(relative_error) & (abs_original > 0)
        
        if np.sum(valid_idx) > 0:
            ax4.scatter(abs_original[valid_idx], relative_error[valid_idx], alpha=0.6, s=1)
            ax4.set_xlabel('Absolute Original Value')
            ax4.set_ylabel('Relative Error')
            ax4.set_title('Relative Error vs Value Magnitude')
            ax4.set_xscale('log')
            ax4.set_yscale('log')
        
        # 5. FP16 범위 한계 시각화
        ax5 = axes[1, 1]
        fp16_max = 65504
        fp16_min = 6.103515625e-05
        
        original_abs = np.abs(df['original'])
        in_range = (original_abs <= fp16_max) & (original_abs >= fp16_min)
        out_range = ~in_range & np.isfinite(df['original'])
        
        ax5.scatter(df['original'][in_range], df['fp16'][in_range], 
                   alpha=0.6, s=1, label='In FP16 range', color='green')
        ax5.scatter(df['original'][out_range], df['fp16'][out_range], 
                   alpha=0.6, s=1, label='Out of FP16 range', color='red')
        ax5.axhline(y=fp16_max, color='red', linestyle='--', alpha=0.7, label='FP16 max')
        ax5.axhline(y=-fp16_max, color='red', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Original Value')
        ax5.set_ylabel('FP16 Value')
        ax5.set_title('FP16 Range Limitations')
        ax5.legend()
        ax5.set_yscale('symlog')
        ax5.set_xscale('symlog')
        
        # 6. 지수 분포 비교
        ax6 = axes[1, 2]
        
        # FP32와 FP16의 지수 범위 비교
        fp32_exp_range = range(-126, 128)  # FP32 지수 범위
        fp16_exp_range = range(-14, 16)    # FP16 지수 범위
        
        ax6.bar([f'FP32\n({min(fp32_exp_range)} to {max(fp32_exp_range)})'], 
               [len(fp32_exp_range)], alpha=0.7, label='FP32 Exponent Range')
        ax6.bar([f'FP16\n({min(fp16_exp_range)} to {max(fp16_exp_range)})'], 
               [len(fp16_exp_range)], alpha=0.7, label='FP16 Exponent Range')
        ax6.set_ylabel('Number of Exponent Values')
        ax6.set_title('Exponent Range Comparison')
        ax6.legend()
        
        plt.tight_layout()
        plt.show()

class NaNInvestigator:
    """NaN 발생 원인을 조사하는 클래스"""
    
    def __init__(self):
        self.nan_sources = []
        self.operation_log = []
    
    def monitor_operations(self, tensor1, tensor2, operation, result):
        """연산 모니터링"""
        log_entry = {
            'operation': operation,
            'input1_stats': self._get_tensor_stats(tensor1),
            'input2_stats': self._get_tensor_stats(tensor2) if tensor2 is not None else None,
            'result_stats': self._get_tensor_stats(result),
            'nan_introduced': torch.isnan(result).sum().item() > 0
        }
        
        if log_entry['nan_introduced']:
            self.nan_sources.append(log_entry)
        
        self.operation_log.append(log_entry)
        return result
    
    def _get_tensor_stats(self, tensor):
        """텐서 통계 계산"""
        if tensor is None:
            return None
        
        return {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'min': float(tensor.min()) if tensor.numel() > 0 else None,
            'max': float(tensor.max()) if tensor.numel() > 0 else None,
            'mean': float(tensor.mean()) if tensor.numel() > 0 else None,
            'nan_count': int(torch.isnan(tensor).sum()),
            'inf_count': int(torch.isinf(tensor).sum()),
            'very_large_count': int((torch.abs(tensor) > 1e10).sum()),
            'very_small_count': int((torch.abs(tensor) < 1e-10).sum())
        }
    
    def create_problematic_scenarios(self):
        """문제가 될 수 있는 시나리오 생성 및 테스트"""
        scenarios = []
        
        # 1. 매우 큰 값들의 곱셈
        large_vals = torch.tensor([1e20, 1e25, 1e30], dtype=torch.float16)
        result1 = large_vals * large_vals
        scenarios.append(('Large multiplication', large_vals, large_vals, result1))
        
        # 2. 매우 작은 값들의 나눗셈
        small_vals = torch.tensor([1e-20, 1e-25, 1e-30], dtype=torch.float16)
        tiny_vals = torch.tensor([1e-35, 1e-38, 1e-40], dtype=torch.float16)
        result2 = small_vals / tiny_vals
        scenarios.append(('Small division', small_vals, tiny_vals, result2))
        
        # 3. 지수 함수
        large_inputs = torch.tensor([50, 100, 200], dtype=torch.float16)
        result3 = torch.exp(large_inputs)
        scenarios.append(('Exponential overflow', large_inputs, None, result3))
        
        # 4. 로그 함수 (음수나 0 입력)
        problematic_inputs = torch.tensor([0, -1, -10], dtype=torch.float16)
        result4 = torch.log(problematic_inputs)
        scenarios.append(('Log of non-positive', problematic_inputs, None, result4))
        
        # 5. Gradient explosion 시나리오
        gradients = torch.tensor([1e5, 1e6, 1e7], dtype=torch.float16)
        learning_rate = torch.tensor(0.1, dtype=torch.float16)
        result5 = gradients * learning_rate
        scenarios.append(('Gradient explosion', gradients, learning_rate, result5))
        
        return scenarios
    
    def analyze_scenarios(self):
        """시나리오 분석"""
        scenarios = self.create_problematic_scenarios()
        
        print("NaN/Inf Generation Scenarios Analysis")
        print("=" * 60)
        
        for scenario_name, input1, input2, result in scenarios:
            print(f"\nScenario: {scenario_name}")
            print("-" * 40)
            
            print(f"Input 1: {input1}")
            if input2 is not None:
                print(f"Input 2: {input2}")
            print(f"Result: {result}")
            
            nan_count = torch.isnan(result).sum().item()
            inf_count = torch.isinf(result).sum().item()
            
            print(f"NaN count: {nan_count}")
            print(f"Inf count: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                print("⚠️  PROBLEMATIC SCENARIO DETECTED!")
                
                # FP32로 같은 연산 수행해서 비교
                if input2 is not None:
                    fp32_result = input1.float() * input2.float()  # 예시 연산
                else:
                    fp32_result = torch.exp(input1.float()) if 'Exponential' in scenario_name else torch.log(input1.float())
                
                print(f"FP32 result: {fp32_result}")
                print(f"FP32 NaN count: {torch.isnan(fp32_result).sum().item()}")
                print(f"FP32 Inf count: {torch.isinf(fp32_result).sum().item()}")

def create_custom_float_format_simulator():
    """커스텀 float 포맷 시뮬레이터"""
    
    class CustomFloatFormat:
        def __init__(self, sign_bits=1, exponent_bits=8, mantissa_bits=23):
            self.sign_bits = sign_bits
            self.exponent_bits = exponent_bits
            self.mantissa_bits = mantissa_bits
            self.total_bits = sign_bits + exponent_bits + mantissa_bits
            self.bias = (2 ** (exponent_bits - 1)) - 1
            
        def get_limits(self):
            """포맷의 한계값 계산"""
            max_exponent = (2 ** self.exponent_bits) - 2  # 모든 1은 특수값용
            min_exponent = 1  # 0은 denormal/zero용
            
            # 최대값: 2^(max_exp - bias) * (2 - 2^(-mantissa_bits))
            max_val = (2 ** (max_exponent - self.bias)) * (2 - 2**(-self.mantissa_bits))
            
            # 최소 정규값: 2^(min_exp - bias)
            min_normal = 2 ** (min_exponent - self.bias)
            
            # 최소 denormal값: 2^(1 - bias - mantissa_bits)
            min_denormal = 2 ** (1 - self.bias - self.mantissa_bits)
            
            return {
                'max': max_val,
                'min_normal': min_normal,
                'min_denormal': min_denormal,
                'exponent_range': (min_exponent - self.bias, max_exponent - self.bias)
            }
        
        def analyze_representation_density(self, value_range):
            """주어진 범위에서의 표현 가능한 값의 밀도 분석"""
            # 이것은 실제 구현보다는 개념적 분석
            limits = self.get_limits()
            
            analysis = {
                'format': f"Sign:{self.sign_bits} Exp:{self.exponent_bits} Mantissa:{self.mantissa_bits}",
                'total_bits': self.total_bits,
                'limits': limits,
                'representation_count': 2 ** self.total_bits - 2,  # NaN, Inf 제외
            }
            
            return analysis
    
    # 다양한 포맷 테스트
    formats = {
        'FP32': CustomFloatFormat(1, 8, 23),
        'FP16': CustomFloatFormat(1, 5, 10),
        'BF16': CustomFloatFormat(1, 8, 7),
        'Custom_High_Precision': CustomFloatFormat(1, 6, 25),  # 높은 정밀도
        'Custom_Large_Range': CustomFloatFormat(1, 10, 5),     # 큰 범위
        'Custom_Balanced': CustomFloatFormat(1, 7, 16),        # 균형잡힌 포맷
    }
    
    print("Custom Float Format Analysis")
    print("=" * 60)
    
    for format_name, format_obj in formats.items():
        analysis = format_obj.analyze_representation_density(None)
        limits = analysis['limits']
        
        print(f"\n{format_name}:")
        print(f"  Format: {analysis['format']}")
        print(f"  Total bits: {analysis['total_bits']}")
        print(f"  Max value: {limits['max']:.2e}")
        print(f"  Min normal: {limits['min_normal']:.2e}")
        print(f"  Min denormal: {limits['min_denormal']:.2e}")
        print(f"  Exponent range: {limits['exponent_range']}")
        print(f"  Representable values: {analysis['representation_count']:,}")

# 실행 및 데모 함수들
def run_comprehensive_analysis():
    """포괄적인 정밀도 분석 실행"""
    print("Starting Comprehensive Float Precision Analysis...")
    
    # 1. 비트 레벨 분석
    analyzer = FloatBitAnalyzer()
    
    # 테스트 텐서 생성
    test_tensor_fp32 = torch.randn(1000, dtype=torch.float32) * 1000
    test_tensor_fp16 = test_tensor_fp32.half()
    
    print("\n1. Bit-level Analysis")
    print("-" * 30)
    
    fp32_bits = analyzer.analyze_tensor_bits(test_tensor_fp32, 'fp32')
    fp16_bits = analyzer.analyze_tensor_bits(test_tensor_fp16, 'fp16')
    
    print("FP32 bit analysis:")
    for key, value in fp32_bits.items():
        if isinstance(value, dict):
            print(f"  {key}: {dict(list(value.items())[:5])}...")  # 처음 5개만 표시
        else:
            print(f"  {key}: {value}")
    
    print("\nFP16 bit analysis:")
    for key, value in fp16_bits.items():
        if isinstance(value, dict):
            print(f"  {key}: {dict(list(value.items())[:5])}...")
        else:
            print(f"  {key}: {value}")
    
    # 2. 정밀도 범위 탐구
    print("\n2. Precision Range Exploration")
    print("-" * 30)
    
    explorer = PrecisionRangeExplorer()
    conversion_df = explorer.test_precision_conversion()
    
    print(f"Tested {len(conversion_df)} conversions")
    print(f"Max conversion error: {conversion_df['conversion_error'].max():.2e}")
    print(f"Mean conversion error: {conversion_df['conversion_error'].mean():.2e}")
    
    # 시각화
    explorer.plot_precision_analysis(conversion_df)
    
    # 3. NaN 원인 조사
    print("\n3. NaN Source Investigation")
    print("-" * 30)
    
    investigator = NaNInvestigator()
    investigator.analyze_scenarios()
    
    # 4. 커스텀 포맷 분석
    print("\n4. Custom Float Format Analysis")
    print("-" * 30)
    
    create_custom_float_format_simulator()

# Colab에서 실행하기 위한 간단한 시작 함수
def start_monitoring():
    """모니터링 시작"""
    print("🔍 Float Precision Monitoring Started!")
    print("Use the following functions:")
    print("- run_comprehensive_analysis(): 포괄적인 분석 실행")
    print("- MultiPrecisionTrainer(): ResNet 훈련 모니터링")
    print("- FloatBitAnalyzer(): 비트 레벨 분석")
    print("- NaNInvestigator(): NaN 원인 조사")

if __name__ == "__main__":
    start_monitoring()