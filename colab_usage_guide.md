# ResNet Float Precision 모니터링 - Colab 사용 가이드

## 🚀 빠른 시작

### 1. 라이브러리 설치 및 임포트
```python
# Colab에서 필요한 패키지 설치
!pip install seaborn matplotlib pandas

# 코드 실행
exec(open('resnet_precision_monitor.py').read())
exec(open('float_bit_analyzer.py').read())
```

### 2. 기본 모니터링 시작
```python
# 모니터링 시작
start_monitoring()

# 포괄적인 분석 실행
run_comprehensive_analysis()
```

## 📊 주요 기능별 사용법

### A. ResNet 다중 정밀도 훈련 모니터링

```python
# 1. 트레이너 초기화
trainer = MultiPrecisionTrainer(model_name='resnet18', num_classes=10)

# 2. 데이터 준비
trainloader, testloader = trainer.prepare_data(batch_size=32)

# 3. 여러 정밀도로 훈련 테스트
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for precision in ['fp32', 'fp16']:
    print(f"Testing {precision}")
    
    model = trainer.create_model(precision).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 한 에폭 훈련 (모니터링 포함)
    avg_loss = trainer.train_epoch(
        model, trainloader, criterion, optimizer,
        precision, device, monitor_frequency=5
    )
    
    print(f"{precision} average loss: {avg_loss:.6f}")

# 4. 결과 분석
trainer.analyze_precision_comparison()
trainer.plot_precision_analysis()
trainer.generate_detailed_report()
```

### B. 비트 레벨 분석

```python
# 1. 분석기 초기화
analyzer = FloatBitAnalyzer()

# 2. 테스트 데이터 생성
test_data = torch.randn(1000) * 10000  # 큰 범위의 랜덤 데이터

# 3. FP32와 FP16 비교 분석
fp32_tensor = test_data.float()
fp16_tensor = test_data.half()

fp32_analysis = analyzer.analyze_tensor_bits(fp32_tensor, 'fp32')
fp16_analysis = analyzer.analyze_tensor_bits(fp16_tensor, 'fp16')

print("FP32 Analysis:", fp32_analysis)
print("FP16 Analysis:", fp16_analysis)

# 4. 개별 값의 비트 패턴 확인
sample_value = 12345.67
print("FP32 bits:", analyzer.float32_to_bits(sample_value))
print("FP16 bits:", analyzer.float16_to_bits(sample_value))
```

### C. NaN 발생 원인 조사

```python
# 1. NaN 조사기 초기화
investigator = NaNInvestigator()

# 2. 문제가 되는 시나리오 분석
investigator.analyze_scenarios()

# 3. 커스텀 연산 모니터링
a = torch.tensor([1e20, 1e25], dtype=torch.float16)
b = torch.tensor([1e15, 1e20], dtype=torch.float16)
result = investigator.monitor_operations(a, b, 'multiplication', a * b)

# 4. NaN 소스 확인
print("NaN sources found:", len(investigator.nan_sources))
for source in investigator.nan_sources:
    print(f"Operation: {source['operation']}")
    print(f"Result stats: {source['result_stats']}")
```

### D. 정밀도 변환 테스트

```python
# 1. 범위 탐구기 초기화
explorer = PrecisionRangeExplorer()

# 2. 정밀도 변환 테스트 실행
conversion_results = explorer.test_precision_conversion()

# 3. 결과 분석
print("Conversion test results:")
print(conversion_results.describe())

# 4. 시각화
explorer.plot_precision_analysis(conversion_results)

# 5. 그래디언트 오버플로우 위험 분석 (모델 훈련 후)
# model = your_trained_model
# overflow_risks = explorer.analyze_gradient_overflow_risk(model)
# print("Overflow risks:", overflow_risks)
```

## 🔍 특정 문제 해결 가이드

### 1. FP16에서 Loss가 NaN이 되는 경우

```python
# 문제 진단
def diagnose_fp16_nan_loss():
    # 작은 배치로 테스트
    model = models.resnet18(num_classes=10).half().cuda()
    criterion = nn.CrossEntropyLoss()
    
    # 더미 데이터로 테스트
    dummy_input = torch.randn(4, 3, 32, 32).half().cuda()
    dummy_target = torch.randint(0, 10, (4,)).cuda()
    
    # Forward pass 모니터링
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    print(f"Output stats: min={output.min()}, max={output.max()}, has_nan={torch.isnan(output).any()}")
    print(f"Loss: {loss.item()}, is_nan={torch.isnan(loss)}")
    
    # 그래디언트 모니터링
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad)
            if torch.isnan(grad_norm) or grad_norm > 1000:
                print(f"Problematic gradient in {name}: norm={grad_norm}")

diagnose_fp16_nan_loss()
```

### 2. 커스텀 float 포맷 실험

```python
# 커스텀 포맷 시뮬레이터 실행
create_custom_float_format_simulator()

# 특정 지수/가수 비트 조합 테스트
custom_format = CustomFloatFormat(sign_bits=1, exponent_bits=6, mantissa_bits=17)
limits = custom_format.get_limits()
print("Custom format limits:", limits)
```

### 3. 실시간 훈련 모니터링

```python
class RealTimeMonitor:
    def __init__(self):
        self.loss_history = []
        self.nan_detected = False
    
    def monitor_batch(self, batch_idx, loss, model):
        # Loss 기록
        self.loss_history.append(float(loss.item()))
        
        # NaN 체크
        if torch.isnan(loss):
            self.nan_detected = True
            print(f"🚨 NaN detected at batch {batch_idx}!")
            
            # 모델 파라미터 진단
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in parameter: {name}")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradient: {name}")
        
        # 주기적 리포트
        if batch_idx % 10 == 0:
            recent_losses = self.loss_history[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"Batch {batch_idx}: Avg Loss = {avg_loss:.6f}")

# 사용 예제
monitor = RealTimeMonitor()
# 훈련 루프에서 monitor.monitor_batch(batch_idx, loss, model) 호출
```

## 📈 결과 해석 가이드

### Loss NaN의 주요 원인들:

1. **그래디언트 폭발**: 매우 큰 그래디언트 값
2. **Learning rate 과다**: 업데이트 스텝이 너무 큼
3. **수치적 불안정성**: FP16의 제한된 범위
4. **배치 정규화 문제**: 작은 분산으로 인한 나눗셈 오류

### 해결 방법:

1. **Gradient Clipping 적용**
2. **Learning Rate 감소**
3. **Mixed Precision 훈련 사용**
4. **Loss Scaling 적용**

```python
# 안정적인 FP16 훈련을 위한 설정
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 훈련 루프에서
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🎯 연구 방향 제안

1. **지수/가수 비트 비율 최적화**: 특정 도메인에 최적화된 포맷 찾기
2. **적응적 정밀도**: 훈련 과정에서 동적으로 정밀도 조절
3. **하이브리드 접근법**: 레이어별로 다른 정밀도 사용
4. **정밀도 인식 최적화**: 정밀도 제약을 고려한 최적화 알고리즘

이 도구들을 사용하여 FP16 훈련의 수치적 안정성을 체계적으로 분석하고 개선할 수 있습니다!