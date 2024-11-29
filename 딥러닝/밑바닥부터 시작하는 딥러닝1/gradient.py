import numpy as np

def numerical_gradient(f, X):  # 입력한 X의 차원에 따라 적절한 기울기 계산 함수 호출
    if X.ndim == 1: # .ndim : 배열의 차원 수
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)   # np.zeros_like(x) : x만큼의 사이즈인 0으로 가득 찬 Array 생성
        
        for idx, x in enumerate(X):  # 인덱스와 원소를 동시 접근
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad
    
def _numerical_gradient_no_batch(f, x):   # 기울기 구하기
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad