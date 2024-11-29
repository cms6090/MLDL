import numpy as np

def step_function(x):
    return np.array(x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # np.exp : e^{-x}로 변환

def softmax(x):
    """ 소프트맥스 함수
    Input : array
    Output : array
    """
    
    # Input 값에 Input 값의 최댓값을 뺀다. -> 값이 커졌을 때 runtimewarning이 뜨는 것을 방지하지 위해
    array_x = x - np.max(x)
    exp_x = np.exp(array_x)
    result = exp_x/np.sum(exp_x)

    return result

def relu(x):
    return np.maximum(0, x)


def cross_entropy_error(y, t):  # 신경망의 예측 결과와 실제 레이블 간의 손실을 계산
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)