import pandas as pd
import matplotlib.pyplot as plt

stu_scores = pd.read_csv('student_scores.csv')

# plt.scatter(stu_scores.Hours, stu_scores.Scores)
# plt.show()

def loss_function(m, b, points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        total_loss += (y - (m*x + b)) ** 2
    return total_loss / float(len(points))

def gradient_decent(m_cur, b_cur, points, l):
    m_grad = 0
    b_grad = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores

        m_grad += -(2/n) * x * (y- (m_cur * x + b_cur))
        b_grad += -(2/n) * (y- (m_cur * x + b_cur))

    m = m_cur - l * m_grad
    b = b_cur - l * b_grad
    return m, b


m = 0
b = 0
l = 0.0001
epoches = 800

for i in range(epoches):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_decent(m, b, stu_scores, l)
    
print(m, b)
print(loss_function(m, b, stu_scores))
plt.scatter(stu_scores.Hours, stu_scores.Scores, color='black')
plt.plot(list(range(0, 11)), [m * x + b for x in range(0, 11)], color='red')
plt.show()