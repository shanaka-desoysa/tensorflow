import pandas as pd
import matplotlib.pyplot as plt

batch_result = [[13.36782193183899, 0.33670586840811867, 0.91220000000000001], [10.780983924865723, 0.3779737023271133, 0.90410000000000001]]
cols = ['Time', 'Loss', 'Accuracy']
idx = ['128', '256']
df = pd.DataFrame(batch_result, index=idx, columns=cols)
print(df)
df.Time.plot(kind='bar', color='r')
df.Loss.plot(kind='bar', color='b')
df.Accuracy.plot(kind='bar', color='g')
#plt.plot(idx, df['Time'], 'bo', label='Real data')
#plt.plot(X, pred_Y, 'ro', label='Predicted data')
#plt.xlabel('Standardized X')
#plt.ylabel('Y')

plt.legend()
plt.show()
'''
cols = ['Time', 'Avg. Loss', 'Accuracy']
idx = ['128', '256']
df = pd.DataFrame(batch_result, index=idx, columns=cols)
print(df)

X, Y = all_xs, all_ys
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, pred_Y, 'ro', label='Predicted data')
plt.xlabel('Standardized X')
plt.ylabel('Y')
'''