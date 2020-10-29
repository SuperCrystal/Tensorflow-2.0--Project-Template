import numpy as np
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras import models, layers

# data pre-processing
data = pd.read_csv('data.csv', index_col=False)
x = data['x'].to_numpy()
y = data['y'].to_numpy()
print(x.shape)
print(y.shape)

# model definition
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(1,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

# model building
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['AUC'])

# model training
history = model.fit(x, y,
                    batch_size=2,
                    epochs=30,
                    validation_split=0.2)

# model evaluation
import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

# plot_metric(history,"loss")

# model test
# predict score
model.predict([1,2,3])
# predict class
# model.predict_classes([1,2,3])  # 即将弃用
print(np.argmax(model.predict(np.array([63,32])), axis=-1))

# model saving
# 仅保存权重张量，使用时需重新定义模型并进行权重加载
model.save_weights('./weights/mini.ckpt', save_format='tf')

# 保存模型和权重
model.save('./weights/mini_model', save_format=tf)
# 恢复
model_loaded = tf.keras.models.load_model('./weights/mini_model')
print(model_loaded.evaluate(np.array([12,39])))