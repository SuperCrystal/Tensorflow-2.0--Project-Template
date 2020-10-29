# Tensorflow-2.0--Project-Template
This is the project template for tensorflow-2.0 based projects.

pipeline如下：

模型定义，数据读取，optimizer，训练过程，metric，validate

文件分布如下：

- utils
  - config.py: process configs | 从json文件处理为dict，或者bunch
  - dirs.py: process dirs | 处理和dir相关的
  - logger.py: wrapper of tensorboard writer | 
  - utils.py: process args
- mains
  - main.py: train & validate the model
- models
  - model.py: model definition
- trainers
  - trainer.py: define train_step and train_epoch
- configs
  - config.json
- data_loader
  - data_generator.py

![diagram](https://gitee.com/cry-star/pics/raw/master/img/diagram.png)