本项目使用3755类手写识别数据集进行汉字识别，方法为基于resnet50的标签映射算法实现，具体算法细节可以查看论文Large scale classifification in deep neural network with Label Mapping，论文地址https://arxiv.org/abs/1806.02507

代码在windows下可以直接运行decode_3755.ipynb，所需模型已训练好，放在new_models文件夹下。代码在linux下需要对map函数和eval函数做出相应删改，在代码中已经标注。

数据集已上传至百度网盘，链接：https://pan.baidu.com/s/1Oc2FqXBu1DV_OPj6brbCWA?pwd=hd2p 
提取码：hd2p。

环境配置为python3.7 + pytorch 1.12.0 + torchvision 0.13.0