## DRGAN 
is a framework for doctor recommendation using GAN.

#Content

* ask120_data:    针对ask120, 存放原始、预处理、中间结果数据的目录
    * 46.sql:  从数据库导出数据的脚本
* xywy_data:      同上
* model:   存放保存下来的模型参数
* src:  源代码
    * ask120_dataPrepare.py & xywy_dataPrepare.py     对数据集进行预处理，供模型训练和测试，依次调用__main__中的 extract_data_with_best_answer1，feature_process2，split_train_test3，generate_uniform_pair，generate_test_samples （或者generate_test_random_samples）
    * drgan_train.py     DRGAN 模型的训练和测试
    * cfgan_train.py     IRGAN-ir 模型的训练和测试
    * qagan_train.py     IRGAN-qa 模型的训练和测试
    * core.py            公共模块，QACNN 和评分函数的实现
    * util.py            公共模型，测试评价指标的实现
    * Discriminator.py   公共模型，以core.py 为基础的判别器模型
    * Generator.py       公共模型，以core.py 为基础的生成器模型
    * cf                 为IRGAN-ir 模型 提供调用的模块文件夹
    * log                记录训练和测试中间数据的日志文件夹
