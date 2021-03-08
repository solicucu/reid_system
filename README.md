## 轻量快速行人重识别演示系统程序说明

Server:本代码已经集成本项目提出的核心算法SSNet及前端展示为一体。

### 前后端部分

采用的是Django后端框架来实现演示系统。

可以在命令行执行：python manage.py runserver 或者在pycharm直接运行项目

打开链接：<http://127.0.0.1:8000/reid/login/> 就可以进入登陆界面，如果没有账号，可以先注册再登录方可使用行人重识别服务。

服务界面大概如下，在第一个选项可以选择不同的模型，包括提出的ssnet。

![avatar](/imgs/show.png)



### 算法部分（./Server/algorithm）

在model里面有我们的ssnet和其他模型的详细实现，整个算法的代码运行入口在run文件夹，有很多.sh，运行脚本ssnet_train.sh 或者anynet.sh 均可以得到我们模型性能度量的结果。

