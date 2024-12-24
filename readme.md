### **1.配置环境**

环境配置：Anaconda 2.6 +pycharm 3.8 +tensorflow 1.x

#### 1.1 需要下载

在配置环境前你需要 在官网下载 Anaconda 及 Pycharm(专业版）

#### 1.2 安装Anaconda 并创建虚拟环境

##### 1.2.1安装Anaconda，正常装即可（注意：长远考虑不要安装在C盘）

##### 1.2.2 创建虚拟环境

TUNA 还提供了 Anaconda 仓库与第三方源（conda-forge、msys2、pytorch等，查看完整列表，更多第三方源可以前往校园网联合镜像站查看）的镜像，各系统都可以通过修改用户目录下的 .condarc 文件来使用 TUNA 镜像源。Windows 用户无法直接创建名为 .condarc 的文件，可先执行

```
 conda config --set show_channel_urls yes 
```

生成该文件之后再修改。注：由于更新过快难以同步，我们不同步pytorch-nightly, pytorch-nightly-cpu, ignite-nightly这三个包。

在所有应用中找到 Anaconda，点开终端，输入 

```
conda config --set show_channel_urls yes
```

 终端运行后会生成 .condarc文件该文件的路径为：C:\用户\username 用记事本打开，并将下面的代码粘贴复制到该文件中。

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
```

执行完上述操作后，打开终端，键入 

```
conda create -n XXX python=3.8
```

其中，XXX为你想要创建的虚拟环境的名称，python的版本也可以自行更改。

#### 2.打开终端并运行代码

点击侧边栏 终端，激活虚拟环境：

```
conda activate XXX
```

计入虚拟环境后，运行 cifar-10.py 和 new_cifar-10.py 即可

CIFAR-10 数据集在 TensorFlow 中内置，程序会自动下载和加载它。