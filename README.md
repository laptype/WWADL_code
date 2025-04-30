# WWADL_code


# No Module named “causal_conv1d_cuda”

1. https://blog.csdn.net/IT_AGENT_PY/article/details/144576282
2. pip install causal-conv1d will compile the CUDA part. (这个可能有效？)
https://github.com/Dao-AILab/causal-conv1d/issues/9
[Mamba windows 环境安装踩坑](https://gitcode.csdn.net/66c9bc121338f221f9235f34.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6Njc3MjMzLCJleHAiOjE3MzczNjMwMzIsImlhdCI6MTczNjc1ODIzMiwidXNlcm5hbWUiOiJMYVB0XyJ9.Hgn40iMpukN-M6YjdOEh2qi8E7h0CHyzO77lIwrmnqc&spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-3-139382288-blog-144576282.235%5Ev43%5Epc_blog_bottom_relevance_base1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-3-139382288-blog-144576282.235%5Ev43%5Epc_blog_bottom_relevance_base1&utm_relevant_index=4)


docker 安装：
https://zhuanlan.zhihu.com/p/675938110


causal-conv1d

[quesion](https://stackoverflow.com/questions/78199621/typeerror-causal-conv1d-fwd-incompatible-function-arguments)


[修改setup代码 link](https://github.com/state-spaces/mamba/issues/40#issuecomment-1849095898)


完整步骤
1. 确保cuda118

```bash
# clone video-mamba-suite
git clone --recursive https://github.com/OpenGVLab/video-mamba-suite.git

# create environment
conda create -n video-mamba-suite python=3.9
conda activate video-mamba-suite

# install pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install h5py pandas scipy torchinfo

# install requirements
pip install -r requirement.txt

# install mamba
cd causal-conv1d
#python setup.py develop
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
cd mamba
python setup.py develop
# MAMBA_FORCE_BUILD=TRUE pip install .
cd ..
```

```angular2html
# create environment
conda create -n video-mamba-suite python=3.9
conda activate video-mamba-suite

# install pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

cd /root/shared-nvme/causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```