# Install detectron2


conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

cd detectron2
