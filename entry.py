import os
import json
import socket
import yaml

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    num_hosts = len(hosts)
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    num_gpus = os.environ['SM_NUM_GPUS']
    
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
    # os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1' # only support P4d
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'
        
        
    file_name = './hostfile.txt'

    hostfile_info = []
    for i in range(len(hosts)):
        host_ip = socket.gethostbyname(hosts[i])
        hostfile_info.append(str(host_ip) + ' slots=' + str(num_gpus))

    with open(file_name, 'w') as f:
        f.write('\n'.join(hostfile_info))

    print("-----hosts-----")
    print('\n'.join(hostfile_info))

    os.system("aws s3 sync {} {} --exclude global_step120/* --no-progress".format(os.environ['PRETRAINED_MODEL_S3_PATH'], '/tmp/pretrain_model'))
    print("Finish download pretrained model from {}.".format(os.environ['PRETRAINED_MODEL_S3_PATH']))
    os.system("ls /tmp/pretrain_model")

    os.system("pip install -e .[deepspeed,metrics,bitsandbytes,qwen]")
    os.system("pip install --upgrade deepspeed==0.14.0")

    os.system("cp {} {}".format(file_name, "./examples/full_multi_gpu"))
    os.chdir("./examples/full_multi_gpu")
    os.system("chmod +x ./multi_node_sft.sh")
    os.environ["NUM_GPUS"] = str(num_gpus)
    os.environ["NUM_NODES"] = str(num_hosts)
    os.environ["HOSTFILE"] = str(file_name)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.system("/bin/bash -c ./multi_node_sft.sh")

    
    
    
