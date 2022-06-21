pip install -r requirements.txt
apt-get -y update
apt-get install git
apt-get install nano
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
bash
