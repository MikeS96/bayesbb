# bayesbb
Implementation of Bayes By Backprop - PGM Fall 21

```
# Install Venv
python3 -m pip install --user virtualenv
# Create Env
python3 -m virtualenv <bname>
# Activate env
source <bname>/bin/activate
```

Install Pytorch 1.10
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install matplotlib
pip3 install seaborn
pip3 install neptune-client
pip install neptune-sacred
pip install sacred
pip install pymango
pip3 install tqdm
pip install -U scikit-learn
```

Adding Neptune API KEY to variables
```
echo export 'NEPTUNE_API_TOKEN="<NEPTUNE_API_KEY_HERE>"' >> ~/.bashrc
```