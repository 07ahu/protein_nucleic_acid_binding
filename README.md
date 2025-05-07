# protein_nucleic_acid_binding

First in order to run all the code we need to, we must install conda. If you don't already have it downloaded, follow these steps:

1.
For intel mac:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```
For Apple Silicon (M1/M2) Macs:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

2. Run installer
```bash
bash Miniconda3-latest-MacOSX-*.sh
```

3. Start an environment named my_env
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate
conda create --name my_env
conda activate my_env
```

4. Download every package:
```bash
conda install pandas
conda install matplotlib
conda install keras -c conda-forge
pip install tensorflow
conda install scikit-learn
conda install openpyxl
```

Next time you want to run the code, don't do this all over, just run this:
```bash
conda init
conda activate my_env
```

Now, we need data of different proteins for you to run/test your model. 

An example of a spreadsheet that is input into the model can be found (here)[Data_spreadsheet.xlsx]
Make sure you have these column titles: iptm,	ptm,	mpdockq,	pdockq,	average_plddt,	average_pae,	contact_pairs,	title,	kd

For the model:
python3 softmax2.py
