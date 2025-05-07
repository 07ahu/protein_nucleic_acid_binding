# protein_nucleic_acid_binding

First, we need a TRAINING dataset of different proteins for you to run the model. 

We need data of nucleic acid and protein bindings along with their Kd values and sequences.
We used: (http://www.pdbbind.org.cn/)[http://www.pdbbind.org.cn/]

To use this website:
1. Make an account + log in
2. Navigate to "DATA" found on the top navigation bar
3. Change "Search in" from "All Complexes" to "Protein-Nucleic Acid Complexes and press "Search"
4. Choose as many complexes as you want. We used a dataset of around 100. To get the sequences and Kd, click on the protein PDB number link.
5. The Protein and Nucleic Acid sequences are found in "Protein/NA Sequence" by clicking "Check fasta file" *KEEP THESE FOR THE NEXT STEPS
6. The Kd is found in "Affinity (Kd/Ki/IC50)" Make sure the units are Nm, if not convert them!
7. Create a spreadsheet and make sure to add the number from step 6 into a column called "kd". Also, make a column called "title" with the PDB ID.

Time to use the sequences you found to put through AlphaFold

1. Navigate to (https://alphafoldserver.com/)[https://alphafoldserver.com/] and log in.
2. Press "+ Add entity"
3. Change the second "Entity type" to DNA or RNA depending on what it was (you can check in the fasta file with the sequences) And make sure the first "Entity type" is Protein
4. Enter the Protein sequence and the DNA/RNA sequence 5'--3'
5. Press "Continue and preview job" Make sure to title it the PDB ID, or something identifiable 
6. Once it finishes running, click into it and press "Download" at the top.
7. It will download a folder with around 17 files. An example can be found (here)[example]

REPEAT THESE 14 STEPS ABOVE FOR ALL THE COMPLEXES YOU WANT AND MAKE SURE THE DOWNLOADED FOLDERS ARE ALL WITHIN ONE FOLDER


An example of a spreadsheet that is input into the model can be found (here)[Data_spreadsheet.xlsx]
Make sure you have these column titles: iptm,	ptm,	mpdockq,	pdockq,	average_plddt,	average_pae,	contact_pairs,	title,	kd


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

# Next time you want to run the code, don't do this all over, just run this:
```bash
conda init
conda activate my_env
```

For the model:
python3 softmax2.py
