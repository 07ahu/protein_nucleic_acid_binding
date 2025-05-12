# protein_nucleic_acid_binding

First, we need a TRAINING dataset of different proteins for you to run the model. 

We need data of nucleic acid and protein bindings along with their Kd values and sequences.
We used: [http://www.pdbbind.org.cn/](http://www.pdbbind.org.cn/)

To use this website:
1. Make an account + log in
2. Navigate to "DATA" found on the top navigation bar
3. Change "Search in" from "All Complexes" to "Protein-Nucleic Acid Complexes and press "Search"
4. Choose as many complexes as you want. We used a dataset of around 100. To get the sequences and Kd, click on the protein PDB number link.
5. The Protein and Nucleic Acid sequences are found in "Protein/NA Sequence" by clicking "Check fasta file" *KEEP THESE FOR THE NEXT STEPS
6. The Kd is found in "Affinity (Kd/Ki/IC50)" Make sure the units are Nm, if not convert them!
7. Create a spreadsheet and make sure to add the number from step 6 into a column called "kd". Also, make a column called "title" with the PDB ID.

Time to use the sequences you found to put through AlphaFold

1. Navigate to [https://alphafoldserver.com/](https://alphafoldserver.com/) and log in.
2. Press "+ Add entity"
3. Change the second "Entity type" to DNA or RNA depending on what it was (you can check in the fasta file with the sequences) And make sure the first "Entity type" is Protein
4. Enter the Protein sequence and the DNA/RNA sequence 5'--3'
5. Press "Continue and preview job" Make sure to title the job the corresponding PDB ID.
6. Once it finishes running, click into it and press "Download" at the top.
7. It will download a folder with around 17 files. An example can be found [here](examples/fold_2g4b)

REPEAT THESE 14 STEPS ABOVE FOR ALL THE COMPLEXES YOU WANT AND MAKE SURE THE DOWNLOADED FOLDERS ARE ALL WITHIN ONE FOLDER
* repeat for not only all your training dataset complexes, but also your testing dataset for after the model is created.

Now make sure you have outputs_a3.py in that folder too, we're going to run that code!

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
If there are any left that you need to download search up how to conda install that package and you'll type in conda install ____

# Next time you want to run the code, don't do this all over, just run this:
```bash
conda init
conda activate my_env
```

To actually run the code, navigate into your folder first, then run:
```bash
cd path/to/your/folder/with/outputs_a3.py
python3 outputs_a3.py --output_dir=path/to/your/alphafold/output/folders
```
# This should result in a new file called output.xlsx

To create the final training spreadsheet, we want to combine your spreadsheet with Kd values with output.xlsx.
To do this, we want to make sure the titles correspond to fit the right Kd with the other data. 
You can sort both sheets via the title column and then merge by copy and pasting.

An example of the final spreadsheet that is input into the model can be found [here](examples/Data_spreadsheet.xlsx)
Make SURE you have these column titles: iptm,	ptm,	mpdockq,	pdockq,	average_plddt,	average_pae,	contact_pairs,	title,	kd

Now, we want to make a testing dataset that can be tried once the model is created!
Just take out a chunk of your current spreadsheet and move it to a new one called testing.xlsx
(You can also repeat the exact same steps above to make a new spreadsheet)

BUT, make sure your testing dataset does not have Kd values! Store them in another sheet to compare results later with. 

# Time to make the model!
For the model:
python3 softmax.py --dataset=path/to/training/dataset

this code has a threshold number - use these steps to make your own threshold.

# Try other models:
