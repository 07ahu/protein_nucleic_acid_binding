#This code grabs these outputs: binding pairs less than 10 Å between protein and nucleic acid, plddt, iptm, ptm,  pae, mpdockq, pdockq score
# Version: 9/25/24

## Class for the AlphaPulldown Outputs grabbing LIS score, distance, contact_pair, and average PAE score
import json
import numpy as np
from Bio.PDB import PDBParser
from itertools import accumulate
import pandas as pd
from scipy.spatial.distance import cdist
import math
import pandas as pd
import os
from absl import flags, app, logging


FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', '.', 'directory where predicted models are stored')
# Example to run: python3 outputs_a3.py -output_dir = /Users/hualj/Desktop/A3/test_outputs/
class af3_outputs:
    def __init__(self, summary_json_file_name: str, full_json_file_name: str, cif_file: str):
        """
        Initialize the AlphafoldOutput class with the provided JSON/CIF files.

        Args:
        - summary_json_file_name (str): Path to the summary json file
        - full_json_file_name (str): Path to the full json file
        - cif_file (str): Path to the cif file
        """
        
        # self.data contains the scores for the whole structure: pae, iptm, ptm (from summary json file)
        with open(summary_json_file_name, 'r') as file:
            self.data = json.load(file)  # 'data' will be a Python dictionary

        # self.data2 contains scores for individual atoms (from full json file)
        with open(full_json_file_name, 'r') as file:
            self.data2 = json.load(file)  # 'data2' will be a Python dictionary
        
        self.cif_data = self._load_cif_data(cif_file) # cif data is an AF3 output that includes coordinates, residue number, model letter, etc.
        self.atom_df = self._load_atom_data() # Creates dataframe with indiviudal atom data for each atom
        self.scores_df = self._load_score_data(full_json_file_name) # Creates dataframe with scores 
        self.atom_atom_contacts = self.calculate_structural_metric() # List with coordinates of A and B that are within threshold distance


    def _load_cif_data(self, cif_file):
        """
        Load CIF file data and label columns

        Args:
        - cif_file (str): string with path file to the cif_file (ranked 0)
        
        Returns:
        - cif_data: dataframe with columns as listed below
        """
        cif_data = []
        with open(cif_file, 'r') as cif:
            start_reading = False
            for line in cif:
                # Step 3: Start reading when we reach the atomic coordinates section
                if line.startswith('_atom_site.pdbx_PDB_model_num'):
                    start_reading = True
                # Stop reading when the coordinates section ends (empty line or next block)
                elif start_reading and line.strip() == '':
                    break
                elif start_reading:
                    cif_data.append(line.strip().split())
        columns = [ 'loop','group_PDB', 'id', 'type_symbol','atom_id',
           'alt_id','comp_id','asym_id','entity_id', 'seq_id'
           'PDB_ins_code','x_coord','y_coord',
           'z_coord','occupancy','iso_or_equiv','seq_id','asym_id','PDB_num'
        ]
        cif_data = pd.DataFrame(cif_data, columns=columns)
        
        print("SEQUENCE ID",cif_data['seq_id'][0])
        
        return cif_data

    def _load_atom_data(self):
        """
        Load the JSON data (atom chain id, plddt, coordinates) of each atom into a pandas DataFrame.

        Returns:
        - atoms (pd.DataFrame): DataFrame containing atom information.
        """
        atoms = []
        for i in range(len(self.data2['atom_chain_ids'])):
            atoms.append({
                'atom_chain_ids': self.data2['atom_chain_ids'][i],
                'atom_plddts': self.data2['atom_plddts'][i],
                'x': float(self.cif_data['x_coord'][i]), 
                'y': float(self.cif_data['y_coord'][i]),
                'z': float(self.cif_data['z_coord'][i]),
            })
        return pd.DataFrame(atoms)
    
    def _load_score_data(self, json_file):
        """
        Loads summary JSON file data into pandas DataFrame (scores of whole structure: iptm, ptm, mpdockq, pdockq, and average plddt)
        
        Returns:
        - scores (pd.DataFrame): DataFrame containing scores
        """
        scores = []
        scores.append({
                'iptm': self.data['iptm'],
                'ptm': self.data['ptm'],
                'mpdockq': self.calculate_mpDockQ(self.score_complex()),
                'pdockq': self.calc_pdockq(),
                'average_plddt': self.average_plddt(),
                'average_pae': self.get_pae(),
                'contact_pairs': self.get_contacts(),
                'title': json_file.split('/')[-2] 
        })
        return pd.DataFrame(scores)

    def get_contacts(self):
        # Filter atoms belonging to fragment A and fragment B
        fragment_a_atoms = self.atom_df[self.atom_df['atom_chain_ids'] == 'A']
        fragment_b_atoms = self.atom_df[self.atom_df['atom_chain_ids'] == 'B']

        # Extract coordinates and other relevant information
        coordinates_a = fragment_a_atoms[['x', 'y', 'z']].values
        coordinates_b = fragment_b_atoms[['x', 'y', 'z']].values

        # Calculate distances using cdist for efficiency
        distances = cdist(coordinates_a, coordinates_b)

        # Find contacts within the threshold
        contacts = distances <= 4.0

        # Get the indices of the 'True' values in the contacts array
        a_indices, b_indices = contacts.nonzero()

        # Extract the corresponding coordinates using the indices
        a_coords = coordinates_a[a_indices]
        b_coords = coordinates_b[b_indices]
        print('old lenght', len(a_coords))
        # Ensure unique contact pairs
        unique_pairs = set()

        for a, b in zip(a_coords, b_coords):
            # Round coordinates to 3 decimal places to account for floating-point precision
            a_rounded = tuple(round(coord, 3) for coord in a)
            b_rounded = tuple(round(coord, 3) for coord in b)
            
            # Sort each pair so (a, b) and (b, a) are treated as the same
            pair = tuple(sorted((a_rounded, b_rounded)))
            unique_pairs.add(pair)

        # Convert the set back to a DataFrame if you need it in that format
        unique_contact_df = pd.DataFrame(unique_pairs, columns=['a_coords', 'b_coords'])

        print(unique_contact_df)
        print('new_length', len(unique_contact_df))
        return len(a_coords)

    def get_pae(self):
        """
        Calculates average PAE score by averaging only the interacting areas of protein and DNA
        
        Returns:
        - average (float): average of the two areas that include protein and DNA
        """

        # Get pae lists from data2 
        # pae_lists is a matrix where each list (within the larger list) is a row in the PAE plot
        pae_lists = pd.DataFrame(self.data2['pae'])

        # Finding the indexes of the atom_df that are of the 'A' (protein) model
        a_protein = self.atom_df[self.atom_df['atom_chain_ids'] == 'A'].index.to_numpy()
        
        # Get pae lists from data2 
        # pae_lists is a matrix where each list (within the larger list) is a row in the PAE plot
        pae_lists = pd.DataFrame(self.data2['pae'])

        # residues_a gets the index where 'A' changes to 'B' (protein to DNA) by getting the last residue number of 'A' 
        # e.g. 1-64 are residue numbers of the protein, 1-11 are residue numbers of DNA
        residues_a = int(max(np.array(self.cif_data['seq_id'].loc[a_protein]).astype(float)))
        
        # rows1 and columns1 contain an area of the bottom left corner of a PAE plot where DNA and protein interact
        rows1 = slice(0,residues_a+1)
        columns1 = slice(residues_a,len(pae_lists))

        # rows2 and columns2 contain the area in the top right corner of the PAE plot where DNA and protein interact
        rows2 = slice(residues_a,len(pae_lists))
        columns2 = slice(0,residues_a+1)

        # Extracts from the pae matrix the bottom left and top right pae values needed
        selected_data = pae_lists.iloc[rows1, columns1]
        selected_data_2 = pae_lists.iloc[rows2, columns2]

        # Add all the values of each, so we can average it
        sum1 = selected_data.to_numpy().sum()
        sum2 = selected_data_2.to_numpy().sum()

        # number_of_numbers contains all the amount of numbers in the selected_data matrices
        # e.g. 64x11 and 11x64
        number_of_numbers = residues_a*(len(pae_lists)-residues_a)
        # Average each sum then average the two sums to return
        average1 = sum1/number_of_numbers
        average2 = sum2/number_of_numbers

        return (average1+average2)/2
    
    def calculate_structural_metric(self, threshold=10.0):
        """
        Calculate the atom-atom contacts within the specified threshold distance.

        Args:
        - threshold (float): Distance threshold to consider atoms in contact.

        Returns:
        - contact_pairs (pd.DataFrame): contains dataframe of x, y, z coordinates of A and B where A and B are within threshold distance
        """
        # Filter atoms belonging to fragment A and fragment B
        fragment_a_atoms = self.atom_df[self.atom_df['atom_chain_ids'] == 'A']
        fragment_b_atoms = self.atom_df[self.atom_df['atom_chain_ids'] == 'B']

        # Extract coordinates and other relevant information
        coordinates_a = fragment_a_atoms[['x', 'y', 'z']].values
        coordinates_b = fragment_b_atoms[['x', 'y', 'z']].values

        # Calculate distances using cdist for efficiency
        distances = cdist(coordinates_a, coordinates_b)

        # Find contacts within the threshold
        contacts = distances <= threshold

        # Get the indices of the 'True' values in the contacts array
        a_indices, b_indices = contacts.nonzero()

        # Extract the corresponding coordinates using the indices
        a_coords = coordinates_a[a_indices]
        b_coords = coordinates_b[b_indices]

        # Create a DataFrame to store the contact pairs and their coordinates
        contact_pairs_df = pd.DataFrame({
            'A_x': a_coords[:, 0],
            'A_y': a_coords[:, 1],
            'A_z': a_coords[:, 2],
            'B_x': b_coords[:, 0],
            'B_y': b_coords[:, 1],
            'B_z': b_coords[:, 2]
        })
        return contact_pairs_df
    
    def score_complex(self):
        '''
        Score all interfaces in the current complex

        Modified from the score_complex() function in MoLPC repo:
        https://gitlab.com/patrickbryant1/molpc/-/blob/main/src/complex_assembly/score_entire_complex.py#L106-154

        Returns:
        - complex_score: value used for calculating mpdockq
        '''
        
        # AB_inds is structured like {'A': [array], 'B': [array]} where each array contains indexes of atoms that are either 'A' or 'B' model
        AB_inds = {
            'A': self.atom_df[self.atom_df['atom_chain_ids'] == 'A'].index.to_numpy(),
            'B': self.atom_df[self.atom_df['atom_chain_ids'] == 'B'].index.to_numpy()
        }
        chains = ['A', 'B']

        #chain_inds = [0,1]
        chain_inds = np.arange(len(chains))

        complex_score = 0

        # Get interfaces per chain
        for i in chain_inds:
            # Iterates through 0 and 1
            
            chain_i = chains[i]
            # chain_i goes through 'A' then 'B'

            chain_df = (self.atom_df[self.atom_df['atom_chain_ids'] == chain_i])
            #Dataframe with rows of atom_df if the chain matches chain_i (A and B)

            chain_coords = chain_df[['x', 'y', 'z']].values
            # Creates new DF with [[x,y,z], [x,y,z]...[x,y,z]] (coordinates of atoms with the chain_i model)

            chain_AB_inds = AB_inds[chain_i]
            # List of indexes for the certain chain_i (A or B)

            l1 = len(chain_AB_inds)
            chain_AB_coords = chain_coords
            # list with [x,y,z] of 'A' or "B" same as chain_coords

            chain_plddt = chain_df['atom_plddts'].values

            for int_i in np.setdiff1d(chain_inds, i):
                # Iterating while excluding previously used i value (1 or 0)
                
                int_chain = chains[int_i]
                # int_chain contains the model letter ('A' or 'B') that isn't the one previously used
                
                int_chain_df = (self.atom_df[self.atom_df['atom_chain_ids'] == int_chain])
                # Dataframe with everything with the according model (opposite of previous for loop)

                int_chain_coords = int_chain_df[['x', 'y', 'z']].values
                #creates new DF with [['x',y,z], [x,y,z]...[x,y,z]] (coordinates of atoms that match the model letter: int_chain)

                int_chain_AB_coords = int_chain_coords
            
                int_chain_plddt = int_chain_df['atom_plddts'].values
                
                # Calc 2-norm
                mat = np.append(chain_AB_coords, int_chain_AB_coords, axis=0)
                a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
                dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
                contact_dists = dists[:l1, l1:]
                contacts = np.argwhere(contact_dists <= 8)
                # The first axis contains the contacts from chain 1
                # The second the contacts from chain 2
                if contacts.shape[0] > 0:
                    av_if_plDDT = np.concatenate((chain_plddt[contacts[:, 0]], int_chain_plddt[contacts[:, 1]])).mean()
                    complex_score += np.log10(contacts.shape[0] + 1) * av_if_plDDT

        return complex_score

    def calculate_mpDockQ(self, complex_score):
        """
        A function that returns a complex's mpDockQ score after
        calculating complex_score
        =average interface plDDT times the logarithm of the number of interface contacts,
        """
        """
        L = 0.827
        x_0 = 261.398
        k = 0.036
        b = 0.221
        
        Changed to values below based on this article: https://www.nature.com/articles/s41467-022-33729-4
        """
        L = 0.728
        x_0 = 309.375
        k = 0.098
        b = 0.262
        return L / (1 + math.exp(-1 * k * (complex_score - x_0))) + b

    def calc_pdockq(self):
        '''Calculate the pDockQ scores
        pdockQ = L / (1 + np.exp(-k*(x-x0)))+b
        L= 0.724 x0= 152.611 k= 0.052 and b= 0.018

        Modified from the calc_pdockq() from FoldDock repo:
        https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py#L62
        '''

        # Get coordinates and plddts per chain
        a_df = (self.atom_df[self.atom_df['atom_chain_ids'] == 'A'])
        b_df = (self.atom_df[self.atom_df['atom_chain_ids'] == 'B'])
        coords1, coords2 = a_df[['x', 'y', 'z']].values, b_df[['x', 'y', 'z']].values
        plddt1, plddt2 = a_df['atom_plddts'].values, b_df['atom_plddts'].values
        # Calc 2-norm
        mat = np.append(coords1, coords2, axis=0)
        a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
        dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
        l1 = len(coords1)
        contact_dists = dists[:l1, l1:]  # upper triangular --> first dim = chain 1
        contacts = np.argwhere(contact_dists <= 10.0) #threshold

        if contacts.shape[0] < 1:
            pdockq = 0
        else:
            # Get the average interface plDDT
            avg_if_plddt = np.average(
                np.concatenate([plddt1[np.unique(contacts[:, 0])], plddt2[np.unique(contacts[:, 1])]]))
            # Get the number of interface contacts
            n_if_contacts = contacts.shape[0]
            x = avg_if_plddt * np.log10(n_if_contacts)
            pdockq = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

        return pdockq
    
    def average_plddt(self):
        '''Average all the plddt scores to get one value that can estimate overall confidence on prediction
        '''
        plddt_values = self.atom_df['atom_plddts'].values
        length = len(plddt_values)
        s = sum(plddt_values)
        return s/length

    def __call__(self):
        '''
        Returns dictionary with all the data from calling each function
        Returns:
        - final_dict: contains the alphafold3 outputs into a dictionary to be called in other code
        '''
        final_dict = {
            'atom_df': self.atom_df, # dataframe with row for each atom
            'score_df': self.scores_df, # dataframe with 1 row of scores
            'contact_pairs': self.atom_atom_contacts  # dataframe of coordinates
        }
        return final_dict

def create_and_update_excel(test, excel_path):
    """
        Creates excel file if it doesn't exist yet and adds a row with the information from scores dataframe
    """
    # Define the file path for the Excel file
    excel_file = excel_path
    # Define the columns for the Excel file
    columns = ['iptm', 'ptm', 'mpdockq', 'pdockq', 'average_plddt', 'average_pae', 'contact_pairs', 'title']

    # Check if the file exists; if not, initialize it
    if not os.path.exists(excel_file):
        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)
        # Save the empty DataFrame to create the Excel file
        df.to_excel(excel_file, index=False)
        print("Excel file initialized.")

    # Extract data from the test function
    dictionary = test()
    scores_df = dictionary['score_df']
    

    # Extract data corresponding to the defined columns
    data = {col: scores_df[col].values[0] if isinstance(scores_df[col], pd.Series) else scores_df[col] for col in columns}

    # Create a DataFrame from the extracted data
    new_row = pd.DataFrame([data])

    # Read the existing Excel file and append the new data
    df = pd.read_excel(excel_file)
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to the Excel file
    df.to_excel(excel_file, index=False)


def main(argv):
    # Parse flags
    FLAGS(argv)  # This line is crucial to parse the flags

    jobs = os.listdir(FLAGS.output_dir)  # Grabbing the directory of where all the fragments are
    count = 0 # Count will help create images with the correct fragment number 

    for job in jobs:
        # job = fragment 1...x
        print(job)
        try:
             # takes the ranked 0 file of the two jsons and cif to be input into class
            json_full = os.path.join(FLAGS.output_dir, job, str(job)+'_full_data_0.json')
            json_sum = os.path.join(FLAGS.output_dir,job,str(job)+'_summary_confidences_0.json')
            cif = os.path.join(FLAGS.output_dir,job,str(job)+'_model_0.cif')
            # We use try so that we skip other files such as output.xlsx that are not fragments
            test = af3_outputs(json_sum,json_full,cif)

            # Now you can access FLAGS.output_dir without errors
            excel_path = os.path.join(FLAGS.output_dir, 'output.xlsx')

            # Example function call using the parsed flag
            create_and_update_excel(test, excel_path)
        
        except Exception as e:
            logging.error((f"An error occurred while loading result_{job}.pkl.gz: {e}"))

######
# Entry point
if __name__ == '__main__':
    app.run(main)

"""
for testing:
print("ATOM DF: ", test._load_atom_data())
print("SCORE DF: ", test._load_score_data())
print("CONTACT PAIR COORDINATES: ", test.calculate_structural_metric())
print("PAE AVERAGE: ",test.get_pae())
print('average plddt: ', test.average_plddt())
    dictionary = test() # contains final_dict
    atoms = dictionary['atom_df'] # provides atom_df
"""
