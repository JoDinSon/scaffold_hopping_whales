# Contains all the necessary code to prepare the molecule:
#   - molecule sanitization (check in "import_prepare_mol" to change advanced sanitiization settings")
#   - geometry optimization (if specified by "do_geom = True"), with the specified settings

from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit import Chem
import numpy as np


def prepare_mol_from_sdf(filename_in, do_geometry=True, do_charge=False, property_name='_GasteigerCharge', max_iter=1000,
                       mmffvariant='MMFF94', seed=26, max_attempts=100):

    vs_library = Chem.SDMolSupplier(filename_in)
    vs_library_prepared = []

    cnt = 0
    nmol = len(vs_library)

    for mol in vs_library:
        cnt += 1
        if cnt % 50 == 0:
            print('Molecule: ' + str(cnt))

        mol, err = prepare_mol(mol, do_geometry, do_charge, property_name, max_iter, mmffvariant, seed, max_attempts)

        if err == 1:
            print ('Molecule ' + str(cnt) + ' of ' + str(nmol) + ' not computed.')
        vs_library_prepared.append(mol)
    return vs_library_prepared

def prepare_mol(mol, do_geometry=True, do_charge=True, property_name='_GasteigerCharge', max_iter=1000,
                       mmffvariant='MMFF94', seed=26, max_attempts=5):

    # 'mmffVariant : “MMFF94” or “MMFF94s”'
    # seeded coordinate generation, if = -1, no random seed provided
    # removes starting coordinates to ensure reproducibility
    # max attempts, to increase if issues are encountered during optimization

    if do_charge is True:
        property_name = '_GasteigerCharge'

    # options for sanitization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE

    # sanitization
    if mol is None:
        err = 1
    else:
        # sanitize
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_fail:
            raise ValueError(sanitize_fail)
            err = 1

        if do_geometry is True:
            mol, err = opt_geometry(mol, max_iter, mmffvariant, seed, max_attempts)

        # calculates or assigns atom charges based on what annotated in do_charge
        mol = rdmolops.RemoveHs(mol)

        if do_charge is True:
            mol, name, err = get_charge(mol, property_name, do_charge)

    if err == 1:
        print('Error in molecule pre-treatment')
        
    return mol, err


def opt_geometry(mol, max_iter, mmffvariant, seed, max_attempts):

    err = 0
    try:
        mol = rdmolops.AddHs(mol)
        a = AllChem.EmbedMolecule(mol, useRandomCoords=True, useBasicKnowledge=True, randomSeed=seed, clearConfs=True, maxAttempts=max_attempts)
        if a == -1:
            err = 0

        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iter, mmffVariant=mmffvariant)
    except ValueError:
        err = 1
    except TypeError:
        err = 1

    return mol, err


def get_charge(mol, property_name, do_charge):

    from rdkit.Chem import AllChem as Chem
    err = 0

    # partial charges
    if do_charge is False:
        err = check_mol(mol, property_name, do_charge)
        if err == 0:
            # prepares molecule
            mol = Chem.RemoveHs(mol)
            n_at = mol.GetNumAtoms()
            # takes properties
            list_prop = mol.GetPropsAsDict()
            string_values = list_prop[property_name]  # extracts the property according to the set name
            string_values = string_values.split("\n")
            w = np.asarray(map(float, string_values))
        else:
            mol = Chem.AddHs(mol)
            n_at = mol.GetNumAtoms()
            w = np.ones((n_at, 1)) / n_at
            w = np.asarray(map(float, w))  # same format as previous calculation
            property_name = 'equal_w'
            err = 0
        # extract properties
        for atom in range(n_at):
            mol.GetAtomWithIdx(atom).SetDoubleProp(property_name, w[atom])

        mol = Chem.RemoveHs(mol)

    # Gasteiger-Marsili Charges
    elif (do_charge is True) and (err is 0):
        Chem.ComputeGasteigerCharges(mol)
        err = check_mol(mol, property_name, do_charge)

    return mol, property_name, err


# ----------------------------------------------------------------------------------------------------------------------
def check_mol(mol, property_name, do_charge):
    """
    checks if the property (as specified by "property_name") is annotated and gives err = 0 if it is
    """
    n_at = mol.GetNumAtoms()
    if do_charge is False:
        list_prop = mol.GetPropsAsDict()
        string_values = list_prop[property_name]  # extracts the property according to the set name
        if string_values == '' or string_values == ['']:
            err = 1
        else:
            err = 0
    else:
        from rdkit.Chem import AllChem as Chem
        err = 0
        atom = 0
        while atom < n_at:
            value = mol.GetAtomWithIdx(atom).GetProp(property_name)
            # checks for error (-nan, inf, nan)
            if value == '-nan' or value == 'nan' or value == 'inf':
                err = 1
                break

            atom += 1

    # checks for the number of atoms
    if n_at < 4:
        err = 1

    return err


# ----------------------------------------------------------------------------------------------------------------------
def do_map(mol, fig_name=None, lab_atom=False, text=False, MapMin=0, MapMax=1):

    from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    # Setup parameters
    scale = -1  
    colmap = 'bwr'


    mol, charge_prop, err = get_charge(mol, property_name='_GasteigerCharge', do_charge=True)
    if err == 1:
        print('Error in charge calculation')
        return

    n_at = mol.GetNumAtoms()
    # Calcualte charge for each atom using list comprehension
    charge = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(n_at)]

    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.backgroundColour = (1, 1, 1)
    
    if not lab_atom:
        opts.addAtomIndices = False

    # 3. Generate the Map
    # We pass the drawer object directly to handle the rendering
    SimilarityMaps.GetSimilarityMapFromWeights(mol, charge, 
                                              draw2d=drawer,
                                              colorMap=colmap, 
                                              scale=scale)

    # 4. CRITICAL: Finish drawing and convert to an image format Matplotlib can see
    drawer.FinishDrawing()
    img_binary = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_binary))

    # 5. Display using Matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")


    if fig_name is not None:
        img.save(fig_name)

    return plt.show()


def frequent_scaffolds(suppl, output_type='supplier'):
    """
     starting from a supplier file, the function computes the most frequently recurring scaffolds and returns them as a
     supplier file (if output_type='supplier') or as a counter file.
     """

    from collections import Counter
    scaff_list = []
    for mol in suppl:
        scaff_list.append(Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol))

    freq_scaffolds = Counter()
    for scaff in scaff_list:
        freq_scaffolds[scaff] += 1

    freq_scaffolds = freq_scaffolds.most_common()

    if output_type is 'supplier':
        # converts it back in a supplier file,
        suppl_new = []
        for row in freq_scaffolds:
            mol = Chem.MolFromSmiles(row[0])
            mol.SetProp("_Name", str(round((row[1]/len(suppl))*100,2))+'%') # assigns the molecule name as the percentage occurrence
            suppl_new.append(mol)

        freq_scaffolds = suppl_new


    return freq_scaffolds
