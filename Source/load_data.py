#----------------------------------------------------------------------
# Script for loading CAMELS data
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import h5py
import readfof
from torch_geometric.data import Data, DataLoader
from Source.constants import *
from Source.plotting import *
import illustris_python as il

#--- FEATURES CHOICES ---#

#use_hmR = 1     # 1 for using the half-mass radius as feature
use_vel = 0     # 1 for using subhalo velocity as feature
#only_positions = 0  # 1 for using only positions as features
#galcen_frame = 0    # 1 for writing positions and velocities in the central galaxy rest frame (otherwise it uses the total center of mass)

#--- NORMALIZATION ---#

#Nstar_th = 10   # Minimum number of stellar particles required to consider a galaxy
#radnorm = 8.    # Ad hoc normalization for half-mass radius
#velnorm = 100.  # Ad hoc normalization for velocity. Use velnorm=1. for galcen_frame=1

#--- LOADING DATA ROUTINES ---#

# Import h5py file to construct the general dataset
# See for explanation of different field in Illustris files: https://www.tng-project.org/data/docs/specifications/#sec2a
# See also https://camels.readthedocs.io/en/latest/
def general_tab(path):

    basePath = '/scratch/gpfs/ab4671/TNG300-1/files'
    fields = ['SubhaloPos']
    subhalos = il.groupcat.loadSubhalos(basePath,99,fields=fields)
    
    
    # Read hdf5 file
    #f = h5py.File(path, 'r')

    # Load subhalo features
    SubhaloPos = subhalos#['SubhaloPos']  # since only one field, no need to index...
    print(subhalos.shape)
    SubhaloPos=SubhaloPos[:int(len(subhalos)/100000)]
    print(SubhaloPos.shape)
    #f["Subhalo/SubhaloPos"][:]/boxsize
    #SubhaloMassType = f["Subhalo/SubhaloMassType"][:,4]
    #SubhaloLenType = f["Subhalo/SubhaloLenType"][:,4]
    #SubhaloHalfmassRadType = f["Subhalo/SubhaloHalfmassRadType"][:,4]/radnorm
    #SubhaloVel = f["Subhalo/SubhaloVel"][:]/velnorm
    #SubhaloVel = np.sqrt(np.sum(SubhaloVel**2., 1))/velnorm
    #HaloID = np.array(f["Subhalo/SubhaloGrNr"][:], dtype=np.int32)

    # Load halo features
    #HaloMass = f["Group/GroupMass"][:]
    #HaloMass = f["Group/Group_M_Crit200"][:]
    #GroupPos = f["Group/GroupPos"][:]/boxsize
    #GroupPos = f["Group/GroupCM"][:]/boxsize
    #GroupVel = f["Group/GroupVel"][:]/velnorm

    # Neglect halos with zero mass
    #indexes = np.argwhere(HaloMass>0.).reshape(-1)

    #f.close()

    # Create general table with subhalo properties
    # Host halo ID, 3D position, stellar mass, number of stellar particles, stellar half-mass radius, 3D velocity
    #tab = np.column_stack((HaloID, SubhaloPos, SubhaloMassType, SubhaloLenType, SubhaloHalfmassRadType, SubhaloVel))

    #tab = tab[tab[:,4]>0.]          # restrict to subhalos with stars
    #tab = tab[tab[:,5]>Nstar_th]    # more or less equivalent to the condition above
    #tab[:,4] = np.log10(tab[:,4])   # take the log of the stellar mass

    # Once restricted to a minimum number of stellar particles, remove this feature since it is not observable
    #tab = np.delete(tab, 5, 1)
    
    #if not use_hmR:
    #    tab = np.delete(tab, 5, 1)  # remove SubhaloHalfmassRadType if not required

    #if only_positions:
    #    tab = np.column_stack((tab[:,0],tab[:,1],tab[:,2],tab[:,3]))

    tab = SubhaloPos
    
    return tab#, HaloMass, GroupPos, GroupVel, indexes

def ___general_tab(snapdir, snapnum=4):
    
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]
    
    FoF     = readfof.FoF_catalog(snapdir,snapnum,long_ids=False,
                                  swap=False,SFR=False,read_IDs=False)
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    print(pos_h.shape)
    vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s 
    mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h

    #part    = FoF.GroupLen        #number of particles in the halo
    #p_mass  = mass[0]/part[0]     #mass of a single particle in Msun/h
    #mass    = p_mass*(part*(1.0-part**(-0.6))) #corect FoF masses    #?????
    
    # Neglect halos with zero mass
    indexes = np.argwhere(mass>0.).reshape(-1)   # may not be needed for quijote?
    
    return mass, pos_h, vel_h, indexes

# Split training and validation sets
def split_datasets(dataset):

    random.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


# Correct periodic boundary effects
# Some halos close to a boundary could have subhalos at the other extreme of the box, due to periodic boundary conditions
# Just add or substract a length boxe in such cases to correct this artifact
def correct_boundary(pos, boxlength=1.):

    for i, pos_i in enumerate(pos):
        for j, coord in enumerate(pos_i):
            if coord > boxlength/2.:
                pos[i,j] -= boxlength
            elif -coord > boxlength/2.:
                pos[i,j] += boxlength

    return pos


# Main routine to load data and create the dataset
# simsuite: simulation suite, either "IllustrisTNG" or "SIMBA"
# simset: set of simulations:
#   CV: Use simulations with fiducial cosmological and astrophysical parameters, but different random seeds (27 simulations total)
#   LH: Use simulations over latin-hypercube, varying over cosmological and astrophysical parameters, and different random seeds (1000 simulations total)
# n_sims: number of simulations, maximum 27 for CV and 1000 for LH
def create_dataset(simsuite = "Illustris-3", simset = "", n_sims = 1):

    print("Using "+simsuite+" simulation suite, " +str(n_sims)+" simulations.")

    dataset = []
    subs = 0    # Number of subhalos

    for sim in range(n_sims):

        # To see ls of columns of file, type in shell: h5ls -r fof_subhalo_tab_033.hdf5
        #snapdir = simpath + str(sim)
        
        # Load the table of galactic features from a single simulation
        SubhaloPos = general_tab(simsuite)

        # For each halo in the simulation:
        if 1: #for ind in halolist:

            """
            # Select subhalos within a halo with index ind
            tab_halo = tab[tab[:,0]==ind][:,1:]

            # Consider only halos with at least one satellite galaxy (besides the central)
            if tab_halo.shape[0]>1:

                # In case you want to employ the stellar center of mass velocity:
                #velCMstar = np.dot(np.transpose(tab_halo[:,-3:]),tab_halo[:,3])/np.sum(tab_halo[:,3])

                # If galcen_frame==1, write positions and velocities in the rest frame of the central galaxy
                if galcen_frame:
                    tab_halo[:,:3] -= tab_halo[0,:3]
                    tab_halo[:,-3:] -= tab_halo[0,-3:]
                # Otherwise, write the positions and velocities as the relative position and velocity to the host halo
                else:
                    tab_halo[:,0:3] -= HaloPos[ind]
                    tab_halo[:,-3:] -= HaloVel[ind]

                # Correct periodic boundary effects
                tab_halo[:,:3] = correct_boundary(tab_halo[:,:3])
            """
            """
                # If use velocity, compute the modulus of the velocities and create a new table with these values
                if use_vel:
                    if galcen_frame:
                        subhalovel = np.log10(1.+np.sqrt(np.sum(tab_halo[:,-3:]**2., 1)))   # for this, use velnorm = 1. better
                    else:
                        subhalovel = np.log10(np.sqrt(np.sum(tab_halo[:,-3:]**2., 1)))  # use this way in case you normalize velocities
                    newtab = np.column_stack((tab_halo[:,:-3], subhalovel))
                else:
                    newtab = tab_halo[:,:-3]
            """
            
            newtab = SubhaloPos
            print(newtab.shape)
            # Take as global quantities of the halo the number of subhalos ###and the total stellar mass
            u = np.zeros((1,1), dtype=np.float32)
            u[0,0] = SubhaloPos.shape[0]  # number of halos
            #if not only_positions:
            #    u[0,1] = np.log10(np.sum(10.**tab_halo[:,3]))
            print(u.shape)
            # Create the graph of the halo
            # x: features (includes positions), pos: positions, u: global quantity
            graph = Data(x=torch.tensor(newtab[...,:2], dtype=torch.float32), pos=torch.tensor(SubhaloPos, dtype=torch.float32), y=torch.tensor(newtab[...,2], dtype=torch.float32), u=torch.tensor(u, dtype=torch.float32))

            # Update the total number of subhalos
            subs += graph.x.shape[0]

            dataset.append(graph)

    print("Total number of halos", len(dataset), "Total number of subhalos", subs)

    # Number of features
    node_features = graph.x.shape[1]   # should just be 2 (for x and y)... might want to input z too?
    print(node_features)
    return dataset, node_features
