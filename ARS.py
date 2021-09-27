import numpy as np
import os, sys
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt 
from scipy.spatial.transform import Rotation as Kabsch

#%matplotlib ipympl

home_path = "/home/boittier/Documents/AdjustReference-System/"
#home_path = "/home/eric/Documents/PhD/AdjustReference-System/"
sys.path.insert(1, home_path)
from Cube import read_charges_refined, read_cube

BOHR_TO_ANGSTROM = 0.529177

def usage():
    s = """

    Take the MDCM charges from a conformation in cubefile_1 and 
    return the position of the charges (in local, and new global coordinates) 
    for the second conformation

    ARS.py charges.xyz cubefile_1.cub cubefile_2.cub frames.txt
    
    """
    print(s)


def plot1():
    plot_labels = False
    plot_pos_1 = True
    plot_pos_2 = True
    plot_vectors = False

    fig = plt.figure()
    ax = Axes3D(fig, elev=0, azim=60)

    # Transpose to the right shape for plotting
    a_p = np.array(atom_positions).T
    a_p1 = np.array(atom_positions_plus).T
    c_p = np.array(c_positions).T
    c_p_l = np.array(c_positions_local).T
    c_p_g = np.array(c_positions_global).T

    # Plotting axes
    if plot_vectors:
        for frame_vec in frame_vectors:
            for il, local_vector_i in enumerate(frame_vec):
                atom_index = frame_atoms[0][il] - 1
                plot_axe(il, local_vector_i, atom_index)
       
    #  Plotting points
    if plot_pos_1:
        ax.plot(a_p[0], a_p[1], a_p[2], c='gray', linestyle = 'None', marker="o")
        ax.plot(c_p[0], c_p[1], c_p[2], marker="o", c='orange',linestyle = 'None', alpha=0.8)

        if plot_labels:
            label = ['{:d}'.format(i) for i in range(n_charges)]
            for i, pos in enumerate(c_p.T):
                ax.text(pos[0], pos[1], pos[2], label[i])

    # ax.plot(c_p_l[0], c_p_l[1], c_p_l[2], marker="o", c="g", linestyle = 'None', alpha=0.8)
    # label = ['{:d}'.format(i) for i in range(n_charges)]
    # for i, pos in enumerate(c_p_l.T):
    #     ax.text(pos[0], pos[1], pos[2], label[i])

    if plot_pos_2:
        ax.plot(a_p1[0], a_p1[1], a_p1[2], c='k',  linestyle = 'None', marker="o")
        ax.plot(c_p_g[0], c_p_g[1], c_p_g[2], marker="x", c="r", linestyle = 'None', alpha=0.8)
    
        if plot_labels:    
            label = ['{:d}'.format(i) for i in range(n_charges)]
            for i, pos in enumerate(c_p_g.T):
                ax.text(pos[0], pos[1], pos[2], label[i])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    if plot_labels:
        plt.legend()
    
    plt.show()


def save_charges(charge_positions, charges, filename="out_charges.xyz"):
    file = open(filename, "w")
    file.write("{}\n".format(len(charge_positions)))
    file.write("s                      x[A]                      y[A]                      z[A]                      q[e]\n")

    c = 1
    for xyz, q in zip(charge_positions, charges):
        print(c, xyz)
        c+=1
        if q < 0:
            letter = "O"
        else:
            letter = "N"
        file.write("{0:} {1:.16f} {2:.16f} {3:.16f} {4:.16f}\n".format(letter, xyz[0], 
            xyz[1],xyz[2],float(q)))
    file.close()



def plot_axe(il, local_vector, atom_index, c="k"):
    # print(il, local_vector, atom_index)
    atom_pos = atom_positions[atom_index]
    x = [atom_pos[0], atom_pos[0] + local_vector[0][0]]
    y = [atom_pos[1], atom_pos[1] + local_vector[0][1]]
    z = [atom_pos[2], atom_pos[2] + local_vector[0][2]]
    plt.plot(x, y, z, c='r', label="x")
    x = [atom_pos[0], atom_pos[0] + local_vector[1][0]]
    y = [atom_pos[1], atom_pos[1] + local_vector[1][1]]
    z = [atom_pos[2], atom_pos[2] + local_vector[1][2]]
    plt.plot(x, y, z, '--g', label="y")
    x = [atom_pos[0], atom_pos[0] + local_vector[2][0]]
    y = [atom_pos[1], atom_pos[1] + local_vector[2][1]]
    z = [atom_pos[2], atom_pos[2] + local_vector[2][2]]
    plt.plot(x, y, z, ':b', label="z")
    print("check for orthogality: ", np.dot(local_vector[2], local_vector[0]))

def get_local_axis(atom_pos, frame_atoms):
    """
    Inputs:
                atom_positions, frames
    Returns: 
                List of Lists of Frame Vectors [ [x_v, y_v, z_v], ...  ] in order of frames
    """
    n_frames = len(frame_atoms)
    frame_vectors = []
    for f in range(n_frames):
        a_index , b_index, c_index = frame_atoms[f]
        a,b,c = frame_atoms[f]
                     # adjust indexing
        a = atom_pos[a-1]
        b = atom_pos[b-1]
        c = atom_pos[c-1]
        distance_ab = distance.euclidean(a, b)
        b1_x = (a[0] - b[0])/distance_ab
        b1_y = (a[1] - b[1])/distance_ab
        b1_z = (a[2] - b[2])/distance_ab

        distance_bc = distance.euclidean(b, c)
        b2_x = (b[0] - c[0])/distance_bc
        b2_y = (b[1] - c[1])/distance_bc
        b2_z = (b[2] - c[2])/distance_bc

        #  Z axes
        ez1 = np.array([b1_x, b1_y, b1_z])
        ez2 = np.array([b1_x, b1_y, b1_z])
        ez3 = np.array([b2_x, b2_y, b2_z])
        
        #  Y axes
        ey1 = np.zeros(3)
        ey1[0] = b1_y*b2_z-b1_z*b2_y
        ey1[1] = b1_z*b2_x-b1_x*b2_z
        ey1[2] = b1_x*b2_y-b1_y*b2_x
        re_x = np.sqrt(ey1[0]**2 + ey1[1]**2 + ey1[2]**2)
        ey1[0] = ey1[0]/re_x
        ey1[1] = ey1[1]/re_x
        ey1[2] = ey1[2]/re_x

        #ey1 = np.cross(ez1, ez3) 

        ey2 = ey3 = ey1
          
        #  X axes
        ex1 = np.zeros(3)
        ex3 = np.zeros(3)
        #  ex1 and ex2
        ex1[0] = ez1[1]*ey1[2]-ez1[2]*ey1[1]
        ex1[1] = ez1[2]*ey1[0]-ez1[0]*ey1[2]
        ex1[2] = ez1[0]*ey1[1]-ez1[1]*ey1[0]
        re_x = np.sqrt(ex1[0]**2 + ex1[1]**2 + ex1[2]**2)
        ex1[0] = ex1[0]/re_x
        ex1[1] = ex1[1]/re_x
        ex1[2] = ex1[2]/re_x   
        ex2 = ex1
        ex1 = np.cross(ey1, ez1)
        ex2 = ex1
        
        #  ex3
        ex3[0] = ez3[1]*ey3[2]-ez3[2]*ey3[1]
        ex3[1] = ez3[2]*ey3[0]-ez3[0]*ey3[2]
        ex3[2] = ez3[0]*ey3[1]-ez3[1]*ey3[0]
        re_x = np.sqrt(ex3[0]**2 + ex3[1]**2 + ex3[2]**2)
        ex3[0] = ex3[0]/re_x
        ex3[1] = ex3[1]/re_x
        ex3[2] = ex3[2]/re_x  

        ex3 = np.cross(ey3, ez3) 

        frame_vectors.append(([ex1, ey1, ez1], 
                              [ex2, ey2, ez2], 
                              [ex3, ey3, ez3]))
    return frame_vectors


def read_cube_file(filepath):
    pcube_data, pcube_meta = read_cube(filepath)
    ap = []
    an = []
    for i in pcube_meta["atoms"]:
        atom = list(i[1])
        ap.append([x *BOHR_TO_ANGSTROM for x in atom[1:]])
        an.append(atom[1])
    return ap, an

def read_mdcm_xyz(filepath):
    xyz_file = open(filepath).readlines()
    n_charges = int(xyz_file[0])
    #  read number of charges from first line (xyz format)
    charge_lines = xyz_file[2:n_charges+2]
    # Read atoms and charges
    c_positions = []
    c_charges = []
    for charge in charge_lines:
        on, x, y, z, c = charge.split()
        c_positions.append([float(x), float(y) , float(z)])
        c_charges.append(float(c))
    return c_positions, c_charges


if __name__ == "__main__":
    """ 
    ARS.py charges.xyz cubefile_1.cub cubefile_2.cub frames.txt output_filename.xyz
    """
    xyz_file_name = sys.argv[1]
    pcube = sys.argv[2]
    pcube_2 = sys.argv[3]
    frame_file = sys.argv[4]
    output_filename = sys.argv[5] 
    # Open XYZ file
    c_positions, c_charges = read_mdcm_xyz(xyz_file_name)
    n_charges = len(c_charges)
    
    # Open Cube files
    atom_positions, atom_names = read_cube_file(pcube)
    atom_positions_plus, atom_names = read_cube_file(pcube_2)
    n_atoms = len(atom_names)

    # Match each charge to a nucleus
    charge_atom_associations = []
    atom_charge_dict = {}
    for i_charge in range(n_charges):
        #  initial distance, which can be compared to find smaller values
        min_distance = np.Inf 

        for j_atom in range(n_atoms):       
            d = distance.euclidean(c_positions[i_charge], atom_positions[j_atom])
            if d < min_distance:
                atom_association = j_atom
                min_distance = d
        
        charge_atom_associations.append([i_charge, atom_association])
        
        if atom_association not in list(atom_charge_dict.keys()):
            atom_charge_dict[atom_association] = [i_charge]
        else:
            atom_charge_dict[atom_association].append(i_charge)


    """
    Check that all atoms/charges are included in 
    """
    #  Atoms
    set1 = set(range(n_atoms))
    set2 = set(atom_charge_dict.keys())
    if set1 != set2:
        print(set1, set2)
        print("Something is wrong with Atoms?")
        #sys.exit()
    #  Charges
    set1 = set(range(n_charges))
    flat_list = []
    for sublist in list(atom_charge_dict.values()):
        for item in sublist:
            flat_list.append(item)
    set2 = set(flat_list)
    if set1 != set2:
        print(set1, set2)
        print("Something is wrong with Charges?")
        #sys.exit()
    print("atom_charge_dict: ", atom_charge_dict)
        
    # Get frames
    frame = open(frame_file).readlines()
    frame_atoms = []
    frames = frame[1:]
    n_frames = len(frames)
    for f in frames:
        a1, a2, a3 = f.split()
        frame_atoms.append([int(a1), int(a2), int(a3)])

    atom_positions = np.array(atom_positions)
    frame_vectors_plus = get_local_axis(atom_positions_plus, frame_atoms)

    # Calculate local axes and transform charges
    # Calculate the new axes for each frame
    frame_vectors = get_local_axis(atom_positions, frame_atoms)

    """
    Global ==> Local
    """
    #  Find the position of the charges in the local axes
    #  Create a new array for the 'local' charges
    c_pos_shape = np.array(c_positions).shape
    c_positions_local = np.zeros(c_pos_shape)
    
    used_atoms = []
    for f in range(n_frames):
        #  Loop through the atoms in the frame
        #  print(frame_atoms[f])
        for ai, atom_index in enumerate(frame_atoms[f]):
            #  print(atom_index-1)
            atom_index -= 1
            if atom_index in list(atom_charge_dict.keys()) and atom_index not in used_atoms:

                charges = atom_charge_dict[atom_index]
                ex, ey, ez = frame_vectors[f][ai]
                #  Find the associated charges for that atom, and loop
                for charge in charges:
                    c_pos_global = c_positions[charge]
                    atom_pos_xyz = atom_positions[atom_index]
                    
                    #  Find the distance between the charge and the atom it belongs to
                    r = np.array(c_pos_global) - np.array(atom_pos_xyz)
                    
                    local_x_pos =  np.dot(ex, r)
                    local_y_pos =  np.dot(ey, r)
                    local_z_pos =  np.dot(ez, r)

                    c_positions_local[charge][0] = local_x_pos
                    c_positions_local[charge][1] =  local_y_pos 
                    c_positions_local[charge][2] = local_z_pos
                    
            used_atoms.append(atom_index)


    """
    Local ==> Global
    """
    #  Find the position of the charges in the local axes
    #  Create a new array for the 'local' charges
    c_pos_shape = np.array(c_positions).shape
    c_new_local = np.zeros(c_pos_shape)
    c_positions_global = np.zeros(c_pos_shape)

    used_atoms = []
    for f in range(n_frames):
        #  Loop through the atoms in the frame
        for ai, atom_index in enumerate(frame_atoms[f]):
            print(atom_index-1)
            atom_index -= 1
            if atom_index in list(atom_charge_dict.keys()) and atom_index not in used_atoms:
                charges = atom_charge_dict[atom_index]
                ex, ey, ez = frame_vectors_plus[f][ai]
                # Find the associated charges for that atom, and loop
                for charge in charges:
                    c_pos_local = c_positions_local[charge]
                    atom_pos_xyz = atom_positions_plus[atom_index]

                    c_l_x = c_pos_local[0] 
                    c_l_y = c_pos_local[1] 
                    c_l_z = c_pos_local[2]
                                    
                    x_vec =  np.multiply(ex, c_l_x)
                    y_vec =  np.multiply(ey, c_l_y)
                    z_vec =  np.multiply(ez, c_l_z)

                    sum_of_components = x_vec + y_vec + z_vec
                    print(atom_index, charges, sum_of_components)
                    #  translate back to the center of atoms (for the new conformation)
                    c_positions_global[charge] = sum_of_components + atom_pos_xyz
                    
            used_atoms.append(atom_index)

    print("RMSD using the Kabsch algorithm: {}".format(Kabsch.align_vectors(c_positions_global, c_positions)[1]))

    save_charges(c_positions_global, c_charges, filename=output_filename)
    #plot1()
